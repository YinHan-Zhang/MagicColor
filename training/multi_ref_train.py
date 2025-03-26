import sys
sys.path.append("../training")
sys.path.append("../src")
import argparse
import math
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision.ops import masks_to_boxes

import os
import logging
import tqdm
import time

from accelerate import Accelerator
import transformers
import numpy as np
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
import shutil

from accelerate.utils import DistributedDataParallelKwargs
import diffusers
from diffusers import (
    ControlNetModel,
    MultiControlNetModel,
    DDIMScheduler,
    AutoencoderKL,
)

from diffusers.optimization import get_scheduler



from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import accelerate
import cv2
import numpy as np

from dataset_configuration import prepare_dataset,resize_max_res_tensor
from inference.image_pair_edit_pipeline_multi_ref import ImagePairEditPipeline

from src.models.mutual_self_attention_multi_scale import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.refunet_2d_condition import RefUNet2DConditionModel
from src.models.dino_model import FrozenDinoV2Encoder


from PIL import Image
from transformers import CLIPVisionModelWithProjection
from setting_config import setting_configs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


logger = get_logger(__name__, log_level="INFO")

def log_validation(
    args,
    test_loader,
    vae,
    net,
    refnet_tokenizer,
    refnet_text_encoder,
    refnet_image_enc,
    controlnet_tokenizer,
    controlnet_text_encoder,
    controlnet_image_enc,
    scheduler,
    accelerator,
    outputdir,
    iters,
    setting_config,
    edge2_src_mode,
    preprocessor,
    weight_dtype,
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
   
    if ori_net.reference_unet:
        reference_unet = copy.deepcopy(ori_net.reference_unet)
    else:
        reference_unet = None
    if ori_net.controlnet:
        controlnet = copy.deepcopy(ori_net.controlnet)
    else:
        controlnet = None
    denoising_unet = copy.deepcopy(ori_net.denoising_unet)

    if accelerator.device.type == 'cuda':
        generator = torch.cuda.manual_seed(42)
    else:
        generator = torch.Generator().manual_seed(42)
    
    pipe = ImagePairEditPipeline(
        reference_unet=reference_unet,
        controlnet=controlnet,
        denoising_unet=denoising_unet,  
        vae=vae,
        refnet_tokenizer=refnet_tokenizer,
        refnet_text_encoder=refnet_text_encoder,
        refnet_image_enc=refnet_image_enc,
        controlnet_tokenizer=controlnet_tokenizer,
        controlnet_text_encoder=controlnet_text_encoder,
        controlnet_image_enc=controlnet_image_enc,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device, weight_dtype)
    
    save_path = os.path.join(outputdir, f"{iters:06d}")
    os.makedirs(save_path, exist_ok=True)

    pipe_outs = pipe(
        dataloader=test_loader,
        dino_encoder=dino_encoder,
        denosing_steps=20,
        processing_res=512,
        match_input_res=False,
        batch_size=1,
        show_progress_bar=True,
        guidance_scale=9,
        setting_config=setting_config,
        edge2_src_mode=edge2_src_mode,
        preprocessor=preprocessor,
        generator=generator,
        weight_dtype=weight_dtype,
    )
    for index, pipe_out in enumerate(pipe_outs):
        to_save_dict = pipe_out.to_save_dict
        to_save_dict['pred2'] = pipe_out.img_pil
            
        import cv2
        images_tensor = []
    
        for image_name, image in to_save_dict.items():
            ## save by grid
            if image_name in ['ref1','raw2','edge2','pred2']:
                image_np = np.array(image)
                image_tensor = torch.tensor(image_np, dtype=torch.uint8)
                image = torch.cat([image_tensor], 1)
                images_tensor.append(image)
                
            gen_img = torch.cat(images_tensor, 1)
            gen_img_np = cv2.cvtColor(gen_img.cpu().numpy(), cv2.COLOR_RGB2BGR)
            image_save_path = os.path.join(save_path, f'output_{index}.png')
            cv2.imwrite(
                image_save_path, gen_img_np
            )

    del pipe
    torch.cuda.empty_cache()

def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = os.path.join(save_dir, f"checkpoint-{ckpt_num:06d}/{prefix}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)
    
    if prefix == 'controlnet':
        model.save_pretrained(save_dir + f"/checkpoint-{ckpt_num:06d}/controlnet")
    else:
        state_dict = model.state_dict()
        torch.save(state_dict, save_path)

class Net(nn.Module):
    def __init__(
        self,
        reference_unet: RefUNet2DConditionModel,
        controlnet: MultiControlNetModel,
        denoising_unet: UNet2DConditionModel,
        reference_control_writer,
        reference_control_reader,
        point_net = None
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.controlnet = controlnet
        self.denoising_unet = denoising_unet
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.point_net=point_net

    def forward(
        self,
        unet_input,
        timesteps,
        refnet_input,
        refnet_image_prompt_embeds,
        controlnet_inputs,
        controlnet_image_prompt_embeds,
        uncond_fwd: bool = False,
        uncond_control: bool = False,
        ref_ft = None
    ):
        down_block_res_samples, mid_block_res_sample = None, None
        ref_ft = None
     
        if not uncond_fwd:
            if self.reference_unet:
                ref_timesteps = torch.zeros_like(timesteps)
                ref_ft_all = self.reference_unet(
                    refnet_input,
                    ref_timesteps,
                    encoder_hidden_states=refnet_image_prompt_embeds,
                    return_dict=True,
                ).output
                up_ft_index = 1
                ref_ft = ref_ft_all['up_ft'][up_ft_index] 
                ref_ft = ref_ft.mean(0, keepdim=True)
                
               
                self.reference_control_reader.update(self.reference_control_writer)
                
        if self.controlnet:
            noisy_latents, sketch_cond , controlnet_image = controlnet_inputs

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=controlnet_image_prompt_embeds,
                controlnet_cond=[sketch_cond, controlnet_image],
                return_dict=False,
            )
        
        if uncond_control:
            model_pred = self.denoising_unet(
                unet_input,
                timesteps,
                encoder_hidden_states=refnet_image_prompt_embeds,
                ref_ft=ref_ft
            ).sample
        else:
            model_pred = self.denoising_unet(
                unet_input,
                timesteps,
                encoder_hidden_states=refnet_image_prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                ref_ft=ref_ft
            ).sample
        return model_pred
    
def collate_fn(examples):
    ref1 = torch.stack([example['img1'] for example in examples]).float()
    raw2 = torch.stack([example['img2'] for example in examples]).float()
    clip_values = torch.stack([example['clip_images'] for example in examples]).float() # float32

    patches = [example['patches'] for example in examples]
    ctrl_img = torch.stack([example['ctrl_img'] for example in examples]).float()

    batch = {
        'ref1': ref1,
        'raw2': raw2,
        'clip_images': clip_values,
        'patches': patches,
        'ctrl_img': ctrl_img,
        'extract': batch["extract"]
    }
    return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation")
    
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--refnet_clip_vision_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_clip_vision_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--annotator_ckpts_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_ckpt_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="webvid",
        required=True,
        help="Specify the dataset name used for training/validation.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data",
        required=True,
        help="The Root Dataset Path.",
    )
    parser.add_argument(
        "--trainlist",
        type=str,
        default="",
        required=True,
        help="train file listing the training files",
    )
    parser.add_argument(
        "--vallist",
        type=str,
        default=None,
        help="val file listing the validation files",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=336,
        required=True,
        help="height of training images"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=596,
        required=True,
        help="width of training images"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        required=True,
        help="stride of training video"
    )
    
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="saved_models",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--recom_resolution",
        type=int,
        default=512,
        help=(
            "The resolution for resizeing the input images and the depth/disparity to make full use of the pre-trained model from \
                from the stable diffusion vae, for common cases, do not change this parameter"
        ),
    )
    #TODO : Data Augmentation
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=70)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )

    # using EMA for improving the generalization
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    # dataloaderes
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # how many steps csave a checkpoints
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Validation steps",
    )
    parser.add_argument(
        "--no_eval_before_train",
        action="store_true",
        help="Whether not to do evalation before training."
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    
    # noise offset?::: #TODO HERE
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    
    # validations every 5 Epochs
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        "--validation_ref1",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_raw2",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_edit2",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_point_ref",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_point_main",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--setting_config",
        type=str,
        default="v06-lineart_anime-unet-edit2-refnet-ref1-controlnet-lineart-edge2",
        help="setting name (default: v00-naive-unet-edit2-raw2-refnet-ref1)"
    )
    parser.add_argument(
        "--edge2_src_train_mode",
        type=str,
        default="edit",
        help="setting name (default: edit), (available: raw, edit)"
    )
    parser.add_argument(
        "--edge2_src_eval_mode",
        type=str,
        default="raw",
        help="setting name (default: raw), (available: raw, edit)"
    )

    # get the local rank
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.dataset_path is None:
        raise ValueError("Need either a dataset name or a DataPath.")

    return args
    
    
def main():
    
    ''' ------------------------Configs Preparation----------------------------'''
    # give the args parsers
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    args = parse_args()

    output_dir = os.path.join(args.output_root, args.tracker_project_name)
    # save the tensorboard log files
    logging_dir = os.path.join(output_dir, args.logging_dir)

    # tell the gradient_accumulation_steps, mix precison, and tensorboard
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True) # only the main process show the logs
    # set the warning levels
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Doing I/O at the main proecss
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    
    ''' ------------------------Non-NN Modules Definition----------------------------'''
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler')
    logger.info("loading the noise scheduler from {}".format(args.pretrained_model_name_or_path),main_process_only=True)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    
    # channel setting
    setting_config = args.setting_config
    preprocessor = setting_configs[args.setting_config].get('preprocessor', 'lineart_anime')
    in_channels_reference_unet = setting_configs[args.setting_config].get('in_channels_reference_unet', 4)
    in_channels_denoising_unet = setting_configs[args.setting_config].get('in_channels_denoising_unet', 4)
    in_channels_controlnet = setting_configs[args.setting_config].get('in_channels_controlnet', 4)

    # load sketch preprocessor
    if preprocessor == 'lineart':
        from src.annotator.lineart import BatchLineartDetector
        preprocessor = BatchLineartDetector(args.annotator_ckpts_path)
    elif preprocessor == 'lineart_anime':
        from src.annotator.lineart_anime import  LineartAnimeDetector
        preprocessor =  LineartAnimeDetector(args.annotator_ckpts_path)
        print("lineart_anime preprocessor is loaded.") 
    else:
        preprocessor = None

    # load model weight
    if args.checkpoint_path:
        denoising_unet_ckpt_path = os.path.join(args.checkpoint_path, 'denoising_unet.pth')
        reference_unet_ckpt_path = os.path.join(args.checkpoint_path, 'reference_unet.pth')
    else:
        denoising_unet_ckpt_path=None
        reference_unet_ckpt_path=None

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):

        vae = AutoencoderKL.from_pretrained(
            # args.pretrained_model_name_or_path,
            "../ckpt/sd-vae-ft-mse",
            # subfolder='vae',
            use_safetensors=False
        )

        denoising_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            in_channels=in_channels_denoising_unet,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            use_safetensors=False
        )
        if denoising_unet_ckpt_path:
            denoising_unet.load_state_dict(torch.load(denoising_unet_ckpt_path, map_location="cpu"), strict=False)

        
        assert in_channels_reference_unet and in_channels_denoising_unet
        if in_channels_reference_unet and in_channels_denoising_unet:
            reference_unet = RefUNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path,subfolder="unet",
                in_channels=in_channels_reference_unet,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
                use_safetensors=False
            )
            refnet_tokenizer = CLIPTokenizer.from_pretrained(args.refnet_clip_vision_encoder_path)
            refnet_text_encoder = CLIPTextModel.from_pretrained(args.refnet_clip_vision_encoder_path)
            refnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(args.refnet_clip_vision_encoder_path)
            if reference_unet_ckpt_path:
                reference_unet.load_state_dict(torch.load(reference_unet_ckpt_path, map_location="cpu"), strict=False)
        else:
            reference_unet = None
            refnet_tokenizer = None
            refnet_text_encoder = None
            refnet_image_enc = None

        if in_channels_controlnet:
            # load model weight
            if args.controlnet_ckpt_path:
                controlnet_sketch_ckpt_path = args.controlnet_ckpt_path + f"/controlnet_sketch"
                controlnet_multi_ckpt_path = args.controlnet_ckpt_path + f"/controlnet_multi"

                controlnet_multi = ControlNetModel.from_pretrained(
                    controlnet_multi_ckpt_path,
                    in_channels=in_channels_controlnet,
                    low_cpu_mem_usage=False,
                    ignore_mismatched_sizes=True,
                    # use_safetensors=False
                )
                controlnet_sketch = ControlNetModel.from_pretrained(
                    controlnet_sketch_ckpt_path,
                    in_channels=in_channels_controlnet,
                    low_cpu_mem_usage=False,
                    ignore_mismatched_sizes=True,
                    # use_safetensors=False
                )
                controlnet = MultiControlNetModel(controlnets=[controlnet_sketch, controlnet_multi])
                print(f"load controlnet from {controlnet_sketch_ckpt_path} success !!!")
            else:
                import json
                with open('./ckpt/controlnet/config.json', "r") as f:
                    config = json.load(f)
                controlnet_multi = ControlNetModel(**config)

                controlnet_sketch = ControlNetModel.from_pretrained(
                    args.controlnet_model_name_or_path,
                    in_channels=in_channels_controlnet,
                    low_cpu_mem_usage=False,
                    ignore_mismatched_sizes=True,
                    # use_safetensors=False
                )
                controlnet = MultiControlNetModel(controlnets=[controlnet_sketch, controlnet_multi])
                print(f"load controlnet from origin ...")

            controlnet_tokenizer = CLIPTokenizer.from_pretrained(args.controlnet_clip_vision_encoder_path)
            controlnet_text_encoder = CLIPTextModel.from_pretrained(args.controlnet_clip_vision_encoder_path)
            controlnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(args.controlnet_clip_vision_encoder_path)
        else:
            controlnet = None
            controlnet_tokenizer = None
            controlnet_text_encoder = None
            controlnet_image_enc = None
                  
    # Freeze vae and text and image encoder
    vae.requires_grad_(False)
    # set ref unet to trainable.
    if reference_unet:
        refnet_text_encoder.requires_grad_(False)
        refnet_image_enc.requires_grad_(False)
        for name, param in reference_unet.named_parameters():
            if "up_blocks.3" in name: 
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
    # freeze controlnet 
    if controlnet:
        controlnet.requires_grad_(True)
        controlnet.nets[0].requires_grad_(True)
        controlnet.nets[1].requires_grad_(True)
        controlnet_text_encoder.requires_grad_(False)
        controlnet_image_enc.requires_grad_(False) 
    # set denosie unet to trainable.
    denoising_unet.requires_grad_(True)

    if reference_unet:
        reference_control_writer = ReferenceAttentionControl(
            reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )
    else:
        reference_control_writer = None
        reference_control_reader = None

    net = Net(
        reference_unet,
        controlnet,
        denoising_unet,
        reference_control_writer,
        reference_control_reader,
    )

    # using checkpint  for saving the memories
    if args.gradient_checkpointing:
        if reference_unet:
            reference_unet.enable_gradient_checkpointing()
        if controlnet:
            controlnet.nets[0].enable_gradient_checkpointing()
            controlnet.nets[1].enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()
        
    # how many cards did we use: accelerator.num_processes
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # optimizer settings
    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # load dataset
    with accelerator.main_process_first():
        (train_loader, test_loader), dataset_config_dict = prepare_dataset(
            dataset_name=args.dataset_name,
            datapath=args.dataset_path,
            trainlist=args.trainlist,
            vallist=args.vallist,
            height=args.height,
            width=args.width,
            stride=args.stride,
            batch_size=args.train_batch_size,
            test_batch=1,
            datathread=args.dataloader_num_workers,
            logger=logger,
            # collate_fn=collate_fn,
        )

    # because the optimizer not optimized every time, so we need to calculate how many steps it optimizes,
    # it is usually optimized by 
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    net, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        net, optimizer, train_loader, lr_scheduler
    )

    # scale factor.
    rgb_latent_scale_factor = 0.18215

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    else:
        weight_dtype = torch.float32
        args.mixed_precision = accelerator.mixed_precision

    # weight_dtype = torch.float32
    print(weight_dtype)
    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    if reference_unet:
        reference_unet.to(accelerator.device, dtype=weight_dtype)
        refnet_text_encoder.to(accelerator.device, dtype=weight_dtype)
        refnet_image_enc.to(accelerator.device, dtype=weight_dtype)
    if controlnet:
        controlnet.to(accelerator.device, dtype=weight_dtype)
        controlnet_text_encoder.to(accelerator.device, dtype=weight_dtype)
        controlnet_image_enc.to(accelerator.device, dtype=weight_dtype)

    denoising_unet.to(accelerator.device, dtype=weight_dtype)

    if preprocessor is not None:
        preprocessor.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)


    # Here is the DDP training: actually is 4
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    output_val_dir = f"{output_dir}/multi_ref_validation"
    os.makedirs(output_val_dir, exist_ok=True)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    global dino_encoder
    dino_encoder = FrozenDinoV2Encoder()
    dino_encoder.to(accelerator.device, dtype=weight_dtype)

    import lpips
    loss_fn = lpips.LPIPS(net='vgg').to(accelerator.device, dtype=weight_dtype)

    # using the epochs to training the model
    for epoch in range(first_epoch, args.num_train_epochs):
        net.train() 
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(net):
                
                ref1 = batch['img1'].to(weight_dtype)
                raw2 = batch['img2'].to(weight_dtype)

                curr_h, curr_w = batch["img1"].shape[-2:]
                ctrl_channel = 1024
                patch_size = 504
                device = 'cuda'
                prompt_fea = torch.zeros((*batch["ctrl_img"][0].shape, ctrl_channel)).to(
                    device, dtype=weight_dtype
                )

                for curr_b, curr_ins_img in enumerate(batch["patches"]):
                    curr_ins_id, curr_ins_patch = curr_ins_img[0], curr_ins_img[1].to(prompt_fea)
                    
                    curr_ins_patch = curr_ins_patch.squeeze(0)

                    if curr_ins_id.shape[0] > 0:
                        with torch.cuda.amp.autocast():
                            image_features = dino_encoder(curr_ins_patch)
                            image_features = controlnet.nets[1].dino_adapter(image_features).to(
                                prompt_fea
                            )
                       
                        for id_ins, curr_ins in enumerate(curr_ins_id.tolist()[0]):
                            all_vector = image_features[id_ins] 
                            global_vector = all_vector[0:1] 

                            down_s = patch_size // 14

                            patch_vector = (
                                all_vector[1 : 1 + down_s * down_s]
                                .view(1, down_s, down_s, -1)
                                .permute(0, 3, 1, 2)
                            )

                            curr_mask = batch["ctrl_img"][curr_b] == curr_ins 
                            curr_mask = curr_mask.squeeze(0)
                            
                            if curr_mask.max() < 1:
                                continue
                            
                            curr_box = masks_to_boxes(curr_mask[None])[0].int().tolist() 
                            
                            height, width = (
                                curr_box[3] - curr_box[1],
                                curr_box[2] - curr_box[0],
                            )

                            x = torch.linspace(-1, 1, height)
                            y = torch.linspace(-1, 1, width)

                            xx, yy = torch.meshgrid(x, y)
                            grid = torch.stack((xx, yy), dim=2).to(patch_vector)[None] 
                            
                            
                            warp_fea = F.grid_sample(
                                patch_vector,
                                grid,
                                mode="bilinear",
                                padding_mode="reflection",
                                align_corners=True,
                            )[0].permute(1, 2, 0)
                            
                            
                            small_mask = curr_mask[
                                curr_box[1] : curr_box[3], curr_box[0] : curr_box[2]
                            ]

                            curr_pix_num = small_mask.sum().item()
                            all_ins = np.arange(0, curr_pix_num)
                            sel_ins = np.random.choice(
                                all_ins,size=(curr_pix_num // 10,),replace=True,)
                            warp_fea[small_mask][sel_ins] = global_vector

                            prompt_fea[curr_b][
                                curr_box[1] : curr_box[3], curr_box[0] : curr_box[2]
                            ][small_mask] = warp_fea[small_mask]

             
                batch["conditioning_pixel_values"] = prompt_fea.permute(0, 3, 1, 2)

                controlnet_image = batch["conditioning_pixel_values"].to(
                    device, dtype=weight_dtype
                )

                # resize 
                ref1_resized = resize_max_res_tensor(ref1, recom_resolution=512) 
                raw2_resized = resize_max_res_tensor(raw2, recom_resolution=512)

                # encode RGB to lantents
                ref1_latents = vae.encode(ref1_resized).latent_dist.sample()
                ref1_latents = ref1_latents * rgb_latent_scale_factor

                raw2_latents = vae.encode(raw2_resized).latent_dist.sample()
                raw2_latents = raw2_latents * rgb_latent_scale_factor

                edge1_src_resized = ref1_resized
                edge2_src_resized = raw2_resized
                
                if batch["extract"]:
                    edge2_resized = preprocessor(edge2_src_resized)
                    edge2_resized[edge2_resized <= 0.25] = 0
                    edge2_resized = edge2_resized.repeat(1, 3, 1, 1) * 2 - 1.
                    edge2_resized = edge2_resized.to(device, dtype=weight_dtype)
                else:
                    edge2_resized = edge2_src_resized.to(device, dtype=weight_dtype)
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(raw2_latents) # create noise
                bsz = raw2_latents.shape[0]
                
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=raw2_latents.device)
                timesteps = timesteps.long()
                
                # add noise to the depth lantents
                noisy_raw2_latents = noise_scheduler.add_noise(raw2_latents, noise, timesteps)
                
                # uncond：
                uncond_fwd = False  
                uncond_control = random.random() < 0.1 

                # Get condition
                if uncond_fwd: # text
                    prompt = ""
                    def prompt2embeds(prompt, tokenizer, text_encoder):                    
                        text_inputs = tokenizer(
                            prompt,
                            padding="do_not_pad",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_input_ids = text_inputs.input_ids.to(text_encoder.device) #[1,2]
                        empty_text_embed = text_encoder(text_input_ids)[0].to(weight_dtype)
                        image_prompt_embeds = empty_text_embed.repeat((noisy_raw2_latents.shape[0], 1, 1))[:,0,:].unsqueeze(1)  # [B, 1, 1024]
                        return image_prompt_embeds
                    if reference_unet:
                        refnet_image_prompt_embeds = prompt2embeds(prompt, refnet_tokenizer, refnet_text_encoder)
                    else:
                        refnet_image_prompt_embeds = None

                    if controlnet:
                        controlnet_image_prompt_embeds = prompt2embeds(prompt, controlnet_tokenizer, controlnet_text_encoder)
                    else:
                        controlnet_image_prompt_embeds = None
                else:
                   
                    clip_img = batch["clip_images"]
                    def img2embeds(clip_image, image_enc):
                        clip_image_embeds = image_enc(
                            clip_image.to(device, dtype=image_enc.dtype)
                        ).image_embeds
                        encoder_hidden_states = clip_image_embeds.unsqueeze(1)
                        return encoder_hidden_states
                    if reference_unet:
                        refnet_image_prompt_embeds = img2embeds(clip_img, refnet_image_enc)
                    else:
                        refnet_image_prompt_embeds = None

                    if controlnet:
                        controlnet_image_prompt_embeds = img2embeds(clip_img, refnet_image_enc)
                    else:
                        controlnet_image_prompt_embeds = None

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(raw2_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # model input
                refnet_input = ref1_latents
                controlnet_inputs = (noisy_raw2_latents, edge2_resized, controlnet_image)
                unet_input = torch.cat([noisy_raw2_latents], dim=1)

                # predict the noise residual
                noise_pred = net(
                    unet_input,
                    timesteps,
                    refnet_input,
                    refnet_image_prompt_embeds,
                    controlnet_inputs,
                    controlnet_image_prompt_embeds,
                    uncond_fwd,
                    uncond_control
                )
                noise_scheduler.set_timesteps(num_inference_steps=1, device=raw2_latents.device)
                noisy_edit2_latents = noise_scheduler.step(noise_pred, 1, noisy_raw2_latents).prev_sample

                rgb_latent = noisy_edit2_latents / rgb_latent_scale_factor
                rgb_out = vae.decode(rgb_latent, return_dict=False)[0]

                perceptual_loss = loss_fn.forward(raw2_resized.to(raw2_latents.device), rgb_out.to(raw2_latents.device))[0][0][0][0].to(raw2_latents.device)

                def weighted_mse_loss(input, target, weight):
                    return torch.mean(weight * (input - target) ** 2)

                fore_value = (
                    -0.5 * (1 + np.cos(np.pi * global_step / args.max_train_steps))
                    + 2.0
                )
                
                edge_w = torch.cat([x[2] for x in batch["patches"]], 0)
                weight_mask = torch.ones_like(edge_w).to(weight_dtype)

                weight_mask[edge_w != 0] *= fore_value

                loss = weighted_mse_loss(
                    noise_pred.float(),
                    target.float(),
                    weight_mask.unsqueeze(1).repeat(1, target.shape[1], 1, 1),
                )
                
                loss = loss + 0.1 * perceptual_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # currently the EMA is not used.
            if accelerator.sync_gradients:
                if reference_unet:
                    reference_control_reader.clear()
                    reference_control_writer.clear()

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                # validation and save
                if accelerator.is_main_process:
                    # validations
                    if (global_step - 1) % args.validation_steps == 0:
                        net.eval()
                        edge2_src_eval_mode='raw'
                        log_validation(
                            args=args,
                            test_loader=test_loader,
                            vae=vae,
                            net=net,
                            refnet_tokenizer=refnet_tokenizer,
                            refnet_text_encoder=refnet_text_encoder,
                            refnet_image_enc=refnet_image_enc,
                            controlnet_tokenizer=controlnet_tokenizer,
                            controlnet_text_encoder=controlnet_text_encoder,
                            controlnet_image_enc=controlnet_image_enc,
                            scheduler=noise_scheduler,
                            accelerator=accelerator,
                            outputdir=output_val_dir,
                            iters=global_step,
                            setting_config=setting_config,
                            edge2_src_mode=edge2_src_eval_mode,
                            preprocessor=preprocessor,
                            weight_dtype=weight_dtype,
                        )

                    # saving the checkpoints
                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            start_time = time.time()
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                                
                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)
                            # save whole weight
                            save_path = os.path.join(output_dir, f"checkpoint-{global_step:06d}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {output_dir}")

                            if global_step % args.checkpointing_steps == 0:
                                unwrap_net = accelerator.unwrap_model(net)
                                if reference_unet:
                                    save_checkpoint(
                                        unwrap_net.reference_unet,
                                        output_dir,
                                        "reference_unet",
                                        global_step,
                                        total_limit=args.checkpoints_total_limit,
                                    )
                                if controlnet:
                                    save_checkpoint(
                                        unwrap_net.controlnet,
                                        output_dir,
                                        "controlnet",
                                        global_step,
                                        total_limit=args.checkpoints_total_limit,
                                    )

                                save_checkpoint(
                                    unwrap_net.denoising_unet,
                                    output_dir,
                                    "denoising_unet",
                                    global_step,
                                    total_limit=args.checkpoints_total_limit,
                                )
                        
                            current_time = time.time()
                            duration = current_time - start_time
                            print(f'checkpoint saving: duration = {duration:.4f}s.')

            logs = {"step_loss": loss.detach().item(),"lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Stop training
            if global_step >= args.max_train_steps:
                break
        
    # Create the pipeline for training and savet
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__=="__main__":
    main()



