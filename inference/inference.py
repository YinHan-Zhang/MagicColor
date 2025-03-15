import argparse
import os
import logging

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

import sys
sys.path.append("../utils")
from seed_all import seed_all
import matplotlib.pyplot as plt


from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    ControlNetModel,
    # UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer

from transformers import CLIPVisionModelWithProjection
from image_pair_edit_pipeline import ImagePairEditPipeline
from models.mutual_self_attention import ReferenceAttentionControl
from models.unet_2d_condition import UNet2DConditionModel
from models.refunet_2d_condition import RefUNet2DConditionModel

from setting_config import setting_configs



if __name__=="__main__":
    
    use_seperate = True
    logging.basicConfig(level=logging.INFO)
    
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run MonoDepth Estimation using Stable Diffusion."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model path from hugging face or local dir",
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
        "--checkpoint_path", type=str, help="Path to our trained checkpoint."
    )

    parser.add_argument(
        "--validation_raw1",
        type=str,
        default=None,
        help="valiadtion raw1 image",
    )
    parser.add_argument(
        "--validation_ref1",
        type=str,
        default=None,
        help="valiadtion ref1 image",
    )
    parser.add_argument(
        "--validation_raw2",
        type=str,
        default=None,
        help="valiadtion raw2 image",
    )
    parser.add_argument(
        "--validation_edit2",
        type=str,
        default=None,
        help="validation edit2 image",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=20,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=512,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 512.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 512.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )

    # other settings
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )

    parser.add_argument(
        "--setting_config",
        type=str,
        default="v00-naive-unet-edit2-raw2-refnet-ref1",
        help="setting name (default: v00-naive-unet-edit2-raw2-refnet-ref1)"
    )
    parser.add_argument(
        "--edge2_src_mode",
        type=str,
        default="raw",
        help="edge2 src mode (default: raw)"
    )
    
    args = parser.parse_args()
    
    if args.batch_size==0:
        args.batch_size = 1  # set default batchsize
    
    # -------------------- Preparation --------------------
    # Random seed
    if args.seed is None:
        import time
        args.seed = int(time.time())
    seed_all(args.seed)

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"output dir = {args.output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")
    
    
    # -------------------Data----------------------------
    logging.info("Inference Image Path from {}, {}, and {}".format(args.validation_ref1, args.validation_raw2, args.validation_edit2))

    # -------------------- Model --------------------
    if args.half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    setting_config = args.setting_config
    preprocessor = setting_configs[args.setting_config].get('preprocessor', None)
    in_channels_reference_unet = setting_configs[args.setting_config].get('in_channels_reference_unet', None)
    in_channels_denoising_unet = setting_configs[args.setting_config].get('in_channels_denoising_unet', None)
    in_channels_controlnet = setting_configs[args.setting_config].get('in_channels_controlnet', None)
    edge2_src_mode = args.edge2_src_mode

    if preprocessor == 'lineart':
        from src.annotator.lineart import BatchLineartDetector
        preprocessor = BatchLineartDetector(args.annotator_ckpts_path)
        preprocessor.to(device, dtype)
    else:
        preprocessor = None
    
    # declare a pipeline
    if not use_seperate:
        raise NotImplementedError
        # pipe = ImagePairEditPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=dtype, use_safetensors=True)
        # print("Using Completed")
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,subfolder='vae')
        scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler')
        denoising_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,subfolder="unet",
            in_channels=in_channels_denoising_unet,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            torch_dtype=dtype
        )

        denoising_unet_path = os.path.join(args.checkpoint_path, 'denoising_unet.pth')
        reference_unet_path = os.path.join(args.checkpoint_path, 'reference_unet.pth')
        controlnet_path = os.path.join(args.checkpoint_path, 'controlnet.pth')

        # load checkpoint
        denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False)
        print('denoising unet loaded.')

        if os.path.exists(reference_unet_path):
            reference_unet = RefUNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path,subfolder="unet",
                in_channels=in_channels_reference_unet,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
                torch_dtype=dtype
            )
            # refnet_tokenizer = CLIPTokenizer.from_pretrained(args.refnet_clip_vision_encoder_path)
            refnet_tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer')
            # refnet_text_encoder = CLIPTextModel.from_pretrained(args.refnet_clip_vision_encoder_path)
            refnet_text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder')
            refnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(args.refnet_clip_vision_encoder_path)
            reference_unet.load_state_dict(torch.load(reference_unet_path, map_location="cpu"))
            print('reference unet loaded.')
        else:
            reference_unet = None
            refnet_tokenizer = None
            refnet_text_encoder  = None
            refnet_image_enc = None
            print('no reference unet found.')

        if os.path.exists(controlnet_path):
            controlnet = ControlNetModel.from_pretrained(
                args.controlnet_model_name_or_path,
                in_channels=in_channels_controlnet,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True
            )
            controlnet_tokenizer = CLIPTokenizer.from_pretrained(args.controlnet_clip_vision_encoder_path)
            controlnet_text_encoder = CLIPTextModel.from_pretrained(args.controlnet_clip_vision_encoder_path)
            controlnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(args.controlnet_clip_vision_encoder_path)
            controlnet.load_state_dict(torch.load(controlnet_path, map_location="cpu"))
            print('controlnet loaded.')
        else:
            controlnet = None
            controlnet_tokenizer = None
            controlnet_text_encoder = None
            controlnet_image_enc = None
            print('no controlnet found.')
        
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
        print("Using Seperated Modules")
    
    logging.info("loading pipeline whole successfully.")
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    if device == 'cuda':
        generator = torch.cuda.manual_seed(args.seed)
    else:
        generator = torch.manual_seed(args.seed)

    with torch.no_grad():
        os.makedirs(args.output_dir, exist_ok=True)
        # load the example image.
        raw1 = Image.open(args.validation_raw1)
        ref1 = Image.open(args.validation_ref1)
        raw2 = Image.open(args.validation_raw2)
        edit2 = Image.open(args.validation_edit2)
        pipe_out = pipe(
            ref1,
            raw2,
            edit2,
            denosing_steps=args.denoising_steps,
            processing_res=args.processing_res,
            match_input_res=not args.output_processing_res,
            batch_size=args.batch_size,
            show_progress_bar=True,
            guidance_scale=args.guidance_scale,
            setting_config=args.setting_config,
            edge2_src_mode=args.edge2_src_mode,
            preprocessor=preprocessor,
            generator=generator,
        )

        to_save_dict = pipe_out.to_save_dict
        to_save_dict['raw1'] = raw1
        to_save_dict['pred2'] = pipe_out.img_pil
        for image_name, image in to_save_dict.items():
            image_save_path = os.path.join(args.output_dir, f'{image_name}.png')
            if os.path.exists(image_save_path):
                logging.warning(
                    f"Existing file: '{image_save_path}' will be overwritten"
                )
            image.save(image_save_path)

