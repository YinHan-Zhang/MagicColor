import argparse
import os
import logging
import pickle
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

import sys
sys.path.append("../../MagicColor")
sys.path.append("../training")
sys.path.append("../src")
sys.path.append("../utils")

from seed_all import seed_all, load_seed
import matplotlib.pyplot as plt

from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    ControlNetModel,
    MultiControlNetModel,
    # UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer

from transformers import CLIPVisionModelWithProjection
from models.dino_model import FrozenDinoV2Encoder

from image_pair_edit_pipeline_multi_ref import ImagePairEditPipeline
from models.mutual_self_attention import ReferenceAttentionControl
from models.unet_2d_condition import UNet2DConditionModel
from models.refunet_2d_condition import RefUNet2DConditionModel
from annotator.lineart_anime import LineartAnimeDetector

from setting_config import setting_configs
from torch.utils.data import DataLoader, Dataset
from dataloader.inference_loader import InferPairDataset
from dataloader.image_multipair_loader_add import PairDataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run MonoDepth Estimation using Stable Diffusion."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--refnet_clip_vision_encoder_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_clip_vision_encoder_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--annotator_ckpts_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None ,help="Path to our trained checkpoint."
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
        "--output_dir", type=str, required=False, help="Output directory."
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
        default="default",
        help="" # training.setting_config
    )
    parser.add_argument(
        "--edge2_src_mode",
        type=str,
        default="raw",
        help="edge2 src mode (default: raw)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="inference data dir"
    )
    
    args = parser.parse_args()
    
    if args.batch_size==0:
        args.batch_size = 1  # set default batchsize
    
    return args


if __name__=="__main__":
    
    use_seperate = True
    logging.basicConfig(level=logging.INFO)
    
    args = parse_args()
    
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
    

    # -------------------- Model --------------------
    if args.half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    preprocessor = "lineart_anime"
    in_channels_reference_unet = 4
    in_channels_denoising_unet = 4
    in_channels_controlnet = 4

    edge2_src_mode = args.edge2_src_mode

    if preprocessor == 'lineart':
        from src.annotator.lineart import BatchLineartDetector
        preprocessor = BatchLineartDetector(args.annotator_ckpts_path)
        preprocessor.to(device, dtype)
        print("lineart preprocessor is loaded.")
    elif preprocessor == 'lineart_anime':
        preprocessor = LineartAnimeDetector(args.annotator_ckpts_path)
        preprocessor.to(device, dtype)
        print("lineart_anime preprocessor is loaded.")     
    else:
        preprocessor = None
        print("preprocessor not load !!!")
    
    # declare a pipeline
    if not use_seperate:
        raise NotImplementedError
    else:
        vae = AutoencoderKL.from_pretrained(
            "ckpt/sd-vae-ft-mse",
            use_safetensors=False
            )
        scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler', use_safetensors=False)
        denoising_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            in_channels=in_channels_denoising_unet,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            torch_dtype=dtype, 
            use_safetensors=False
        )
        denoising_unet_path = os.path.join(args.checkpoint_path, 'denoising_unet.pth')
        reference_unet_path = os.path.join(args.checkpoint_path, 'reference_unet.pth')
        

        # load checkpoint
        denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False)
        print('denoising unet is loaded.')
        
        # load ref unet
        if os.path.exists(reference_unet_path):
            reference_unet = RefUNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="unet",
                in_channels=in_channels_reference_unet,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
                torch_dtype=dtype, 
                use_safetensors=False
            )
            
            refnet_tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer', use_safetensors=False)
            refnet_text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder', use_safetensors=False)
            refnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(args.refnet_clip_vision_encoder_path, use_safetensors=False)
            reference_unet.load_state_dict(torch.load(reference_unet_path, map_location="cpu"), strict=False)
            print('reference unet loaded.')
        else:
            reference_unet = None
            refnet_tokenizer = None
            refnet_text_encoder  = None
            refnet_image_enc = None
            print('no reference unet found.')
        
        # load controlnet
        if not os.path.exists(args.checkpoint_path):
            import json
            with open('/ckpt/controlnet/config.json', "r") as f:
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
            controlnet_tokenizer = CLIPTokenizer.from_pretrained(args.controlnet_clip_vision_encoder_path, use_safetensors=False)
            controlnet_text_encoder = CLIPTextModel.from_pretrained(args.controlnet_clip_vision_encoder_path, use_safetensors=False)
            controlnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(args.controlnet_clip_vision_encoder_path, use_safetensors=False)
            print('not pretrained controlnet loaded.')
        else:
            controlnet_multi_path = os.path.join(args.checkpoint_path, 'controlnet_multi')
            controlnet_multi = ControlNetModel.from_pretrained(
                controlnet_multi_path,
                in_channels=in_channels_controlnet,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True
            )
            controlnet_sketch_path = os.path.join(args.checkpoint_path, 'controlnet_sketch')
            controlnet_sketch = ControlNetModel.from_pretrained(
                controlnet_sketch_path,
                in_channels=in_channels_controlnet,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True
            )
            controlnet = MultiControlNetModel(controlnets=[controlnet_sketch, controlnet_multi])
            controlnet_tokenizer = CLIPTokenizer.from_pretrained(args.controlnet_clip_vision_encoder_path, use_safetensors=False)
            controlnet_text_encoder = CLIPTextModel.from_pretrained(args.controlnet_clip_vision_encoder_path, use_safetensors=False)
            controlnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(args.controlnet_clip_vision_encoder_path, use_safetensors=False)
            print('controlnet loaded.')

        global dino_encoder
        dino_encoder = FrozenDinoV2Encoder()
        dino_encoder.to(device, dtype)
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
            scheduler=scheduler
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


    data_dir = args.data_dir
    dataset = InferPairDataset(
        data_dir=data_dir,
        dataset_name=None,
        datalist=None,
        height=512,
        width=512,
        stride=4,
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False
    )
    
    with torch.no_grad():
        pipe_outs = pipe(
            dino_encoder=dino_encoder,
            dataloader=dataloader,
            denosing_steps=args.denoising_steps,
            processing_res=args.processing_res,
            match_input_res= not args.output_processing_res,
            batch_size=args.batch_size,
            show_progress_bar=True,
            guidance_scale=args.guidance_scale,
            setting_config=args.setting_config,
            edge2_src_mode=args.edge2_src_mode,
            preprocessor=preprocessor,
            generator=generator,
            weight_dtype=dtype,
        )
        for i, pipe_out in enumerate(pipe_outs):
            # save result
            to_save_dict = pipe_out.to_save_dict
            to_save_dict['pred2'] = pipe_out.img_pil
            
            images_tensor = []
            for image_name, image in to_save_dict.items():
                if image_name in ['ref1','raw2','edge2','pred2']:
                    image_save_path = os.path.join(data_dir[i]+f"/00", f'{image_name}.png')
                    if os.path.exists(image_save_path):
                        logging.warning(
                            f"Existing file: '{image_save_path}' will be overwritten"
                        )
                    print(f"img save in {image_save_path}")
                    os.makedirs(data_dir[i]+f"/00", exist_ok=True)
                    image.save(image_save_path)