import argparse
import os
import logging
import pickle
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

import sys
sys.path.append("/hpc2hdd/home/yzhang472/work/colorization")
sys.path.append("/hpc2hdd/home/yzhang472/work/colorization/training")
sys.path.append("/hpc2hdd/home/yzhang472/work/colorization/src")
sys.path.append("/hpc2hdd/home/yzhang472/work/colorization/utils")
from seed_all import seed_all, load_seed
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
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.refunet_2d_condition import RefUNet2DConditionModel

from setting_config import setting_configs

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
        "--validation_bg",
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
        help="setting name (default: v00-naive-unet-edit2-raw2-refnet-ref1)" # training.setting_config
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
    
    # if args.checkpoint_path:
    #     # 加载保存的随机状态文件
    #     with open(args.checkpoint_path+'/random_states_0.pkl', 'rb') as f:
    #         random_states = pickle.load(f)
    #         load_seed(random_states)
    #         print("load random seed ! ")
    # else:
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

    # setting_config = args.setting_config 
    # preprocessor = setting_configs[args.setting_config].get('preprocessor', None)
    # in_channels_reference_unet = setting_configs[args.setting_config].get('in_channels_reference_unet', None)
    # in_channels_denoising_unet = setting_configs[args.setting_config].get('in_channels_denoising_unet', None)
    # in_channels_controlnet = setting_configs[args.setting_config].get('in_channels_controlnet', None)
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
        from src.annotator.lineart_anime import LineartAnimeDetector
        preprocessor = LineartAnimeDetector(args.annotator_ckpts_path)
        preprocessor.to(device, dtype)
        print("lineart_anime preprocessor is loaded.")     
    else:
        preprocessor = None
        print("preprocessor not load !!!")
    
    # declare a pipeline
    if not use_seperate:
        raise NotImplementedError
        # pipe = ImagePairEditPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=dtype, use_safetensors=True)
        # print("Using Completed")
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae",use_safetensors=False)
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
        controlnet_path = os.path.join(args.checkpoint_path, 'controlnet.pth')
        
        # torch.set_rng_state(.state_dict['random_state'])

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
        if os.path.exists(controlnet_path):
            controlnet = ControlNetModel.from_pretrained(
                args.controlnet_model_name_or_path,
                in_channels=in_channels_controlnet,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
                use_safetensors=False
            )
            controlnet_tokenizer = CLIPTokenizer.from_pretrained(args.controlnet_clip_vision_encoder_path, use_safetensors=False)
            controlnet_text_encoder = CLIPTextModel.from_pretrained(args.controlnet_clip_vision_encoder_path, use_safetensors=False)
            controlnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(args.controlnet_clip_vision_encoder_path, use_safetensors=False)
            controlnet.load_state_dict(torch.load(controlnet_path, map_location="cpu"), strict=False)
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
        
        if args.validation_ref1:
            # load the example image.
            ref1 = Image.open(args.validation_ref1)
            raw2 = Image.open(args.validation_raw2)
            edit2 = Image.open(args.validation_edit2)
            # bg = Image.open(args.validation_bg)
            pipe_out = pipe(
                    ref1,
                    raw2,
                    edit2,
                    # bg,
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

            # save result
            to_save_dict = pipe_out.to_save_dict
            to_save_dict['pred2'] = pipe_out.img_pil

            import cv2
            images_tensor = []
            for image_name, image in to_save_dict.items():
                # save by single img
                if image_name in ['ref1','edge2','pred2']:
                    image_save_path = os.path.join(args.output_dir, f'{image_name}.png')
                    # if os.path.exists(image_save_path):
                    #     logging.warning(
                    #         f"Existing file: '{image_save_path}' will be overwritten"
                    #     )
                    image.save(image_save_path)

            # import cv2
            # images_tensor = []
            # for image_name, image in to_save_dict.items():
            #     ## save by grid
            #     if image_name in ['ref1','raw2','edge2','pred2']:
            #         image_np = np.array(image)
            #         image_tensor = torch.tensor(image_np, dtype=torch.uint8)
            #         image = torch.cat([image_tensor], 1)
            #         images_tensor.append(image)
                
            # gen_img = torch.cat(images_tensor, 1)
            # gen_img_np = cv2.cvtColor(gen_img.cpu().numpy(), cv2.COLOR_RGB2BGR)
            # # 保存拼接后的图像
            # image_save_path = os.path.join(args.output_dir, f'output.png')
            # cv2.imwrite(
            #     image_save_path, gen_img_np
            # )
        else:
            # val_dir = "/hpc2hdd/home/yzhang472/work/colorization/video_dataset/clip_image_pair/val_data"
            # save_dir = "/hpc2hdd/home/yzhang472/work/colorization/video_dataset/clip_image_pair/ablation_res"
            val_dir = '/hpc2hdd/home/yzhang472/work/colorization/video_dataset/comparsion'
            save_dir = '/hpc2hdd/home/yzhang472/work/colorization/video_dataset/comparsion_val'
            os.makedirs(save_dir, exist_ok=True)
            if True:
                for i in tqdm(range(1,6)):
                    ref1_path = os.path.join(val_dir+f"/{i}", f'frame1.jpg')
                    raw2_path = os.path.join(val_dir+f"/{i}", f'frame3.jpg')
                    edit2_path = os.path.join(val_dir+f"/{i}", f'frame3.jpg')
                    ref1 = Image.open(ref1_path).convert("RGB")
                    raw2 = Image.open(raw2_path).convert("RGB")
                    edit2 = Image.open(edit2_path).convert("RGB")
                
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

                    # save result
                    to_save_dict = pipe_out.to_save_dict
                    to_save_dict['pred2'] = pipe_out.img_pil
                    
                    import cv2
                    images_tensor = []
                    for image_name, image in to_save_dict.items():
                        # save by single img
                        if image_name in ['ref1','raw2','edge2','pred2']:
                            image_save_path = os.path.join(save_dir+f"/{i:02d}", f'{image_name}.png')
                            # if os.path.exists(image_save_path):
                            #     logging.warning(
                            #         f"Existing file: '{image_save_path}' will be overwritten"
                            #     )
                            os.makedirs(save_dir+f"/{i:02d}", exist_ok=True)
                            image.save(image_save_path)

                        # ## save by grid
                        # if image_name in ['ref1','raw2','edge2','pred2']:
                        #     image_np = np.array(image)
                        #     image_tensor = torch.tensor(image_np, dtype=torch.uint8)
                        #     image = torch.cat([image_tensor], 1)
                        #     images_tensor.append(image)
                        
                    # gen_img = torch.cat(images_tensor, 1)
                    # gen_img_np = cv2.cvtColor(gen_img.cpu().numpy(), cv2.COLOR_RGB2BGR)
                    # # 保存拼接后的图像
                    # image_save_path = os.path.join(args.output_dir, f'output_{i}.png')
                    # cv2.imwrite(
                    #     image_save_path, gen_img_np
                    # )

            else:
                infer_path = "/hpc2hdd/home/yzhang472/work/colorization/validation/teaser_new/animal1"
                frame_folder = [infer_path]
                for ins_dir in frame_folder:
                    ref1_path = os.path.join(ins_dir, f'img1.png')
                    raw2_path = os.path.join(ins_dir, f'img2.png')
                    edit2_path = os.path.join(ins_dir, f'img2.png')
                    ref1 = Image.open(ref1_path).convert("RGB")
                    raw2 = Image.open(raw2_path).convert("RGB")
                    edit2 = Image.open(edit2_path).convert("RGB")
                
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

                    # save result
                    to_save_dict = pipe_out.to_save_dict
                    to_save_dict['pred2'] = pipe_out.img_pil
                    
                    import cv2
                    images_tensor = []
                    for image_name, image in to_save_dict.items():
                        # save by single img
                        if image_name in ['ref1','pred2']:
                            image_save_path = os.path.join(infer_path, f'{image_name}_noinstance.png')
                            if os.path.exists(image_save_path):
                                logging.warning(
                                    f"Existing file: '{image_save_path}' will be overwritten"
                                )
                            image.save(image_save_path)

                    #     ## save by grid
                    #     if image_name in ['ref1','raw2','edge2','pred2']:
                    #         image_np = np.array(image)
                    #         image_tensor = torch.tensor(image_np, dtype=torch.uint8)
                    #         image = torch.cat([image_tensor], 1)
                    #         images_tensor.append(image)
                        
                    # gen_img = torch.cat(images_tensor, 1)
                    # gen_img_np = cv2.cvtColor(gen_img.cpu().numpy(), cv2.COLOR_RGB2BGR)
                    # # 保存拼接后的图像
                    # image_save_path = os.path.join(args.output_dir, f'output_{i}.png')
                    # cv2.imwrite(
                    #     image_save_path, gen_img_np
                    # )
