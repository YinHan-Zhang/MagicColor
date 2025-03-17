import gradio as gr
import torch
import torchvision
from PIL import Image
import numpy as np
import os
import logging
from typing import List, Dict, Any
import argparse
import cv2
from tqdm import tqdm
import shutil
import sys
sys.path.append("../Grounded-Segment-Anything")

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import build_sam, build_sam_hq, SamPredictor

# Recognize Anything Model & Tag2Text
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS
from multi_ref_infer import parse_args, InferPairDataset, DataLoader, ImagePairEditPipeline, FrozenDinoV2Encoder, AutoencoderKL, DDIMScheduler, UNet2DConditionModel, RefUNet2DConditionModel, ControlNetModel, MultiControlNetModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection,LineartAnimeDetector

# -------------------- SAM, RAM, Grounding DINO --------------------
def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
   
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
   
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())
    return boxes_filt, torch.Tensor(scores), pred_phrases

def extract_mask_with_sam(image_path, index, model, ram_model, predictor, device):
   
    image_pil, image = load_image(image_path)

    # RAM 
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])
    raw_image = image_pil.resize((384, 384))
    raw_image = transform(raw_image).unsqueeze(0).to(device)
    res = inference_ram(raw_image, ram_model)
    tags = res[0].replace(' |', ',')

    # Grounding DINO
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model, image, "person", box_threshold=0.05, text_threshold=0.05, device=device
    )

    # SAM 
    image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    predictor.set_image(image_cv2)
    H, W = image_pil.size[1], image_pil.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    boxes_filt = boxes_filt.cpu()
    nms_idx = torchvision.ops.nms(boxes_filt, scores, 0.05).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv2.shape[:2]).to(device)
    try:
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    except:
        masks = torch.ones((1,1,image.shape[0],image.shape[1]), device=device)
    # save mask
    masks = masks[0]
    os.makedirs("./user_input/masks", exist_ok=True)
    mask_img = torch.zeros(masks.shape[-2:])
    mask_img[masks.cpu().numpy()[0] == True] = 1
    mask_img_uint8 = (mask_img.numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join("./user_input/masks", f"mask_{index}.png"), mask_img_uint8)


def initialize_pipeline(args):

    preprocessor = LineartAnimeDetector(args.annotator_ckpts_path)
    preprocessor.to(device, dtype)

    vae = AutoencoderKL.from_pretrained(
        "../ckpt/sd-vae-ft-mse",
        use_safetensors=False
    )
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler', use_safetensors=False)
    denoising_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        in_channels=4,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype, 
        use_safetensors=False
    )
    reference_unet = RefUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        in_channels=4,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype, 
        use_safetensors=False
    )
    denoising_unet_path = os.path.join(args.checkpoint_path, 'denoising_unet.pth')
    reference_unet_path = os.path.join(args.checkpoint_path, 'reference_unet.pth')
    denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False)
    reference_unet.load_state_dict(torch.load(reference_unet_path, map_location="cpu"), strict=False)

    controlnet_multi_path = os.path.join(args.checkpoint_path, 'controlnet_multi')
    controlnet_multi = ControlNetModel.from_pretrained(
        controlnet_multi_path,
        in_channels=4,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    )
    controlnet_sketch_path = os.path.join(args.checkpoint_path, 'controlnet_sketch')
    controlnet_sketch = ControlNetModel.from_pretrained(
        controlnet_sketch_path,
        in_channels=4,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    )
    controlnet = MultiControlNetModel(controlnets=[controlnet_sketch, controlnet_multi])
    
    refnet_tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer', use_safetensors=False)
    refnet_text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder', use_safetensors=False)
    refnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(args.refnet_clip_vision_encoder_path, use_safetensors=False)
    controlnet_tokenizer = CLIPTokenizer.from_pretrained(args.controlnet_clip_vision_encoder_path, use_safetensors=False)
    controlnet_text_encoder = CLIPTextModel.from_pretrained(args.controlnet_clip_vision_encoder_path, use_safetensors=False)
    controlnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(args.controlnet_clip_vision_encoder_path, use_safetensors=False)
    
    dino_encoder = FrozenDinoV2Encoder().to(device, dtype)
    
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
    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    print("model load success!")
    return pipe, dino_encoder, preprocessor

# -------------------- Gradio --------------------
def run_inference(sketch_image: Image.Image, ref_images: List[Image.Image], args):
    
    dataset = InferPairDataset(
        data_dir=["./user_input"], 
        dataset_name="",
        datalist=None,  
        height=512,
        width=512,
        stride=4,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        pipe_outs = pipe(
            dino_encoder=dino_encoder,
            dataloader=dataloader,
            denosing_steps=args.denoising_steps,
            processing_res=args.processing_res,
            match_input_res=not args.output_processing_res,
            batch_size=args.batch_size,
            show_progress_bar=True,
            guidance_scale=args.guidance_scale,
            setting_config=args.setting_config,
            edge2_src_mode=args.edge2_src_mode,
            preprocessor=preprocessor,
            generator=torch.manual_seed(args.seed),
            weight_dtype=dtype,
        )
    
    result_image = pipe_outs[0].img_pil

    to_save_dict = pipe_outs[0].to_save_dict
    os.makedirs("./user_input/infer_res", exist_ok=True)
    for image_name, image in to_save_dict.items():
        if image_name in ['ref1','raw2','edge2','pred2']:
            image_save_path = os.path.join(f"./user_input/infer_res", f'{image_name}.jpg')
            image.save(image_save_path)
    return result_image


def run_case_inference(case_num):
    case_folder = os.path.join("./case", f"case{case_num}")
    sketch_image = Image.open(os.path.join(case_folder, "sketch.jpg"))

    ref_images = []
    idx = 1
    while True:
        ref_image_path = os.path.join(case_folder, f"case{case_num}_{idx}.jpg")
        if os.path.exists(ref_image_path):
            ref_image = Image.open(ref_image_path)
            ref_images.append(ref_image)
            idx += 1
        else:
            break

    dataset = InferPairDataset(
        data_dir=[case_folder], 
        dataset_name="",
        datalist=None,  
        height=512,
        width=512,
        stride=4,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        pipe_outs = pipe(
            dino_encoder=dino_encoder,
            dataloader=dataloader,
            denosing_steps=args.denoising_steps,
            processing_res=args.processing_res,
            match_input_res=not args.output_processing_res,
            batch_size=args.batch_size,
            show_progress_bar=True,
            guidance_scale=args.guidance_scale,
            setting_config=args.setting_config,
            edge2_src_mode=args.edge2_src_mode,
            preprocessor=preprocessor,
            generator=torch.manual_seed(args.seed),
            weight_dtype=dtype,
        )
    
    result_image = pipe_outs[0].img_pil 
    ref_images.append(result_image)
    
    images_to_show = [sketch_image] + ref_images
    return result_image, images_to_show

def gradio_interface(sketch_image: Image.Image, *ref_images):
    
    shutil.rmtree("./user_input")

    os.makedirs("./user_input", exist_ok=True)
    sketch_image.save("./user_input/user_input.jpg")

    new_ref_images = []

    for idx, image in enumerate(ref_images):
        if image is not None:  
            image_path = os.path.join("./user_input", f"user_input_{idx + 1}.jpg")
            image.save(image_path)
            new_ref_images.append(image)

    for idx in range(len(new_ref_images)):
        image_path = os.path.join("./user_input", f"user_input_{idx + 1}.jpg")
        extract_mask_with_sam(image_path, idx + 1, gounding_dino_model, ram_model, sam_model, device)


    result_image = run_inference(sketch_image, new_ref_images, args)
    return result_image


def add_reference_image(ref_count):
   
    ref_count += 1
    updates = [gr.update(visible=True if i < ref_count else False) for i in range(10)]  
    updates.append(ref_count)  
    return updates


def clear_images():
    return [None] * 11  


if __name__ == "__main__":

    # initial SAM、RAM and Grounding DINO 
    config = "../Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    ram_checkpoint = "../Grounded-Segment-Anything/ckpt/ram_swin_large_14m.pth" 
    grounded_checkpoint = "../Grounded-Segment-Anything/ckpt/groundingdino_swint_ogc.pth"
    sam_checkpoint = "../Grounded-Segment-Anything/ckpt/sam_vit_h_4b8939.pth"
    global gounding_dino_model
    global ram_model
    global sam_model
    global device
    global dtype
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gounding_dino_model = load_model(config, grounded_checkpoint, device)
    ram_model = ram(pretrained=ram_checkpoint, image_size=384, vit='swin_l').to(device)
    ram_model.eval()
    sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    global pipe
    global dino_encoder
    global preprocessor
    global args
    
    args = parse_args()
    args.pretrained_model_name_or_path = '../ckpt/stable-diffusion-v1-5'
    args.refnet_clip_vision_encoder_path = '../ckpt/clip-vit-large-patch14'
    args.controlnet_clip_vision_encoder_path = '../ckpt/clip-vit-large-patch14'
    args.controlnet_model_name_or_path = '../ckpt/controlnet_lineart'
    args.annotator_ckpts_path = '../ckpt/Annotators'
    args.checkpoint_path = "../ckpt/checkpoint-090000" 
    args.output_dir = "./user_input/output"
    args.denoising_steps = 20
    args.processing_res = 512
    args.guidance_scale = 9.0
    args.seed = 123
    args.batch_size = 1
    args.setting_config = "v00-naive-unet-edit2-raw2-refnet-ref1"
    args.edge2_src_mode = "raw"
    args.output_processing_res = True

    pipe, dino_encoder, preprocessor = initialize_pipeline(args)
    
    with gr.Blocks(css=".gradio-container {max-width: 1200px; margin: 0 auto;}") as demo:
       
        gr.Markdown("# MagicColor: Multi-instance Sketch Colorization")
        gr.Markdown("## User Guide: \n 1. Upload a sketch image \n 2. Click add reference (one or more images/instances) \n 3. Click generate result!") #(if you don't have, you can upload a random image and the model automatically extracts the line art.)
        gr.Markdown("We random select a few example and you can click 【select case】 to experience. Case images are thumbnails(缩略图), click on one of them to see them all (sketch, references, result).")
       
        ref_count = gr.State(0)

        with gr.Row():
            # left：upload ref img
            with gr.Column(scale=1, elem_classes="card"):
               
                with gr.Row():
                    sketch_input = gr.Image(label="Sketch Image", type="pil", height=256, width=256)
               
                ref_inputs = []
                for i in range(5):  
                    with gr.Row():
                        for j in range(2):
                            index = i * 2 + j
                            input_img = gr.Image(label=f"Reference Image {index + 1}", type="pil", height=256, width=256, visible=False)
                            ref_inputs.append(input_img)
                gr.Markdown("")  
                add_ref_button = gr.Button("Add Reference Image", variant="secondary", scale=1)
            # right: generate result
            with gr.Column(scale=1, elem_classes="card"):
               
                gr.Markdown("### Generated Result")
                output_image = gr.Image(label="Result Image", type="pil", height=512, width=512)
                
                
                ref_count_display = gr.Number(value=0, label="Number of Reference Instances", interactive=False, scale=1)
                gr.Markdown("")  
            
                submit_button = gr.Button("Generate Result", variant="primary", scale=1)
                gr.Markdown("")  
   
                clear_button = gr.Button("Clear All Instances", variant="secondary", scale=1)

                case_selector = gr.Dropdown(choices=[str(i + 1) for i in range(10)], value="1", label="Select Case")  # 假设最多 3 个案例
                case_button = gr.Button("Show Case", variant="secondary", scale=1)
        

        case_images_display = gr.Gallery(label="Case Sketch and Reference Instances", columns=4, height=256)

        add_ref_button.click(
            fn=add_reference_image,
            inputs=ref_count,
            outputs=ref_inputs + [ref_count_display],
        )
        def update_ref_count(ref_count):
            return ref_count + 1

        add_ref_button.click(
            fn=update_ref_count,
            inputs=ref_count,
            outputs=ref_count,
        )

        clear_button.click(
            fn=clear_images,
            outputs=[sketch_input] + ref_inputs,
        )

        submit_button.click(
            fn=gradio_interface,
            inputs=[sketch_input] + ref_inputs,
            outputs=output_image,
            show_progress=True
        )
 
        case_button.click(
            fn=run_case_inference,
            inputs=case_selector,
            outputs=[output_image, case_images_display],
            show_progress=True
        )
       
    demo.launch(server_name="0.0.0.0" ,server_port=9999, share=True, inbrowser=True)