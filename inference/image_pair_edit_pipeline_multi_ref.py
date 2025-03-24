
from typing import Any, Dict, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import masks_to_boxes
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    ControlNetModel,
    MultiControlNetModel,
    DDIMScheduler,
    AutoencoderKL,
)
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor
from transformers import CLIPVisionModelWithProjection
import sys
sys.path.append("../utils")
from image_util import resize_max_res,chw2hwc
import albumentations as A
from torchvision import transforms as T
from albumentations.pytorch import ToTensorV2
# from utils.colormap import kitti_colormap
# from utils.depth_ensemble import ensemble_depths

from models.mutual_self_attention import ReferenceAttentionControl
from models.unet_2d_condition import UNet2DConditionModel
from models.refunet_2d_condition import RefUNet2DConditionModel
from models.projection import Projection
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from PIL import Image

class ImagePairEditPipelineOutput(BaseOutput):
    img_np: np.ndarray
    img_pil: Image.Image
    to_save_dict: dict



class ImagePairEditPipeline(DiffusionPipeline):
    # two hyper-parameters
    rgb_latent_scale_factor = 0.18215
    
    def __init__(self,
        reference_unet: RefUNet2DConditionModel,
        controlnet: MultiControlNetModel,
        denoising_unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        refnet_tokenizer: CLIPTokenizer,
        refnet_text_encoder: CLIPTextModel,
        refnet_image_enc: CLIPVisionModelWithProjection,
        controlnet_tokenizer: CLIPTokenizer,
        controlnet_text_encoder: CLIPTextModel,
        controlnet_image_enc: CLIPVisionModelWithProjection,
        scheduler: DDIMScheduler,
    ):
        super().__init__()
            
        self.register_modules(
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
        self.empty_text_embed = None
        self.clip_image_processor = CLIPImageProcessor()
        self.patch_size = 504
        
    @torch.no_grad()
    def __call__(
        self,
        dataloader,
        dino_encoder,
        denosing_steps: int = 20,
        processing_res: int = 512,
        match_input_res: bool = False,
        batch_size: int = 1,
        show_progress_bar: bool = True,
        guidance_scale: float = 3.5,
        setting_config: str = 'default',
        edge2_src_mode: str = 'raw',
        preprocessor=None,
        generator=None,
        weight_dtype=float,
    ) -> ImagePairEditPipelineOutput:
        
        # inherit from thea Diffusion Pipeline
        device = self.device
        output_list = []
        if show_progress_bar:
            iterable_bar = tqdm(
                dataloader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable_bar = dataloader
        
        for batch in iterable_bar:

            mtv_condition = batch["ctrl_img"]
            curr_h, curr_w = batch["img1"].shape[-2:]
            ctrl_channel = 1024
            prompt_fea = torch.zeros((*batch["ctrl_img"][0].shape, ctrl_channel)).to(
                device, dtype=weight_dtype
            )
            for curr_b, curr_ins_img in enumerate(batch["patches"]):
                    curr_ins_id, curr_ins_patch = curr_ins_img[0], curr_ins_img[1].to(
                        prompt_fea
                    )
                    curr_ins_patch = curr_ins_patch.squeeze(0)

                    if curr_ins_id.shape[0] > 0:
                        with torch.cuda.amp.autocast():
                            image_features = dino_encoder(curr_ins_patch)
                            image_features = self.controlnet.nets[1].dino_adapter(image_features).to(
                                prompt_fea
                            )
                        
                        for id_ins, curr_ins in enumerate(curr_ins_id.tolist()[0]):
                            all_vector = image_features[id_ins] 
                            global_vector = all_vector[0:1] 

                            down_s = self.patch_size // 14

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
                                all_ins,
                                size=(curr_pix_num // 10,),
                                replace=True,
                            )
                            warp_fea[small_mask][sel_ins] = global_vector

                            prompt_fea[curr_b][
                                curr_box[1] : curr_box[3], curr_box[0] : curr_box[2]
                            ][small_mask] = warp_fea[small_mask]
            
            batch["conditioning_pixel_values"] = prompt_fea.permute(0, 3, 1, 2) # mtv_condition 

            controlnet_image = batch["conditioning_pixel_values"].to(
                device,dtype=weight_dtype
            )

            clip_image = batch["clip_images"]

            def img2embeds(clip_image, image_enc):
                clip_image_embeds = image_enc(
                    clip_image.to(device, dtype=image_enc.dtype)
                ).image_embeds
                encoder_hidden_states = clip_image_embeds.unsqueeze(1)
                return encoder_hidden_states
            
            if self.reference_unet:
                refnet_encoder_hidden_states = img2embeds(clip_image, self.refnet_image_enc)
            else:
                refnet_encoder_hidden_states = None
            if self.controlnet:
                controlnet_encoder_hidden_states = img2embeds(clip_image, self.refnet_image_enc)
            else:
                controlnet_encoder_hidden_states = None

            prompt = ""
            def prompt2embeds(prompt, tokenizer, text_encoder):
                text_inputs = tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                empty_text_embed = text_encoder(text_input_ids)[0].to(self.dtype)
                uncond_encoder_hidden_states = empty_text_embed.repeat((1, 1, 1))[:,0,:].unsqueeze(0)
                return uncond_encoder_hidden_states
            
            if self.reference_unet:
                refnet_uncond_encoder_hidden_states = prompt2embeds(prompt, self.refnet_tokenizer, self.refnet_text_encoder) #.unsqueeze(0)
            else:
                refnet_uncond_encoder_hidden_states = None
            if self.controlnet:
                controlnet_uncond_encoder_hidden_states = prompt2embeds(prompt, self.controlnet_tokenizer, self.controlnet_text_encoder) #.unsqueeze(0)
            else:
                controlnet_uncond_encoder_hidden_states = None
        
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                if self.reference_unet:
                    refnet_encoder_hidden_states = torch.cat(
                        [refnet_uncond_encoder_hidden_states, refnet_encoder_hidden_states], dim=0
                    )
                else:
                    refnet_encoder_hidden_states = None

                if self.controlnet:
                    controlnet_encoder_hidden_states = torch.cat(
                        [controlnet_uncond_encoder_hidden_states, controlnet_encoder_hidden_states], dim=0
                    )
                else:
                    controlnet_encoder_hidden_states = None

            # Ref attn
            if self.reference_unet:
                reference_control_writer = ReferenceAttentionControl(
                    self.reference_unet,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    mode="write",
                    batch_size=batch_size,
                    fusion_blocks="full",
                )
                reference_control_reader = ReferenceAttentionControl(
                    self.denoising_unet,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    mode="read",
                    batch_size=batch_size,
                    fusion_blocks="full",
                )
            else:
                reference_control_writer = None
                reference_control_reader = None
            
            img_pred, to_save_dict = self.single_infer(
                batch,
                controlnet_image,
                dino_encoder,
                num_inference_steps=denosing_steps,
                show_pbar=show_progress_bar,
                guidance_scale=guidance_scale,
                refnet_encoder_hidden_states=refnet_encoder_hidden_states,
                controlnet_encoder_hidden_states=controlnet_encoder_hidden_states,
                reference_control_writer=reference_control_writer,
                reference_control_reader=reference_control_reader,
                setting_config=setting_config,
                edge2_src_mode=edge2_src_mode,
                preprocessor=preprocessor,
                generator=generator,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=device
            )
            
            for k, v in to_save_dict.items():
                to_save_dict[k] = Image.fromarray(
                    chw2hwc(((v.squeeze().detach().cpu().numpy() + 1.) / 2 * 255).astype(np.uint8))
                )
        
            # torch.cuda.empty_cache()  # clear vram cache for ensembling
        
            # ----------------- Post processing -----------------        
            # Convert to numpy
            img_pred = img_pred.squeeze().cpu().numpy().astype(np.float32)
            img_pred_np = (((img_pred + 1.) / 2.) * 255).astype(np.uint8)
            img_pred_np = chw2hwc(img_pred_np)
            img_pred_pil = Image.fromarray(img_pred_np)

            # # Resize back to original resolution
            # if match_input_res:
            #     img_pred_pil = img_pred_pil.resize(input_size)
            #     img_pred_np = np.asarray(img_pred_pil)
                
            output_list.append(
                ImagePairEditPipelineOutput(
                    img_np=img_pred_np,
                    img_pil=img_pred_pil,
                    to_save_dict=to_save_dict
                )
            )        
        return output_list
    
    def __encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device) #[1,2]
    
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype) #[1,2,1024]

    def feature_transfer(self, src_ft, trg_ft):
        """
        func desc: transfer src_ft to trg_ft
        args:
            src_ft: source feature,size = [1,4,64,64]
            trg_ft: target feature,size = [1,4,64,64]

        return: trans_ft
        """
        import torch.nn.functional as F
        def gen_grid(h, w, device, normalize=False, homogeneous=False):
            if normalize:
                lin_y = torch.linspace(-1., 1., steps=h, device=device)
                lin_x = torch.linspace(-1., 1., steps=w, device=device)
            else:
                lin_y = torch.arange(0, h, device=device)
                lin_x = torch.arange(0, w, device=device)
            grid_y, grid_x = torch.meshgrid((lin_y, lin_x))
            grid = torch.stack((grid_x, grid_y), -1)
            if homogeneous:
                grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
            return grid  # [h, w, 2 or 3]

        def normalize_coords(coords, h, w, no_shift=False):
            assert coords.shape[-1] == 2
            if no_shift:
                return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2
            else:
                return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2 - 1.
         
        sd_feat_src = F.normalize(src_ft.squeeze(), p=2, dim=0)
        sd_feat_trg = F.normalize(trg_ft.squeeze(), p=2, dim=0)
        feat_dim = sd_feat_src.shape[0]

        h_src, w_src = src_ft.shape[2], src_ft.shape[3]
        h_trg, w_trg = trg_ft.shape[2], trg_ft.shape[3]

        grid_src = gen_grid(h_src, w_src, device='cuda')
        grid_trg = gen_grid(h_trg, w_trg, device='cuda')
       
        sticker_mask = torch.ones_like(src_ft[0, 0])

        coord_src = grid_src[sticker_mask > 0]
        coord_src = coord_src[torch.randperm(len(coord_src))][:1000]
        coord_src_normed = normalize_coords(coord_src, h_src, w_src)
        grid_trg_normed = normalize_coords(grid_trg, h_trg, w_trg)

        feat_src = F.grid_sample(sd_feat_src[None], coord_src_normed[None, None], align_corners=True).squeeze().T
        feat_trg = F.grid_sample(sd_feat_trg[None], grid_trg_normed[None], align_corners=True).squeeze()
        feat_trg_flattened = feat_trg.permute(1, 2, 0).reshape(-1, feat_dim)

        distances = torch.cdist(feat_src, feat_trg_flattened)
        _, indices = torch.min(distances, dim=1)

        src_pts = coord_src.reshape(-1, 2).cpu().numpy()
        trg_pts = grid_trg.reshape(-1, 2)[indices].cpu().numpy() 

        trans_ft = trg_ft.clone()
        for i, (src_pt, trg_pt) in enumerate(zip(src_pts, trg_pts)):
            src_y, src_x = int(src_pt[0]), int(src_pt[1])
            trg_y, trg_x = int(trg_pt[0]), int(trg_pt[1])

            trans_ft[:, :, trg_y, trg_x] = src_ft[:, :, src_y, src_x] 

        return trans_ft, trg_ft

    @torch.no_grad()
    def single_infer(
        self,
        batch,
        controlnet_image,
        dino_encoder,
        num_inference_steps: int,
        show_pbar: bool,
        guidance_scale: float,
        refnet_encoder_hidden_states: torch.Tensor,
        controlnet_encoder_hidden_states: torch.Tensor,
        reference_control_writer: ReferenceAttentionControl,
        reference_control_reader: ReferenceAttentionControl,
        setting_config: str,
        edge2_src_mode: str,
        preprocessor,
        generator,
        do_classifier_free_guidance,
        device
    ):
        
        ref1 = batch['img1'].to(device)
        raw2 = batch['img2'].to(device)

        to_save_dict = {
            'ref1': ref1,
            'raw2': raw2,
            'gt2': raw2,
        }
        
        # Set timesteps: inherit from the diffusion pipeline
        self.scheduler.set_timesteps(num_inference_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]
        
        # encode image
        ref1_latents = self.encode_RGB(ref1, generator=generator) # 1/8 Resolution with a channel nums of 4. 
        raw2_latents = self.encode_RGB(raw2, generator=generator) # 1/8 Resolution with a channel nums of 4. 
        edge2 = raw2
        
        # # raw img extract sketch
        # edge2 = preprocessor(edge2)
        # edge2[edge2 <= 0.25] = 0
        # edge2 = edge2.repeat(1, 3, 1, 1) * 2 - 1
        # print(f"edge2: {edge2},edge2 out shape: {edge2.shape}")
        edge2_latents = self.encode_RGB(edge2, generator=generator)
        to_save_dict['edge2'] = edge2
        
        # Initial depth map (Guassian noise)
        noisy_edit2_latents = torch.randn(
            raw2_latents.shape, device=device, dtype=self.dtype
        )  # [B, 4, H/8, W/8]
   
        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
        # start denosing steps
        for i, t in iterable:
            
            refnet_input = ref1_latents
            controlnet_inputs = (noisy_edit2_latents, edge2, controlnet_image) # add sketch and controlnet_image to controlnet
            unet_input = torch.cat([noisy_edit2_latents], dim=1)
        
            # 1. add ref to ref unet
            if self.reference_unet: 
                    ref_ft_all = self.reference_unet(
                        refnet_input.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1),
                        torch.zeros_like(t),
                        encoder_hidden_states=refnet_encoder_hidden_states,
                        return_dict=True,
                    ).output
                    
                    up_ft_index = 1
                    ref_ft = ref_ft_all['up_ft'][up_ft_index] 
                    ref_ft = ref_ft.mean(0, keepdim=True) 
            
            if i == 0: # update reference attn at first step
                reference_control_reader.update(reference_control_writer)

            # 2. controlnet
            if self.controlnet:
                noisy_latents, sketch_cond , controlnet_image = controlnet_inputs 
                sketch_cond = sketch_cond.repeat((2 if do_classifier_free_guidance else 1), 1, 1, 1).to(device)
                controlnet_image = controlnet_image.to(device)
                
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_latents.repeat((2 if do_classifier_free_guidance else 1), 1, 1, 1),
                    t,
                    encoder_hidden_states=controlnet_encoder_hidden_states,
                    controlnet_cond=[sketch_cond, controlnet_image], 
                    return_dict=False,
                )
            else:
                down_block_res_samples, mid_block_res_sample = None, None

            # 3. predict the noise residual
            noise_pred = self.denoising_unet(
                unet_input.repeat(
                    (2 if do_classifier_free_guidance else 1), 1, 1, 1).to(dtype=self.denoising_unet.dtype), 
                t, 
                encoder_hidden_states=controlnet_encoder_hidden_states,#refnet_encoder_hidden_states,# 768
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                ref_ft = ref_ft
            ).sample 

            # 4. denoising forward
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # 5. compute the previous noisy sample x_t -> x_t-1
            noisy_edit2_latents = self.scheduler.step(noise_pred, t, noisy_edit2_latents).prev_sample
        
        # clear ref attn
        reference_control_reader.clear()
        reference_control_writer.clear()
        torch.cuda.empty_cache()

        # clip prediction
        edit2 = self.decode_RGB(noisy_edit2_latents)
        edit2 = torch.clip(edit2, -1.0, 1.0)

        return edit2, to_save_dict
        
    
    def encode_RGB(self, rgb_in: torch.Tensor, generator) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        
        # generator = None
        rgb_latent = self.vae.encode(rgb_in).latent_dist.sample(generator)
        rgb_latent = rgb_latent * self.rgb_latent_scale_factor
        return rgb_latent
    
    def decode_RGB(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            rgb_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """

        rgb_latent = rgb_latent / self.rgb_latent_scale_factor
        rgb_out = self.vae.decode(rgb_latent, return_dict=False)[0]
        return rgb_out


