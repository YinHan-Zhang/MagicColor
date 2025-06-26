cd ..
cd inference

pretrained_model_name_or_path='../ckpt/stable-diffusion-v1-5'
refnet_clip_vision_encoder_path='../ckpt/clip-vit-large-patch14'
controlnet_clip_vision_encoder_path='../ckpt/clip-vit-large-patch14'
controlnet_model_name_or_path='../ckpt/controlnet_lineart'
annotator_ckpts_path='../ckpt/Annotators'
checkpoint_path="../ckpt/MagicColor"
vae_path="../ckpt/sd-vae-ft-mse"


#inference config
config='dafault'
input_data_dir="../data/test1" # inference input data dir

CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 0 --mixed_precision=no multi_ref_infer.py  \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --refnet_clip_vision_encoder_path $refnet_clip_vision_encoder_path \
                  --controlnet_clip_vision_encoder_path $controlnet_clip_vision_encoder_path \
                  --controlnet_model_name_or_path $controlnet_model_name_or_path \
                  --checkpoint_path $checkpoint_path \
                  --annotator_ckpts_path $annotator_ckpts_path \
                  --vae_path $vae_path \
                  --output_dir $output_dir \
                  --denoising_steps 20 \
                  --guidance_scale 9.0 \
                  --processing_res 512 \
                  --output_processing_res \
                  --setting_config $config \
                  --batch_size 1 \
                  --data_dir $input_data_dir
