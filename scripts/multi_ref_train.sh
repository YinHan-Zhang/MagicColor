cd ..
cd training

# accelerate config default
pretrained_model_name_or_path='../ckptstable-diffusion-v1-5'
refnet_clip_vision_encoder_path='../ckpt/clip-vit-large-patch14'
controlnet_clip_vision_encoder_path='../ckpt/clip-vit-large-patch14'
controlnet_model_name_or_path='../ckpt/controlnet_lineart'
annotator_ckpts_path='../ckpt/Annotators'
vae_path="../ckpt/sd-vae-ft-mse"
checkpoint_path=""

# dataset config
root_path=''
dataset_name='multi'
trainlist=''
height=512 
width=512 
stride=24 # clip length

# training config
output_root='../outputs'
tracker_project_name='train'
train_batch_size=1
num_train_epochs=10
gradient_accumulation_steps=1
learning_rate=1e-5
lr_warmup_steps=10
dataloader_num_workers=8
resume_from_checkpoint='latest'

vallist=""
validation_steps=2000
checkpointing_steps=2000
checkpoints_total_limit=20
config="default"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --mixed_precision=no --main_process_port 29400 multi_ref_train.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --refnet_clip_vision_encoder_path $refnet_clip_vision_encoder_path \
                  --controlnet_clip_vision_encoder_path $controlnet_clip_vision_encoder_path \
                  --controlnet_model_name_or_path $controlnet_model_name_or_path \
                  --annotator_ckpts_path $annotator_ckpts_path \
                  --checkpoint_path $checkpoint_path \
                  --controlnet_ckpt_path $checkpoint_path \
                  --vae_path $vae_path \
                  --dataset_name $dataset_name \
                  --setting_config $config \
                  --trainlist $trainlist \
                  --vallist $vallist \
                  --dataset_path $root_path \
                  --height $height --width $width --stride $stride \
                  --output_root $output_root \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --gradient_checkpointing \
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --gradient_checkpointing \
                  --no_eval_before_train \
                  --validation_steps $validation_steps \
                  --checkpointing_steps $checkpointing_steps \
                  --checkpoints_total_limit $checkpoints_total_limit \
                  --resume_from_checkpoint $resume_from_checkpoint


