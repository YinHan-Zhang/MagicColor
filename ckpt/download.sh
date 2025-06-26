export HF_ENDPOINT="https://hf-mirror.com"

huggingface-cli download --resume-download lllyasviel/Annotators --local-dir ./Annotators
echo "download Annotators success!"

wget https://hf-mirror.com/lllyasviel/control_v11p_sd15s2_lineart_anime/resolve/main/diffusion_pytorch_model.bin
echo "download controlnet success!"

huggingface-cli download --resume-download openai/clip-vit-large-patch14 --local-dir ./clip-vit-large-patch14
echo "download clip success!"

huggingface-cli download --resume-download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir ./stable-diffusion-v1-5
echo "download stable-diffusion-v1-5 success!"

huggingface-cli download --resume-download stabilityai/sd-vae-ft-mse --local-dir ./sd-vae-ft-mse
echo "download sd-vae-ft-mse!"

wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
echo "download Dinov2 success!"

huggingface-cli download --resume-download yinhan/MagicColor --local-dir ./MagicColor
echo "download MagicColor success!"
