export HF_ENDPOINT="https://hf-mirror.com"

huggingface-cli download --resume-download lllyasviel/Annotators --local-dir ./Annotators
echo "download Annotators success!"

wget https://hf-mirror.com/lllyasviel/control_v11p_sd15s2_lineart_anime/resolve/main/diffusion_pytorch_model.bin
echo "download clip success!"

huggingface-cli download --resume-download openai/clip-vit-large-patch14 --local-dir ./clip-vit-large-patch14
echo "download clip success!"