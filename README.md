# MagicColor: Multi-Instance Sketch Colorization
![](./asset/logo.png)


## Overview
We present MagicColor, a diffusion-based framework for multi-instance sketch colorization. Previous methods can only achieve multi-instance sketch colorization step by step, which is time-consuming and inaccurate. In contrast, our method has the capability of coloring sketches while maintaining consistency, making multi-instance sketch colorization easier.

![](./asset/intro.jpg)

### Gallery
Given a set of reference instances and corresponding line art images, our approach enables coloring sketches while maintaining consistency across multiple instances. Compared to traditional methods, our approach significantly improves coloring efficiency.

![](./asset/teaser.jpg)


## Set up

### Environment

    conda create -n MagicColor python=3.8

    pip install -r requirements.txt

### Repository

    git clone https://github.com/YinHan-Zhang/MagicColor.git
    
    cd MagicColor

Use tools to automatically extract masks:

    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

Install Grounded-Segment-Anything see here -> [install guideline](https://github.com/IDEA-Research/Grounded-Segment-Anything)

    mv automatic_label.py Grounded-Segment-Anything/

    cd Grounded-Segment-Anything

    mkdir ckpt  
    
    # move sam/ram/groundingdino weight to ckpt dir


Then, you can train the model on your dataset:

    python automatic_label.py \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --ram_checkpoint ./ckpt/ram_swin_large_14m.pth \
        --grounded_checkpoint ./ckpt/groundingdino_swint_ogc.pth \
        --sam_checkpoint ./ckpt/sam_vit_h_4b8939.pth \
        --data_dir  ./data \
        --output_dir ./data_res \
        --box_threshold 0.15 \
        --text_threshold 0.15 \
        --iou_threshold 0.15 \
        --device "cuda"

### Download pre-trained weight

    bash download.sh

### Dataset(optional)

- [Sakuga Dataset](https://github.com/KytraScript/SakugaDataset)
- [ATD-12K](https://github.com/lisiyao21/AnimeInterp)
- [Anime-sketch-colorization-pair](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair)

### Train
Dataset Format:

    data/
    ├── dir_name1
        ├──masks/
            ├── mask_1.png # instance mask
            ├── mask_2.png
            ├── ...
            ├── mask_n.png

        ├── dir_name1.jpg # origin image 1

    ├── dir_name2
        ├──masks/
            ├── mask_1.png # instance mask
            ├── mask_2.png
            ├── ...
            ├── mask_n.png

        ├── dir_name2.jpg # origin image 2

then,

    cd scripts

    bash multi_ref_train.sh

### Inference
Dataset Format:

    data/
    ├── dir_name
        ├──masks/
            ├── mask_1.png # reference mask
            ├── mask_2.png
            ├── ...
            ├── mask_n.png

        ├── dir_name_1.jpg # reference instance
        ├── dir_name_2.jpg
        ├── ...
        ├── dir_name_n.jpg

        ├── dir_name.jpg  # sketch image

then, 

    cd scripts
    
    bash multi_ref_infer.sh # modify input_data_dir


### Interface
run the script:
 
    cd inference
    python gradio_app.py


## Limitation

Due to the limitation of computing resources and data, the amount of data for model training is limited. If you have enough computing resources, you can train the model yourself.

## Acknowledgement

Thanks for the reference contributions of these works: 
- MangaNinjia
- ColorizeDiffusion
- DreamBooth