# MagicColor: Multi-Instance Sketch Colorization
![](./asset/logo.png)

## News

- 2025-3-15: 🎊 source code release !

## Overview
We present MagicColor, a diffusion-based framework for multi-instance sketch colorization. Previous methods can only achieve multi-instance sketch colorization step by step, which is time-consuming and inaccurate. In contrast, our method has the capability of coloring sketches while maintaining consistency, making multi-instance sketch colorization easier.

![](./asset/intro.jpg)

### Gallery
Given a set of references and corresponding line art images, our approach enables coloring sketches while maintaining consistency across multiple instances. Compared to traditional methods, our approach significantly improves coloring efficiency.

![](./asset/teaser.jpg)


## Set up

### Environment

    conda create -n MagicColor python=3.10  

    pip install -r requirements.txt

### Repository

    git clone https://github.com/YinHan-Zhang/MagicColor.git
    
    cd MagicColor

Use GSA to automatically extract masks:

    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

Install Grounded-Segment-Anything see here -> [install guideline](https://github.com/IDEA-Research/Grounded-Segment-Anything)

    mv automatic_label.py Grounded-Segment-Anything/

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

### Inference
Dataset Format:

data/
├── dir_name
    ├──masks/
        ├── mask_1.png
        ├── mask_2.png
        ├── ...
        ├── mask_n.jpg

    ├── dir_name_1.jpg
    ├── dir_name_2.jpg
    ├── ...
    ├── dir_name_2.jpg
    ├── sketch.jpg

then,

    cd scripts
    
    bash multi_ref_infer.sh

### Train
Dataset Format:

data/
├── dir_name
    ├──masks/
        ├── mask_1.png
        ├── mask_2.png
        ├── ...
        ├── mask_n.jpg

    ├── dir_name_1.jpg
    ├── dir_name_2.jpg
    ├── ...
    ├── dir_name_2.jpg
    ├── dir_name.jpg

then,

    cd scripts

    bash multi_ref_train.sh


### Interface
run the script:
 
    cd inference
    python gradio_app.py

The gradio demo would look like the UI shown below.

![](./asset/UI.jpg)
