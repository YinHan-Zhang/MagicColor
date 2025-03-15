# MagicColor: Multi-instance Sketch Colorization
![](./asset/logo.png)

## News

- 2025-3-15: 🎊 source code release !

## Overview
We present MagicColor, a diffusion-based framework for multi-instance sketch colorization. Previous methods can only achieve multi-instance sketch colorization step by step, which is time-consuming and inaccurate. In contrast, our method has the capability of coloring sketches while maintaining consistency, making multi-instance sketch colorization easier.

![](./asset/intro.jpg)

### Gallery
Given a set of references and corresponding line art image, our approach enables coloring sketch while maintaining consistency across multiple instances. Compared to traditional methods, our approach significantly improves coloring efficiency.

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

### Download pretrain weight

    bash download.sh

### Train
    cd scripts

    bash multi_ref_train.sh

### Inference 
    cd scripts
    
    bash multi_ref_infer.sh

### Interface
 run the script:

    python gradio_app.py

The gradio demo would look like the UI shown below.

![](./asset/UI.jpg)