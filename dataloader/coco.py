import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

dataDir = './data/coconut_dataset/coco'
dataType = 'train2017'
annFile = os.path.join(dataDir, 'annotations', f'instances_{dataType}.json')

coco = COCO(annFile)
imgIds = coco.getImgIds()
img_id = imgIds[0]  
img_info = coco.loadImgs(img_id)[0]

img_path = os.path.join(dataDir, dataType, img_info['file_name'])
img = Image.open(img_path).convert("RGB")
annIds = coco.getAnnIds(imgIds=img_info['id'])
anns = coco.loadAnns(annIds)

for idx, ann in enumerate(anns):

    instance_mask = coco.annToMask(ann)

    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

    mask = np.maximum(mask, instance_mask * 255)  
    mask = Image.fromarray(mask.astype(np.uint8)).convert('RGB')
    mask.save(f"./img_mask_{idx}.jpg")

img.save("./img.jpg")
