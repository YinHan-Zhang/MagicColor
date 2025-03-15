from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import random
import torch
import os
from transformers import CLIPImageProcessor
import torch.nn.functional as F
import time
from pathlib import Path
import csv
import transforms
from pathlib import Path
import cv2
class PairDataset(Dataset):
    def __init__(
            self, 
            data_dir,
            dataset_name,
            datalist,
            height,
            width,
            stride,
            transform,
        ):
        folder = Path(data_dir)
        self.frame_folder = [os.path.join(data_dir,str(x.name)) for x in folder.iterdir() if x.is_dir()]
        
        self.length = len(self.frame_folder)
        print(f" data from -> {data_dir}, load video total length: {self.length}")

        self.height = height
        self.width = width
        self.stride = stride
        self.transform = transform
        self.clip_image_processor = CLIPImageProcessor()

    def get_batch(self, idx):
        
        frame_folder_path = self.frame_folder[idx]
        
        img1 = os.path.join(frame_folder_path, "frame1.jpg")
        img2 = os.path.join(frame_folder_path, "frame2.jpg")
        img3 = os.path.join(frame_folder_path, "frame3.jpg")

        sample = {}
        
        sample["img1"] = np.array(Image.open(img1).convert('RGB'))
        sample["img2"] = np.array(Image.open(img3).convert('RGB'))
        sample['edge'] = torch.Tensor(cv2.Canny(np.array(Image.open(img3).convert('RGB').resize((self.height // 8, self.width // 8))), 100 , 200,) / 255)
        sample = self.transform(sample)

        sample['img1'] = F.interpolate(sample['img1'][None], size=(self.height, self.width), mode='bilinear', align_corners=False)[0]
        sample['img2'] = F.interpolate(sample['img2'][None], size=(self.height, self.width), mode='bilinear', align_corners=False)[0]
        

        sample['clip_images'] = self.clip_image_processor(
            images=np.uint8(((sample['img1'].numpy().transpose(1,2,0) + 1) / 2) * 255),
            return_tensors="pt"
        ).pixel_values[0] # [H, W, 3]

        return sample

    def __getitem__(self, idx):

        sample = self.get_batch(idx)

        return sample

    def __len__(self):
        return self.length

    def get_img_size(self):
        return (self.height, self.width)
