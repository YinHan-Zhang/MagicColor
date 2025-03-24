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
from tqdm import tqdm
from decord import VideoReader
import sys
sys.path.append("../dataloader")
import transforms
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
        with open(datalist, 'r') as f:
            self.videos = f.readlines()
        self.videos = [v.strip() for v in self.videos]
        self.length = len(self.videos)
        print(f" data from -> {data_dir}, load video total length: {self.length}")

        self.video_folder = data_dir
        self.height = height
        self.width = width
        self.stride = stride
        self.transform = transform
        self.clip_image_processor = CLIPImageProcessor()

    def get_batch(self, idx):
        try:   
            video_path = os.path.join(self.video_folder, self.videos[idx])
            # print(video_path)
            with open(video_path, 'rb') as f:
                video_reader = VideoReader(f)
            video_length = len(video_reader)
            
            clip_length = min(self.stride + 1, video_length - 1)
            if video_length - 1 == clip_length:
                start_idx = 0
            else:
                start_idx = random.randint(0, video_length - clip_length - 1)
            batch_index = np.array([start_idx, start_idx + clip_length])
            clip = video_reader.get_batch(batch_index).asnumpy()
            del video_reader

            sample = {}
            sample['img1'] = clip[0]
            sample['img2'] = clip[1]

            sample = self.transform(sample)

            sample['img1'] = F.interpolate(sample['img1'][None], size=(self.height, self.width), mode='bilinear', align_corners=False)[0]
            sample['img2'] = F.interpolate(sample['img2'][None], size=(self.height, self.width), mode='bilinear', align_corners=False)[0]
            img2 = Image.fromarray(np.uint8(((sample['img2'].numpy().transpose(1,2,0) + 1) / 2) * 255))
            sample['edge'] = torch.Tensor(cv2.Canny(np.array(img2.resize((self.height // 8, self.width // 8))), 100 , 200,) / 255)
            sample['clip_images'] = self.clip_image_processor(
                images=Image.fromarray(np.uint8(((sample['img1'].numpy().transpose(1,2,0) + 1) / 2) * 255)),
                return_tensors="pt"
            ).pixel_values[0] # [H, W, 3]

            return sample
        except:
            return None

    def __getitem__(self, idx):

        sample = self.get_batch(idx)
        if sample is not None:
            return sample
        else:
            return self.__getitem__(idx+1)

    def __len__(self):
        return self.length

    def get_img_size(self):
        return (self.height, self.width)