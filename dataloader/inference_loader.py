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

from dataloader import transforms
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T
import cv2

class InferPairDataset(Dataset):
    def __init__(
            self, 
            data_dir,
            dataset_name,
            datalist,
            height,
            width,
            stride,
            transform=None,
        ):

        self.frame_folder = [data_dir]
        self.length = len(self.frame_folder)
        print(f" data from -> {data_dir}, load data total length: {self.length}")

        self.height = height
        self.width = width
        self.stride = stride

        if transform is not None:
            self.transform = transform
        else:
            transform_list = [
                transforms.RandomHorizontalFlip(p=0.9),
                transforms.ToPILImage(),
                transforms.ToNumpyArray(),
                transforms.ToTensor(),
            ]
            self.transform = transforms.Compose(transform_list)
            
        self.clip_image_processor = CLIPImageProcessor()
        self.patch_size = 504
        self.dino_transforms_noflip = A.Compose(
            [
                A.Resize(self.patch_size, self.patch_size),
                # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
                ToTensorV2(),
            ]
        )
        self.image_transforms = T.Compose([T.ToTensor()])

    def get_data_dict(self, index):
        curr_anno = self.frame_folder[index]  # dataset dir name
       
        img1_path = curr_anno + f"/sketch.jpg"
        img2_path = curr_anno + f"/sketch.jpg"
       
        sample = {}
        sample['img2'] = np.array(Image.open(img2_path).convert('RGB'))

        mul_pixel_values = []
        mul_ctrl_img = []
        mul_patches = []

        file_name = curr_anno + f"/sketch.jpg"

        curr_height = self.height
        curr_width = self.width

        img = np.zeros((curr_height, curr_width, 3), dtype=np.uint8)  
        local_img = np.zeros((curr_height, curr_width, 3), dtype=np.uint8)  
        img_mask = np.zeros_like(np.array(img, dtype=np.uint8))[:, :, 0]
        reloc_img_mask = np.zeros_like(np.array(img, dtype=np.uint8))

        ins_rgb = []  # rgb img
        ins_rgb_id = []  # rgb img id

        ins_num = len(os.listdir(curr_anno + "/masks"))  # mask num

        occupied_regions = []
        
        for p_id in range(ins_num):
            instance_path = f"{curr_anno}/{p_id+1}.jpg"
            ins_img = np.array(Image.open(instance_path).convert("RGB").resize((curr_width, curr_height)), dtype=np.uint8) # load instance img
            mask_loca = curr_anno + f"/masks/mask_{p_id+1}.png"  # load mask img
            ins_mask = np.array(Image.open(mask_loca).resize((curr_width, curr_height)), dtype=np.uint8)
            
            y_indices, x_indices = np.where(ins_mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            mask_size = (y_max - y_min, x_max - x_min)

            target_size = self.adaptive_scale_size(ins_num, mask_size)
            instance = ins_img[y_min:y_max, x_min:x_max, :]
           
            instance = self.normalize_and_convert(instance)
            resized_instance = cv2.resize(instance, target_size[::-1])

            placed = False
            for y_start in range(curr_height - target_size[0] + 1):
                for x_start in range(curr_width - target_size[1] + 1):
                    overlap = False
                    new_region = [(y_start, x_start), (y_start + target_size[0], x_start + target_size[1])]
                    for occupied_region in occupied_regions:
                        if self.check_overlap(new_region, occupied_region):
                            overlap = True
                            break
                    if not overlap:
                        local_ins = np.zeros_like(local_img, dtype=np.uint8)
                        local_ins_mask = np.zeros_like(local_img, dtype=np.uint8)
                        local_ins[y_start:y_start + target_size[0], x_start:x_start + target_size[1], :] = resized_instance
                        local_ins_mask[y_start:y_start + target_size[0], x_start:x_start + target_size[1], :] = p_id + 1
                        
                        reloc_img_mask =  cv2.add(reloc_img_mask, local_ins_mask)
                        local_img = cv2.add(local_img, local_ins)
                    
                        placed = True
                        occupied_regions.append(new_region)
                        break
                if placed:
                    break

        
            if os.path.exists(instance_path):
                ins_rgb.append(ins_img) 
                ins_rgb_id.append(p_id + 1)
            else:
                ins_rgb.append(Image.open(mask_loca).convert("RGB"))  

        print(f"load {len(ins_rgb)} instances ....")
        local_img = Image.fromarray(local_img.astype(np.uint8))
        

        sample["img1"] = np.array(local_img)
        sample = self.transform(sample)
        
        sample['img1'] = F.interpolate(sample['img1'][None], size=(self.height, self.width), mode='bilinear', align_corners=False)[0]
        sample['img2'] = F.interpolate(sample['img2'][None], size=(self.height, self.width), mode='bilinear', align_corners=False)[0]

        sample['clip_images'] = self.clip_image_processor(
            images=np.uint8(((sample["img1"].numpy().transpose(1, 2, 0) + 1) / 2) * 255),
            return_tensors="pt"
        ).pixel_values[0]  # [H, W, 3]

        reloc_img_mask = Image.fromarray(reloc_img_mask[:, :, 0])  

        patches = []
        use_patch = 1
        if use_patch:
            sel_ins = []  # index
            all_ins = ins_rgb_id  
           
            for id_ins, curr_ins in enumerate(all_ins):
                cropped_img = ins_rgb[curr_ins - 1]
                cropped_img = self.dino_transforms_noflip(image=np.array(cropped_img))
                cropped_img = cropped_img["image"] / 255
                patches.append(cropped_img[None])
                sel_ins.append(curr_ins)

            if len(patches) > 0:
                patches = torch.cat(patches, dim=0)
            else:
                patches = torch.zeros(
                    (0, 3, self.patch_size, self.patch_size)
                )

            edges = (
                    cv2.Canny(
                        np.array(Image.open(img1_path).convert('RGB').resize((curr_width // 8, curr_height // 8))),
                        100,
                        200,
                    )
                    / 255
            )

            patches = [
                torch.tensor(sel_ins),
                patches,
                edges,
                file_name,
                (curr_width, curr_height),
            ]

        reloc_img_mask = reloc_img_mask.resize((curr_width, curr_height), Image.NEAREST)

        fea_mask = torch.tensor(np.array(reloc_img_mask))

        mul_ctrl_img.append(fea_mask[None])
        mul_patches.append(patches)

        mul_ctrl_img = torch.cat(mul_ctrl_img, 0)

        data_dict = {
            "img1": sample['img1'],
            "img2": sample['img2'],
            "clip_images": sample['clip_images'],
            "ctrl_img": mul_ctrl_img,  
            "patches": mul_patches,
        }

        return data_dict
        

    def check_overlap(self, region1, region2):
        
        y1_min, x1_min = region1[0]
        y1_max, x1_max = region1[1]
        y2_min, x2_min = region2[0]
        y2_max, x2_max = region2[1]

        return (y1_min < y2_max and y1_max > y2_min) and (x1_min < x2_max and x1_max > x2_min)

    def adaptive_scale_size(self, instance_num, mask_size=(512, 512)):
       
        if instance_num <= 5:
            max_size = 256
        else:
            max_size = 150
            
        h, w = mask_size
        if h > w:
            ratio = max_size / h 
        else:
            ratio = max_size / w 
            
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        return (new_h, new_w)
           

    def normalize_and_convert(self, instance):
        instance = (instance - np.min(instance)) / (np.max(instance) - np.min(instance))
        instance = (instance * 255).astype(np.uint8)

        return instance

    def __getitem__(self, idx):
        
        sample = self.get_data_dict(idx)
        if sample is None:
            return self.__getitem__(idx+1)
        else:
            return sample

    def __len__(self):
        return self.length

    def get_img_size(self):
        return (self.height, self.width)