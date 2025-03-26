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
from panopticapi.utils import rgb2id
from pycocotools.coco import COCO

class PairDataset(Dataset):
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
        folder = Path(data_dir)
        self.frame_folder = [os.path.join(data_dir,str(x.name)) for x in folder.iterdir() if x.is_dir()]
        self.length = len(self.frame_folder)
        # print(f" data from -> {data_dir}, load video total length: {self.length}")

        # coconut dataset
        import json
        self.coco_dataset = []
        self.dataroot = './data/coconut_dataset/coco'
        
        self.dataType = 'train2017'
        # 标注文件路径
        annFile = os.path.join(self.dataroot, 'annotations', f'instances_{self.dataType}.json')
        self.img_dir = f"{self.dataroot}/train2017"
        self.coco = COCO(annFile)

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

        prob = random.uniform(0, 1)
        if prob < 0.4:
            extract = True
            self.anime_data += 1
            curr_anno = self.frame_folder[index]  # dataset dir name
            if not os.path.exists(curr_anno + "/masks"):
                return None
            # modify: ref file name
            base_name = curr_anno.split("/")[-1]
            img1_path = curr_anno + f"/{base_name}.jpg"
            if os.path.exists(img1_path):
                img2_path = img1_path
            else:
                img1_path = curr_anno + f"/frame1.png" # ref1
                img2_path = curr_anno + f"/frame3.png" # gt2
            
            sample = {}
            sample['img2'] = np.array(Image.open(img2_path).convert('RGB'))

            # 初始化相关数据结构
            mul_pixel_values = []
            mul_ctrl_img = []
            mul_patches = []

            file_name = curr_anno.split("/")[-1] + ".png"  # output file name

            curr_height = self.height
            curr_width = self.width

            img = np.zeros((curr_height, curr_width, 3), dtype=np.uint8)  # 初始化黑色图像
            local_img = np.zeros((curr_height, curr_width, 3), dtype=np.uint8)  # 初始化白色图像
            img_mask = np.zeros_like(np.array(img, dtype=np.uint8))[:, :, 0]

            ins_rgb = []  # rgb img
            ins_rgb_id = []  # rgb img id

            ins_num = len(os.listdir(curr_anno + "/masks"))  # mask num
            if ins_num <= 0:
                return None

            occupied_regions = []
            try:
                for p_id in range(ins_num):
                    ins_path = curr_anno + f"/{base_name}_{p_id+1}.jpg"
                    if os.path.exists(ins_path):
                        ins_img = np.array(Image.open(ins_path).convert("RGB").resize((curr_width, curr_height), Image.NEAREST))
                        mask_loca = curr_anno + f"/masks/mask_{p_id+1}.png"
                    else:
                        ins_path = img1_path
                        ins_img = np.array(Image.open(ins_path).convert("RGB").resize((curr_width, curr_height), Image.NEAREST))
                        mask_loca = curr_anno + f"/masks/mask_{p_id}.png"

                    ins_mask = np.array(Image.open(mask_loca).resize((curr_width, curr_height), Image.NEAREST))
                
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
                                local_ins[y_start:y_start + target_size[0], x_start:x_start + target_size[1], :] = resized_instance
                                
                                # local_img = local_img + local_ins
                                local_img = cv2.add(local_img, local_ins)
                            
                                placed = True
                                occupied_regions.append(new_region)
                                break
                        if placed:
                            break

                    if len(ins_mask.shape)!= 2:
                        ins_mask = ins_mask[:, :, 0]
                    
                    _, ins_mask = cv2.threshold(ins_mask, 127, 255, cv2.THRESH_BINARY)
                    ins_mask = (ins_mask / 255).astype(np.uint8)
                    
                    ins_mask_expanded = np.expand_dims(ins_mask, axis=-1)
                    ins_mask_expanded = np.repeat(ins_mask_expanded, ins_img.shape[-1], axis=-1)

                    ori_instance = ins_img * ins_mask_expanded
                   

                    img_mask[ins_mask!= 0] = p_id + 1 


                    if os.path.exists(ins_path):
                        ins_rgb.append(ori_instance) 
                        ins_rgb_id.append(p_id + 1)
                    else:
                        ins_rgb.append(Image.open(mask_loca).convert("RGB"))
                    
                    if ins_num == 1: 
                        ins_rgb.append(ins_img)
                        ins_rgb_id.append(p_id + 1)

                prob_ = random.uniform(0, 1)
                if prob_ < 0.3:
                    sample["img1"] = np.array(Image.open(img1_path).convert('RGB'))
                else:
                    sample["img1"] = np.array(local_img)
                sample = self.transform(sample)

                sample['img1'] = F.interpolate(sample['img1'][None], size=(self.height, self.width), mode='bilinear', align_corners=False)[0]
                sample['img2'] = F.interpolate(sample['img2'][None], size=(self.height, self.width), mode='bilinear', align_corners=False)[0]

                sample['clip_images'] = self.clip_image_processor(
                    images=np.uint8(((sample["img1"].numpy().transpose(1, 2, 0) + 1) / 2) * 255),
                    return_tensors="pt"
                ).pixel_values[0] 

                img_mask = Image.fromarray(img_mask) 

                patches = []
                use_patch = 1
                if use_patch:
                    sel_ins = []
                    img_np_raw = np.array(img, dtype=np.uint8)  
                    mask_np_raw = np.array(img_mask, dtype=np.uint8)

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

                img_mask = img_mask.resize((curr_width, curr_height), Image.NEAREST)

                img = self.image_transforms(img) 

                fea_mask = torch.tensor(np.array(img_mask))

                mul_ctrl_img.append(fea_mask[None])
                mul_patches.append(patches)
                mul_ctrl_img = torch.cat(mul_ctrl_img, 0)

                data_dict = {
                    "img1": sample['img1'],
                    "img2": sample['img2'],
                    "clip_images": sample['clip_images'],
                    "ctrl_img": mul_ctrl_img,
                    "patches": mul_patches,
                    "extract": extract
                }

                return data_dict
                
            except Exception as e:
                print(e)
                return None
        elif 0.4 < prob < 0.85:
            if prob < 0.55:
                extract = False
            else:
                extract = True
            
            data_dir = "./data/new_data"
            case_folder = Path(data_dir)
            case_dir = [os.path.join(data_dir,str(x.name)) for x in case_folder.iterdir() if x.is_dir()]
            idx = random.randint(0, len(case_dir)-1)
            curr_anno = case_dir[idx] 

            if not os.path.exists(curr_anno + "/masks"):
                return None
          
            base_name = curr_anno.split("/")[-1]
            
            img1_path = curr_anno + f"/frame1.png" 
            img2_path = curr_anno + f"/frame3.png" 
            
            sample = {}
            sample['img2'] = np.array(Image.open(img2_path).convert('RGB'))

            mul_ctrl_img = []
            mul_patches = []

            file_name = curr_anno.split("/")[-1] + ".png"  # output file name

            curr_height = self.height
            curr_width = self.width

            img = np.zeros((curr_height, curr_width, 3), dtype=np.uint8)  
            local_img = np.zeros((curr_height, curr_width, 3), dtype=np.uint8) 
            img_mask = np.zeros_like(np.array(img, dtype=np.uint8))[:, :, 0]

            ins_rgb = []  
            ins_rgb_id = []  

            ins_num = len(os.listdir(curr_anno + "/masks"))  # mask num
            if ins_num <= 0:
                return None

            occupied_regions = []
            try:
                for p_id in range(ins_num):
                    ins_path = img1_path
                    ins_img = np.array(Image.open(ins_path).convert("RGB").resize((curr_width, curr_height), Image.NEAREST))
                    mask_loca = curr_anno + f"/masks/mask_{p_id}.png"

                    ins_mask = np.array(Image.open(mask_loca).resize((curr_width, curr_height), Image.NEAREST))
                
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
                                local_ins[y_start:y_start + target_size[0], x_start:x_start + target_size[1], :] = resized_instance
                            
                                local_img = cv2.add(local_img, local_ins)
                            
                                placed = True
                                occupied_regions.append(new_region)
                                break
                        if placed:
                            break

                    if len(ins_mask.shape)!= 2:
                        ins_mask = ins_mask[:, :, 0]
                    
           
                    _, ins_mask = cv2.threshold(ins_mask, 127, 255, cv2.THRESH_BINARY)
                    ins_mask = (ins_mask / 255).astype(np.uint8)
                   
                    ins_mask_expanded = np.expand_dims(ins_mask, axis=-1)
                    ins_mask_expanded = np.repeat(ins_mask_expanded, ins_img.shape[-1], axis=-1)
                  
                    ori_instance = ins_img * ins_mask_expanded
                    img_mask[ins_mask!= 0] = p_id + 1 
                    if os.path.exists(ins_path):
                        ins_rgb.append(ori_instance) 
                        ins_rgb_id.append(p_id + 1)
                    else:
                        ins_rgb.append(Image.open(mask_loca).convert("RGB"))
                    
                    if ins_num == 1:
                        ins_rgb.append(ins_img)
                        ins_rgb_id.append(p_id + 1)

                prob_ = random.uniform(0, 1)
                if prob_ < 0.3:
                    sample["img1"] = np.array(Image.open(img1_path).convert('RGB'))
                else:
                    sample["img1"] = np.array(local_img)
                sample = self.transform(sample)

                sample['img1'] = F.interpolate(sample['img1'][None], size=(self.height, self.width), mode='bilinear', align_corners=False)[0]
                sample['img2'] = F.interpolate(sample['img2'][None], size=(self.height, self.width), mode='bilinear', align_corners=False)[0]

                sample['clip_images'] = self.clip_image_processor(
                    images=np.uint8(((sample["img1"].numpy().transpose(1, 2, 0) + 1) / 2) * 255),
                    return_tensors="pt"
                ).pixel_values[0]  # [H, W, 3]

                img_mask = Image.fromarray(img_mask) 

                patches = []
                use_patch = 1
                if use_patch:
                    sel_ins = []  
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

                img_mask = img_mask.resize((curr_width, curr_height), Image.NEAREST)
                fea_mask = torch.tensor(np.array(img_mask))

                mul_ctrl_img.append(fea_mask[None])
                mul_patches.append(patches)
                mul_ctrl_img = torch.cat(mul_ctrl_img, 0)

                data_dict = {
                    "img1": sample['img1'],
                    "img2": sample['img2'],
                    "clip_images": sample['clip_images'],
                    "ctrl_img": mul_ctrl_img,  
                    "patches": mul_patches,
                    "extract": extract
                }

                return data_dict
                
            except Exception as e:
                print(e)
                return None

        else: # coconut dataset
            try:
                self.coconut_data += 1
                imgIds = self.coco.getImgIds()
                random_img_id = random.choice(imgIds)
                img_id = random_img_id 
                curr_width, curr_height = self.width , self.height
                img_info = self.coco.loadImgs(img_id)[0]
                img_path = os.path.join(self.img_dir, img_info['file_name'])
                coco_img = Image.open(img_path).convert("RGB")
                coco_img = coco_img.resize((curr_width, curr_height), Image.NEAREST)
               
                annIds = self.coco.getAnnIds(imgIds=img_info['id'])
                anns = self.coco.loadAnns(annIds)
            except Exception as e:
                print(e)
                return None

            sample = {}
            sample['img2'] = np.array(Image.open(img_path).convert('RGB'))

            mul_ctrl_img = []
            mul_patches = []

            img = np.zeros((curr_height, curr_width, 3), dtype=np.uint8)  
            local_img = np.zeros((curr_height, curr_width, 3), dtype=np.uint8)  
            img_mask = np.zeros_like(np.array(img, dtype=np.uint8))[:, :, 0]

            ins_rgb = []  # rgb img
            ins_rgb_id = []  # rgb img id
            if len(anns) <= 0:
                return None
            ins_num = len(anns) if len(anns) < 10 else 10
           
            occupied_regions = []
            try:
                for p_id, ann in enumerate(anns):
                    if p_id == 10:
                        break
                    ins_img = np.array(coco_img)
                    instance_mask = self.coco.annToMask(ann)
                
                    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                    
                    mask = np.maximum(mask, instance_mask * 255) 
                    ins_mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((curr_width, curr_height), Image.NEAREST))

                    y_indices, x_indices = np.where(ins_mask > 0)
                    if len(y_indices) == 0 or len(x_indices) == 0:
                        continue
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    mask_size = (y_max - y_min, x_max - x_min)

                    target_size = self.adaptive_scale_size(ins_num, mask_size)
                    instance = ins_img[y_min:y_max, x_min:x_max, :]               
                    instance = self.normalize_and_convert(instance)
                    resized_instance = cv2.resize(instance, target_size[::-1]).astype(np.uint8)
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
                                local_ins[y_start:y_start + target_size[0], x_start:x_start + target_size[1], :] = resized_instance
                                local_img = cv2.add(local_img, local_ins)
                            
                                placed = True
                                occupied_regions.append(new_region)
                                break
                        if placed:
                            break

                    if len(ins_mask.shape)!= 2:
                        ins_mask = ins_mask[:, :, 0]
                 
                    _, ins_mask = cv2.threshold(ins_mask, 127, 255, cv2.THRESH_BINARY)
                    ins_mask = (ins_mask / 255).astype(np.uint8)
                   
                    ins_mask_expanded = np.expand_dims(ins_mask, axis=-1)
                    ins_mask_expanded = np.repeat(ins_mask_expanded, ins_img.shape[-1], axis=-1)
                    ori_instance = ins_img * ins_mask_expanded
                    img_mask[ins_mask!= 0] = p_id + 1  

                    if os.path.exists(img_path):
                        ins_rgb.append(ins_img)
                        ins_rgb_id.append(p_id + 1)
                    else:
                        ins_rgb.append(Image.open(ins_mask).convert("RGB"))  
                    
                    if len(anns) == 1: 
                        ins_rgb.append(ins_img)
                        ins_rgb_id.append(p_id + 1)

                sample["img1"] = np.array(local_img)
                sample = self.transform(sample)

                sample['img1'] = F.interpolate(sample['img1'][None], size=(self.height, self.width), mode='bilinear', align_corners=False)[0]
                sample['img2'] = F.interpolate(sample['img2'][None], size=(self.height, self.width), mode='bilinear', align_corners=False)[0]

                sample['clip_images'] = self.clip_image_processor(
                    images=np.uint8(((sample["img1"].numpy().transpose(1, 2, 0) + 1) / 2) * 255),
                    return_tensors="pt"
                ).pixel_values[0]  # [H, W, 3]

                img_mask = Image.fromarray(img_mask)  

                patches = []
                use_patch = 1
                if use_patch:
                    sel_ins = []  
                    img_np_raw = np.array(img, dtype=np.uint8)  
                    mask_np_raw = np.array(img_mask, dtype=np.uint8)  

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
                                np.array(Image.open(img_path).convert('RGB').resize((curr_width // 8, curr_height // 8))),
                                100,
                                200,
                            )
                            / 255
                    )

                    patches = [
                        torch.tensor(sel_ins),
                        patches,
                        edges,
                        "",
                        (curr_width, curr_height),
                    ]
                img_mask = img_mask.resize((curr_width, curr_height), Image.NEAREST)
                fea_mask = torch.tensor(np.array(img_mask))

                mul_ctrl_img.append(fea_mask[None])
                mul_patches.append(patches)

                mul_ctrl_img = torch.cat(mul_ctrl_img, 0)

                data_dict = {
                    "img1": sample['img1'],
                    "img2": sample['img2'],
                    "clip_images": sample['clip_images'],
                    "ctrl_img": mul_ctrl_img, 
                    "patches": mul_patches,
                    "extract": True
                }

                return data_dict
                    
            except Exception as e:
                print(e)
                return None
       
    def check_overlap(self, region1, region2):
       
        y1_min, x1_min = region1[0]
        y1_max, x1_max = region1[1]
        y2_min, x2_min = region2[0]
        y2_max, x2_max = region2[1]

        return (y1_min < y2_max and y1_max > y2_min) and (x1_min < x2_max and x1_max > x2_min)

    def adaptive_scale_size(self, instance_num, mask_size=(256, 256)):
    
        if instance_num <= 4:
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
     
        denominator = np.max(instance) - np.min(instance)
        if denominator == 0:
            instance_normalized = np.zeros_like(instance)
        else:
            instance_normalized = (instance - np.min(instance)) / denominator
        instance_converted = (instance_normalized * 255).astype(np.uint8)

        return instance_converted

    def __getitem__(self, idx):
       
        sample = self.get_data_dict(idx)
        if sample is None:
            if idx < self.length - 1:
                return self.__getitem__(idx+1)
            else:
                return self.__getitem__(idx-2)
        else:
            return sample

    def __len__(self):
        return self.length

    def get_img_size(self):
        return (self.height, self.width)
