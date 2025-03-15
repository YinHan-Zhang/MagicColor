from __future__ import division
from genericpath import samefile
import torch
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as F
import random
import cv2
import matplotlib.pyplot as plt


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample



class ToTensor(object):
    """Convert numpy array to torch tensor"""
    def __call__(self, sample):
        img1 = np.transpose(sample['img1'], (2, 0, 1))  # [3, H, W]
        # print(img1.shape)
        sample['img1'] = torch.from_numpy(img1) / 255. * 2.0 - 1.0
        img2 = np.transpose(sample['img2'], (2, 0, 1))  # [3, H, W]
        sample['img2'] = torch.from_numpy(img2) / 255. * 2.0 - 1.0
        if 'img3' in sample:
            sample['img3'] = torch.from_numpy(img2) / 255. * 2.0 - 1.0
        # right = np.transpose(sample['img2_'], (2, 0, 1))
        # sample['img2_'] = torch.from_numpy(right) / 255. * 2.0 - 1.0

        return sample

# class Crop_center_square(image):
#     h, w, _ = image.shape
#     min_side = min(h, w)
    
#     # 计算中心点
#     center_x, center_y = w // 2, h // 2
    
#     # 计算裁剪的起始点（top-left corner）
#     start_x = center_x - min_side // 2
#     start_y = center_y - min_side // 2
    
#     # 裁剪
#     cropped_image = image[start_y:start_y + min_side, start_x:start_x + min_side]
class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['img1', 'img2', 'img2_']
        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample


class RandomCrop(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        

        ori_height, ori_width = sample['img1'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            sample['img1'] = np.lib.pad(sample['img1'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
            sample['img2'] = np.lib.pad(sample['img2'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
            sample['img2_'] = np.lib.pad(sample['img2_'],
                                         ((top_pad, 0), (0, right_pad), (0, 0)),
                                         mode='constant',
                                         constant_values=0)

        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width
            # Training: random crop
            if not self.validate:

                self.offset_x = np.random.randint(ori_width - self.img_width + 1)

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2

            sample['img1'] = self.crop_img(sample['img1'])
            sample['img2'] = self.crop_img(sample['img2'])
            sample['img2_'] = self.crop_img(sample['img2_'])
        
        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]


class RandomVerticalFlip(object):
    """Randomly vertically filps"""
    def __init__(self, p=0.09):
        self.p = p

    def __call__(self, sample):
        if np.random.random() < self.p:
            sample['img1'] = np.copy(np.flipud(sample['img1']))
            sample['img2'] = np.copy(np.flipud(sample['img2']))
            sample['img2_'] = np.copy(np.flipud(sample['img2_']))

        return sample


class RandomHorizontalFlip(object):
    """Randomly flip the image and depth map horizontally with a probability of 0.5."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            sample['img1'] = np.flip(sample['img1'], axis=1)
            # sample['img2'] = np.flip(sample['img2'], axis=1)
        if 'img3' in sample:
            sample['img3'] = np.flip(sample['img3'], axis=1)

        return sample


class ToPILImage(object):

    def __call__(self, sample):
        sample['img1'] = Image.fromarray(sample['img1'].astype('uint8'))
        sample['img2'] = Image.fromarray(sample['img2'].astype('uint8'))
        if 'img3' in sample:
            sample['img3'] = Image.fromarray(sample['img3'].astype('uint8'))
        # sample['img2_'] = Image.fromarray(sample['img2_'].astype('uint8'))

        return sample


class ToNumpyArray(object):

    def __call__(self, sample):
        sample['img1'] = np.array(sample['img1']).astype(np.float32)
        sample['img2'] = np.array(sample['img2']).astype(np.float32)
        if 'img3' in sample:
            sample['img3'] = np.array(sample['img3']).astype(np.float32)
        # sample['img2_'] = np.array(sample['img2_']).astype(np.float32)

        return sample


# Random coloring
class RandomContrast(object):
    """Random contrast"""
    def __init__(self, p1=0.5, p2=1.0):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, sample):
        if np.random.random() < self.p1:
            contrast_factor = np.random.uniform(0.8, 1.2)
            sample['img1'] = F.adjust_contrast(sample['img1'], contrast_factor)
            sample['img2'] = F.adjust_contrast(sample['img2'], contrast_factor)

        if np.random.random() < self.p2:
            contrast_factor = np.random.uniform(0.8, 1.2)
            sample['img2_'] = F.adjust_contrast(sample['img2_'], contrast_factor)

        return sample


class RandomGamma(object):
    def __init__(self, p1=0.5, p2=1.0):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, sample):
        if np.random.random() < self.p1:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['img1'] = F.adjust_gamma(sample['img1'], gamma)
            sample['img2'] = F.adjust_gamma(sample['img2'], gamma)

        if np.random.random() < self.p2:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['img2_'] = F.adjust_gamma(sample['img2_'], gamma)

        return sample


class RandomBrightness(object):
    def __init__(self, p1=0.5, p2=1.0):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, sample):
        if np.random.random() < self.p1:
            brightness = np.random.uniform(0.5, 2.0)

            sample['img1'] = F.adjust_brightness(sample['img1'], brightness)
            sample['img2'] = F.adjust_brightness(sample['img2'], brightness)

        if np.random.random() < self.p2:
            brightness = np.random.uniform(0.5, 2.0)

            sample['img2_'] = F.adjust_brightness(sample['img2_'], brightness)

        return sample


class RandomHue(object):
    def __init__(self, p1=0.5, p2=1.0):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, sample):
        if np.random.random() < self.p1:
            hue = np.random.uniform(-0.1, 0.1)

            sample['img1'] = F.adjust_hue(sample['img1'], hue)
            sample['img2'] = F.adjust_hue(sample['img2'], hue)

        if np.random.random() < self.p2:
            hue = np.random.uniform(-0.1, 0.1)

            sample['img2_'] = F.adjust_hue(sample['img2_'], hue)

        return sample


class RandomSaturation(object):
    def __init__(self, p1=0.5, p2=1.0):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, sample):
        if np.random.random() < self.p1:
            saturation = np.random.uniform(0.8, 1.2)
            sample['img1'] = F.adjust_saturation(sample['img1'], saturation)
            sample['img2'] = F.adjust_saturation(sample['img2'], saturation)

        if np.random.random() < self.p2:
            saturation = np.random.uniform(0.8, 1.2)
            sample['img2_'] = F.adjust_saturation(sample['img2_'], saturation)
        
        return sample


class RandomColor(object):
    def __init__(self, p=0.5, p1=0.5, p2=1.0):
        self.p = p
        self.transforms = [
            RandomContrast(p1, p2),
            RandomGamma(p1, p2),
            RandomBrightness(p1, p2),
            RandomHue(p1, p2),
            RandomSaturation(p1, p2)
        ]

    def __call__(self, sample):
        

        sample = ToPILImage()(sample)

        if np.random.random() < self.p:
            # A single transform
            t = random.choice(self.transforms)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(self.transforms)
            for t in self.transforms:
                sample = t(sample)

        sample = ToNumpyArray()(sample)

        return sample