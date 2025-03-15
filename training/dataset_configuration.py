import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")



from torch.utils.data import DataLoader
from dataloader import transforms
import os

# Get Dataset Here
def prepare_dataset(
    dataset_name,
    datapath=None,
    trainlist=None,
    vallist=None,
    height=336,
    width=596,
    stride=16,
    batch_size=1,
    test_batch=1,
    datathread=4,
    logger=None,
    collate_fn=None
):
    
    # set the config parameters
    dataset_config_dict = dict()

    if dataset_name == 'sakuga':
        from dataloader.video_loader import PairDataset as Dataset
    elif dataset_name == 'pair':
        from dataloader.image_pair_loader import PairDataset as Dataset
    elif dataset_name == 'multi':
        from dataloader.image_multipair_loader_add import PairDataset as Dataset
    elif dataset_name == 'no_instance':
        from dataloader.image_multipair_loader_no_instance import PairDataset as Dataset
    else:
        from dataloader.video_loader import PairDataset as Dataset

    train_transform_list = [
        transforms.RandomHorizontalFlip(p=0.9),
        transforms.ToPILImage(),
        transforms.ToNumpyArray(),
        transforms.ToTensor(),
    ]
    train_transform = transforms.Compose(train_transform_list)

    # val_transform_list = [
    #     transforms.ToTensor(),
    # ]
    val_transform = transforms.Compose(train_transform_list)        
        
    train_dataset = Dataset(
        data_dir=datapath,
        dataset_name=dataset_name,
        datalist=trainlist,
        height=height,
        width=width,
        stride=stride,
        transform=train_transform,
    )
    if vallist:
        test_dataset = Dataset(
            data_dir=vallist, # vallist as path, multi_ref
            dataset_name=dataset_name,
            datalist=vallist,
            height=height,
            width=width,
            stride=stride,
            transform=val_transform,
        )
    else:
        test_dataset = None

    img_height, img_width = train_dataset.get_img_size()

    # datathread=0
    # if os.environ.get('datathread') is not None:
    #     datathread = int(os.environ.get('datathread'))
    
    if logger is not None:
        logger.info("Use %d processes to load data..." % datathread)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True, collate_fn=collate_fn)

    if vallist:
        test_loader = DataLoader(test_dataset, batch_size = test_batch, \
                                shuffle = False)
    else:
        test_loader = None
    
    num_batches_per_epoch = len(train_loader)
    
    dataset_config_dict['num_batches_per_epoch'] = num_batches_per_epoch
    dataset_config_dict['img_size'] = (img_height,img_width)
    
    
    return (train_loader,test_loader),dataset_config_dict

def Disparity_Normalization(disparity):
    min_value = torch.min(disparity)
    max_value = torch.max(disparity)
    normalized_disparity = ((disparity -min_value)/(max_value-min_value+1e-5) - 0.5) * 2    
    return normalized_disparity

def resize_max_res_tensor(input_tensor,is_disp=False,recom_resolution=768):
    assert input_tensor.shape[1]==3
    original_H, original_W = input_tensor.shape[2:]
    
    downscale_factor = min(recom_resolution/original_H,
                           recom_resolution/original_W)
    
    resized_input_tensor = F.interpolate(input_tensor,
                                         scale_factor=downscale_factor,mode='bilinear',
                                         align_corners=False)
    
    if is_disp:
        return resized_input_tensor * downscale_factor
    else:
        return resized_input_tensor