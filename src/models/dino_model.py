import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as T
import open_clip
from PIL import Image
from open_clip.transform import image_transform
import sys

from .dinov2 import hubconf

# DINOv2_weight_path = 'dinov2_vitl14_reg4_pretrain.pth'

class FrozenDinoV2Encoder(nn.Module):
    """
    Uses the DINOv2 encoder for image
    """
    def __init__(self, DINOv2_weight_path, device="cuda", freeze=True):
        super().__init__()

        dinov2 = hubconf.dinov2_vitl14_reg(pretrained=False) 
        state_dict = torch.load(DINOv2_weight_path, map_location='cpu')
        dinov2.load_state_dict(state_dict, strict=False)
        self.model = dinov2.to(device, dtype=torch.float16)
        
        self.device = device
        if freeze:
            self.freeze()
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_std =  torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)     


    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image,list):
            image = torch.cat(image,0)

        image = (image.to(self.device)  - self.image_mean.to(self.device)) / self.image_std.to(self.device)
        features = self.model.forward_features(image)
        reg_tokens = features["x_norm_regtokens"]
        patch_tokens = features["x_norm_patchtokens"]
        image_features  = features["x_norm_clstoken"].unsqueeze(1)
        
        hint = torch.cat([image_features, patch_tokens],1) # 8,257,1024
        # hint = self.projector(hint)
        return hint

    def encode(self, image):
        return self(image)

if __name__ == '__main__':
    torch.cuda.set_device(0)
    model = FrozenDinoV2Encoder(device='cuda',freeze=True)
    image = torch.randn(1,3,224,224)
    hint = model(image)
