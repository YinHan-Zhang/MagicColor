import torch
import torch.nn as nn

class Projection(nn.Module):
    def __init__(self):
        super(Projection, self).__init__()
        self.mapping_layer = nn.Linear(768, 1024)
    
    def forward(self, x):
        x = self.mapping_layer(x)
        return x