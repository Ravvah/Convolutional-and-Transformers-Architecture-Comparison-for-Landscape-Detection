import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet152, resnet18
from torchvision.models.convnext import convnext_large
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import pytorch_lightning as pl



class ResNet(nn.Module):

    def __init__(self, num_classes: int = 4, freeze_backbone: bool = False):
        super().__init__()

        self.model = resnet18(progress=True, pretrained=True)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.fc.parameters():
                param.requires_grad = True

        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes)

    
    def forward(self, x:Tensor):
        return self.model(x)



class ConvNexT:
    pass
