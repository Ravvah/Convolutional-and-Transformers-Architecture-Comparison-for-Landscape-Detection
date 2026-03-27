import torch
import torch.nn as nn
import torch.functional as F

from torchvision.models.resnet import resnet152
from torchvision.models.convnext import convnext_large
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import pytorch_lightning as pl



class CNN(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass


class Resnet:

    def __init__(self):
        self.model =resnet152()
        pass



class FasterRCNN(pl.LightningModule):

    def __init__(self, num_classes):
        pass
    
        



class FCN:

    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)

        
        