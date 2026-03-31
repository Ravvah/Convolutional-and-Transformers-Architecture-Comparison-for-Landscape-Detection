from torch import Tensor
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
import torch.nn as nn






class ViT(nn.Module):

    def __init__(self, num_classes: int = 4, freeze_backbone: bool = False):
        super().__init__()
        self.model = vit_b_16(weights = ViT_B_16_Weights.DEFAULT)


        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.heads.parameters():
                param.requires_grad = True