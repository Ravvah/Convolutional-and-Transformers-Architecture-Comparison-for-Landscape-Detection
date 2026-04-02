from torch import Tensor
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
import torch.nn as nn
import timm






class ViT(nn.Module):
    """
    Class for ViT model
    """

    def __init__(self, num_classes: int = 4, freeze_backbone: bool = False):
        super().__init__()
        # self.model = vit_b_16(weights = ViT_B_16_Weights.DEFAULT)
        self.model = timm.create_model("deit_small_patch16_224", pretrained=True)


        in_features = self.model.head.in_features

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.head.parameters():
                param.requires_grad = True
        self.model.head = nn.Linear(in_features, num_classes)



    def forward(self, x:Tensor):
        return self.model(x)