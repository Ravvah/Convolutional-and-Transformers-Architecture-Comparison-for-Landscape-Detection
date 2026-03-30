from torchvision.models.vision_transformer import vit_l_32
import torch.nn as nn





class ViT(nn.Module):

    def __init__(self):
        super().__init__()
        pass



class PreTrainedViT:
    def __init__(self):
        self.model = vit_l_32()