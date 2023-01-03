import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import pretrainedmodels
import segmentation_models_pytorch as smp

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class EfficientNetB3(nn.Module):
    def __init__(self, resnet=False):
        super(EfficientNetB3, self).__init__()
        
        if resnet:
            self.model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
        else:
            self.model = efficientnet_b3(pretrained=True, weights=EfficientNet_B3_Weights.DEFAULT)
                    
    def forward(self, x):
        out = F.softmax(self.model(x))
        out_sum = torch.sum(out, dim=0)
        out_sum = out_sum.unsqueeze(0)
        return out_sum
    
class SegmentationUnetModel(nn.Module):
    def __init__(self, encoder):
        super(SegmentationUnetModel, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        )
    
    def forward(self, x):
        out = self.model(x)
        return out

