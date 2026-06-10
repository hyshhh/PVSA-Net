import torch.nn as nn
from torchvision import models
from mmseg.registry import MODELS


@MODELS.register_module()
class WODISBackbone(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        resnet = models.resnet101(weights=None)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)

        f1 = self.layer1(x)   # 256
        f2 = self.layer2(f1)  # 512
        f3 = self.layer3(f2)  # 1024
        f4 = self.layer4(f3)  # 2048

        return [f1, f2, f3, f4]