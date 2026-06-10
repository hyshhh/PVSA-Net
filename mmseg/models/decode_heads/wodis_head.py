import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class AttentionRefinementModule(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        att = self.avgpool(x)
        att = self.conv(att)
        att = self.sigmoid(att)

        return x * att


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class FeatureFusionModule(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.convblock = ConvBlock(in_channels, out_channels)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(out_channels, out_channels, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        feat = self.convblock(x)

        att = self.avgpool(feat)

        att = self.conv1(att)
        att = self.relu(att)

        att = self.conv2(att)
        att = self.sigmoid(att)

        out = feat * att + feat

        return out


@MODELS.register_module()
class WODISHead(BaseDecodeHead):

    def __init__(self, **kwargs):

        super().__init__(
            input_transform='multiple_select',
            **kwargs)

        self.arm1 = AttentionRefinementModule(2048)
        self.arm2 = AttentionRefinementModule(512)

        self.ffm = FeatureFusionModule(
            in_channels=2048 + 512 + 512,
            out_channels=self.channels
        )

    def forward(self, inputs):

        f1, f2, f3, f4 = inputs

        arm1 = self.arm1(f4)

        arm1 = F.interpolate(
            arm1,
            size=f2.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        arm2 = self.arm2(f2)

        arm12 = torch.cat((arm1, arm2), dim=1)

        fusion = torch.cat((arm12, f2), dim=1)

        out = self.ffm(fusion)

        out = self.cls_seg(out)

        return out