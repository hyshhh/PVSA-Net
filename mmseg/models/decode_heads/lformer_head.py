# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite
import math
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange

from mmcv.cnn import build_norm_layer
from mmcv.cnn import ConvModule, NonLocal2d
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


class MLP(nn.Module):
    """
    Linear Embedding: github.com/NVlabs/SegFormer
    """

    def __init__(self, input_dim=2048, embed_dim=768, identity=False):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        if identity:
            self.proj = nn.Identity()

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 1).reshape(n, -1, h, w)

        return x

class Mlp_decoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.linear_c4 = MLP(self.in_channels[-1], embed_dim//4)
        self.linear_c3 = MLP(self.in_channels[2], embed_dim//4)
        self.linear_c2 = MLP(self.in_channels[1], embed_dim//4)

        self.linear_fuse = ConvModule(
            in_channels= embed_dim//4 * 3,
            out_channels=embed_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True))


    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        # 通道数整合成embed_dim
        _c4 = self.linear_c4(c4)  # (n, c, 32, 32)
        _c3 = self.linear_c3(c3)
        _c2 = self.linear_c2(c2)

        _c4 = resize(_c4, size=inputs[1].size()[2:], mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=inputs[1].size()[2:], mode='bilinear', align_corners=False)
        # _c2 = resize(_c2, size=inputs[stride-1].size()[2:],mode='bilinear',align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))  # (n, c, 128, 128)

        return _c, c1

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class LKC(nn.Module):
    def __init__(self, embed_dim, norm_cfg):
        super().__init__()
        # branches with different kernel sizes
        self.dw7 = ConvModule(embed_dim, embed_dim, kernel_size=7, padding=3,
                              groups=embed_dim, norm_cfg=norm_cfg, act_cfg=None)
        self.dw5 = ConvModule(embed_dim, embed_dim, kernel_size=5, padding=2,
                              groups=embed_dim, norm_cfg=norm_cfg, act_cfg=None)
        self.dw3 = ConvModule(embed_dim, embed_dim, kernel_size=3, padding=1,
                              groups=embed_dim, norm_cfg=norm_cfg, act_cfg=None)

        # pointwise fusion
        self.fuse = ConvModule(embed_dim, embed_dim, kernel_size=1,
                               padding=0, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))

    def forward(self, x):
        b1 = self.dw7(x)
        b2 = self.dw5(x)
        b3 = self.dw3(x)
        out = b1 + b2 + b3
        out = self.fuse(out)
        return out

class Metaformer(nn.Module):
    def __init__(self, embed_dim, norm_cfg, align_corners):
        super().__init__()
        self.align_corners = align_corners

        self.act = h_sigmoid()

        # self.image_pool = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     ConvModule(embed_dim, embed_dim, 1, norm_cfg=norm_cfg))
        self.lck = LKC(embed_dim=embed_dim, norm_cfg=norm_cfg)

        self.dwconv = ConvModule(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=embed_dim,  # 深度卷积
            norm_cfg=norm_cfg,  # 统一用传入的norm_cfg（如SyncBN）
            act_cfg=dict(type='ReLU')  # 移除inplace=True，避免梯度风险
        )

        self.channel_interaction = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 16),
            nn.ReLU(),
            nn.Linear(embed_dim // 16, embed_dim)
        )

        self.spatial_interaction = nn.Sequential(
            ConvModule(
                in_channels=embed_dim,
                out_channels=embed_dim//16,
                kernel_size=1,
                padding=0,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')  # 移除inplace=True
            ),
            nn.Conv2d(
                embed_dim//16,
                embed_dim//16,
                kernel_size=3,
                padding=1,
                groups=embed_dim//16,
                bias=False  # 无BN时建议关闭bias，减少参数
            ),
            nn.Conv2d(embed_dim//16, 1, kernel_size=1, bias=False)
        )

        # learnable fusion
        self.gamma = nn.Parameter(torch.ones(1) * 1e-2)

    def forward(self, _c):
        local_x = self.dwconv(_c)

        # global_x = resize(self.image_pool(_c),
        #                  size=_c.size()[2:],
        #                  mode='bilinear',
        #                  align_corners=self.align_corners)
        global_x = self.lck(_c)

        # -------- Spatial attention --------
        spatial_map = self.act(self.spatial_interaction(local_x))  # [B, 1, H, W]
        global_x = global_x * spatial_map

        # -------- Channel attention --------
        ch = global_x.mean(dim=[2, 3])  # global average pooling
        ch = self.channel_interaction(ch).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        channel_map = self.act(ch)
        local_x = local_x * channel_map

        output = local_x + self.gamma * global_x

        return output


    
@MODELS.register_module()
class L_light_head(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.mlp_decoder = Mlp_decoder(
            in_channels=self.in_channels,
            embed_dim=self.channels)

        self.metaformer = Metaformer(
            embed_dim=self.channels,
            norm_cfg=self.norm_cfg, align_corners=self.align_corners)

        # self.linear_c1 = MLP(input_dim=self.in_channels[0], embed_dim=48)
        self.linear_c1 = nn.Conv2d(self.in_channels[0], 32, kernel_size=1, bias=False)
        self.low_level_fuse = nn.Sequential(
            # 可选：这里也替换为ConvModule
            ConvModule(
                in_channels=self.channels + 32,
                out_channels=self.channels // 8,
                kernel_size=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=None
            ),
            ConvModule(
                in_channels=self.channels // 8,
                out_channels=self.channels // 8,
                kernel_size=3,
                padding=1,
                groups=self.channels // 8,
                norm_cfg=None,  # DWConv后无BN
                act_cfg=None
            ),
            ConvModule(
                in_channels=self.channels // 8,
                out_channels=self.channels,
                kernel_size=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='ReLU')
            )
        )

    def low_level(self, _c, c1):
        _c1 = self.linear_c1(c1)
        output = resize(_c,
                        size=c1.size()[2:],
                        mode=self.interpolate_mode,
                        align_corners=False)
        output = self.low_level_fuse(torch.cat([output, _c1], dim=1))

        return output

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        _c, c1 = self.mlp_decoder(inputs)
        _c = self.metaformer(_c)
        _c = self.low_level(_c, c1)
        out = self.cls_seg(_c)


        return out
