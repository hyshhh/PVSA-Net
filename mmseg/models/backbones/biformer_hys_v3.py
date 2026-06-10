import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from ..utils.common import Attention, AttentionLePE, DWConv
from ..utils.bra_legacy_hys_v3 import BiLevelRoutingAttention
from mmseg.registry import MODELS



def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    # if name == 'sum':
    #     return Summer(PositionalEncodingPermute2D(emb_dim))
    # elif name == 'npe.sin':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
    # elif name == 'npe.coord':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
    # elif name == 'hpe.conv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
    # elif name == 'hpe.dsconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
    # elif name == 'hpe.pointconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        # 如果在注意力前加入卷积核：
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x
class FeatureAlignmentModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureAlignmentModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2
from mmengine.model import BaseModule, ModuleList, Sequential   
from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
class DepthWiseConvModule(BaseModule):
    """Depth-wise Conv Module with optional residual connection."""

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 output_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 use_shortcut=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.activate = build_activation_layer(act_cfg)
        self.use_shortcut = use_shortcut
        self.stride = stride
        self.in_ch = embed_dims
        self.out_ch = output_channels

        # 1x1升维
        self.fc1 = nn.Conv2d(embed_dims, feedforward_channels, 1, 1, bias=True)
        # depthwise卷积
        self.dw_conv = nn.Conv2d(
            feedforward_channels,
            feedforward_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=feedforward_channels,
            bias=True)
        # 1x1降维/输出
        self.fc2 = nn.Conv2d(feedforward_channels, output_channels, 1, 1, bias=True)
        # dropout
        self.drop = nn.Dropout(ffn_drop)

        # shortcut
        if self.use_shortcut:
            if stride != 1 or embed_dims != output_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(embed_dims, output_channels, 1, stride=stride, bias=True),
                    nn.BatchNorm2d(output_channels)
                )
            else:
                self.shortcut = nn.Identity()
        else:
            self.shortcut = None

        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dw_conv(out)
        out = self.activate(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = self.bn(out)

        if self.shortcut is not None:
            out = out + self.shortcut(x)

        return out


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#自适应平均池化，(B, 96, 256, 256) → (B, 96, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_avg = nn.Sequential(
                    nn.Linear(self.dim, self.dim),#如果我的输入向量是96，但是全连接层在
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim, 2))
        self.mlp_max = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim, 2))
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim, self.dim),
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # print("!!!!!!!!!!!!")
        # print(B, C, H, W)#(1,12,256,256)
        x = torch.cat((x1, x2), dim=1)
        # print("a")
        # print(x.shape)

        # Avg. Adaptive normalization
        avg = self.avg_pool(x).view(B, 2 * C)
        # print("b")
        # print("avg shape:", avg.shape)
        avg_attn = self.mlp_avg(avg).softmax(dim=-1)
        avg_x1, avg_x2 = (avg_attn.view(B, 2, 1) * avg.view(B, 2, C)).chunk(2, dim=1)
        avg_x = (avg_x1 + avg_x2).view(B, C)

        # Max. Adaptive normalization
        max = self.max_pool(x).view(B, 2 * C)
        max_attn = self.mlp_max(max).softmax(dim=-1)
        max_x1, max_x2 = (max_attn.view(B, 2, 1) * max.view(B, 2, C)).chunk(2, dim=1)
        max_x = (max_x1 + max_x2).view(B, C)

        y = torch.cat((avg_x, max_x), dim=1)
        y = self.mlp(y).view(B, self.dim, 1)
        channel_weights = y.reshape(B, 2, C, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights

class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=1), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights
class Stem224(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128):
        super().__init__()
        # 使用改进版 DepthWiseConvModule 内部支持 shortcut
        self.stem = nn.Sequential(
            DepthWiseConvModule(in_chans, embed_dim//2, embed_dim//2, kernel_size=3, stride=2, padding=1, use_shortcut=True),
            DepthWiseConvModule(embed_dim//2, embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1, use_shortcut=True)
        )
    def forward(self, x):
        x = self.stem(x)
        return x
    
class BiFormer(nn.Module):
    def __init__(self, depth=[3, 4, 8, 3], in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 # -----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False,
                 norm_eval=False):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.norm_eval = norm_eval
        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers2 = nn.ModuleList()
        self.FAM = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem2 =Stem224(in_chans, embed_dim[0])
        pool1=nn.Conv2d(in_chans, embed_dim[0], kernel_size=(4, 4), stride=(4,4), padding=(0, 0))
        if (pe is not None) and 0 in pe_stages:
            stem2.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
            pool1.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem2 = checkpoint_wrapper(stem2)
            pool1= checkpoint_wrapper(pool1)
        self.downsample_layers2.append(stem2)
        self.pool_layers.append(pool1)
        self.FAM.append(FeatureAlignmentModule(dim=2*embed_dim[0], reduction=1))
        for i in range(3):
            downsample_layer2 = DepthWiseConvModule(embed_dims=embed_dim[i],feedforward_channels=embed_dim[i + 1] // 2,output_channels=embed_dim[i + 1],
        kernel_size=3,stride=2, padding=1,act_cfg=dict(type='GELU'),ffn_drop=0.,use_shortcut=True  # 自动残差 + 通道对齐
    )
            pool1 = nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)) # 第一次平均池化，下采样 ×0.5
            
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer2.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
                pool1.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer2 = checkpoint_wrapper(downsample_layer2)
                pool1= checkpoint_wrapper(pool1)
            self.downsample_layers2.append(downsample_layer2)
            self.pool_layers.append(pool1)
            self.FAM.append(FeatureAlignmentModule(dim=2*embed_dim[i + 1], reduction=1))

        ##########################################################################

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        topks=[16,12,8,6]
        n_win=[7,7,7,7]
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,  # 层缩放初始化值，用于 Layer Scale 技术（类似 ResNet 的残差缩放）
                        topk=topks[i],  # 选择前 top-k 个最相关注意力 token（动态稀疏注意力）。-2 表示全连接
                        num_heads=nheads[i],  # 注意力头数
                        n_win=n_win[i],  # 局部窗口的大小
                        qk_dim=qk_dims[i],  # Q（查询）和 K（键）的维度。控制注意力计算的特征空间维度。
                        qk_scale=qk_scale,  # 缩放系数，用于稳定注意力的 softmax（一般为 1/√d）
                        kv_per_win=kv_per_wins[i],  # 每个窗口采样的 K/V 数量，用于稀疏注意力。-1 表示全量使用
                        kv_downsample_ratio=kv_downsample_ratios[i],  # 对 K/V 特征下采样的比例，减少计算量
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,  # 下采样模式
                        param_attention=param_attention,  # 是否使用参数化的注意力s
                        param_routing=param_routing,  # 是否启用参数化路由。控制 token 之间的信息流方向。
                        diff_routing=diff_routing,  # 是否启用可微分的路由机制
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],  # MLP隐藏层扩展比， 表示MLP内部维度是输入维度的几倍
                        mlp_dwconv=mlp_dwconv,  # 是否在MLP中使用depth-wise卷积增强局部特征
                        side_dwconv=side_dwconv,  # 在注意力旁路分支中使用的 depth-wise 卷积核大小。增强局部特征感受野。
                        before_attn_dwconv=before_attn_dwconv,  # 在注意力前加入的卷积核大小，用于特征增强
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],  # 是否自动为卷积层补零，使得输出尺寸与输入一致
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers2[i](x)  # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x = self.stages[i](x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        return x

    def train(self, mode=True):
        super(BiFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

#################### model variants #######################
