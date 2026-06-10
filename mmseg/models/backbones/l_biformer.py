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
from ..utils.bra_legacy import BiLevelRoutingAttention
from mmseg.registry import MODELS
from timm.models.layers import LayerNorm2d


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

        # 注意力计算
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
        x = x + self.pos_embed(x)   # 通过conv加上位置偏置
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:   # 前置归一化
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

@MODELS.register_module()
class l_BiFormer(nn.Module):
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
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.norm_eval = norm_eval

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        # 位置编码
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,  # 层缩放初始化值，用于 Layer Scale 技术（类似 ResNet 的残差缩放）
                        topk=topks[i],  # 选择前 top-k 个最相关注意力 token（动态稀疏注意力）。-2 表示全连接
                        num_heads=nheads[i],  # 注意力头数
                        n_win=n_win,  # 局部窗口的大小
                        qk_dim=qk_dims[i],  # Q（查询）和 K（键）的维度。控制注意力计算的特征空间维度。
                        qk_scale=qk_scale,  # 缩放系数，用于稳定注意力的 softmax（一般为 1/√d）
                        kv_per_win=kv_per_wins[i],  # 每个窗口采样的 K/V 数量，用于稀疏注意力。-1 表示全量使用
                        kv_downsample_ratio=kv_downsample_ratios[i],  # 对 K/V 特征下采样的比例，减少计算量
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,  # 下采样模式
                        param_attention=param_attention,  # 是否使用参数化的注意力
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
            x = self.downsample_layers[i](x)  # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
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
@MODELS.register_module()
class l_BiFormer_mm(l_BiFormer):
    def __init__(self, pretrained=None, **kwargs):
        # # ----------------------------
        # # 删除不属于 BiFormer 的多余参数
        # # ----------------------------
        # drop_keys = [
        #     'num_heads', 'mlp_ratio', 'out_indices', 'qkv_bias',
        #     'qk_scale', 'patch_norm', 'frozen_stages', 'init_cfg'
        # ]
        # for k in drop_keys:
        #     kwargs.pop(k, None)

        # 确保 init_cfg 被移除
        if 'init_cfg' in kwargs:
            init_cfg = kwargs.pop('init_cfg')
            # 如果需要保留 init_cfg 用于后续权重初始化
            self.init_cfg = init_cfg

        super().__init__(**kwargs)

        # step 1: remove unused segmentation head & norm
        del self.head  # classification head
        del self.norm  # head norm

        # step 2: add extra norms for dense tasks
        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))

        # step 3: initialization & load ckpt
        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)

        # step 4: convert sync bn, as the batch size is too small in segmentation
        # TODO: check if this is correct
        nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            # 从路径加载预训练权重
            print(f'Loading pretrained weights from {pretrained}')
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            # 默认初始化
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        else:
            raise TypeError(f'pretrained must be a str or None, but got {type(pretrained)}')

    def forward_features(self, x: torch.Tensor):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # DONE: check the inconsistency -> no effect on performance
            # in the version before submission:
            # x = self.extra_norms[i](x)
            # out.append(x)
            out.append(self.extra_norms[i](x))
        return tuple(out)

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)

    def train(self, mode=True):
        super(l_BiFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()