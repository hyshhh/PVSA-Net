from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

# class WindowAttention(nn.Module):
#     def __init__(self, dim, n_win, num_heads=8, dropout=0.1, auto_pad=True):
#         """
#         :param dim: 输入的特征维度
#         :param n_win: 每个窗口的大小 (height, width)
#         :param num_heads: 多头注意力的数量
#         :param dropout: Dropout 概率
#         :param auto_pad: 是否自动填充以适应窗口尺寸
#         """
#         super(WindowAttention, self).__init__()
#         self.dim = dim
#         self.n_win = n_win
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.auto_pad = auto_pad
#         # QKV projections
#         self.qkv = nn.Linear(dim, dim * 3)  # 输出三个维度，分别对应Q、K、V       
#         # Position encoding (Depthwise Separable Conv)
#         self.lepe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
#         # Attention activation (softmax or other activation)
#         self.attn_act = nn.Softmax(dim=-1)
#     def forward(self, x):
#         """
#         :param x: 输入数据 (N, H, W, C)
#         :return: 输出数据 (N, H, W, C)
#         """
#         N, H_in, W_in, C = x.size()
#         if self.auto_pad:
#             # 自动填充输入以使其适应窗口大小
#             pad_l = pad_t = 0
#             pad_r = (self.n_win - W_in % self.n_win) % self.n_win
#             pad_b = (self.n_win - H_in % self.n_win) % self.n_win
#             x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
#             _, H, W, _ = x.size()  # padded size
#         else:
#             H, W = H_in, W_in
#             assert H % self.n_win == 0 and W % self.n_win == 0
#         # 将图像切分成窗口
#         x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)
#         # QKV 投影
#         qkv = self.qkv(x)
#         q, k, v = qkv.chunk(3, dim=-1)
#         # 展开 q、k 和 v 到像素级
#         q_pix = rearrange(q, "n p2 h w c -> n p2 (h w) c")
#         k_pix = rearrange(k, "n p2 h w c -> n p2 (h w) c")
#         v_pix = rearrange(v, "n p2 h w c -> n p2 (h w) c")
#         attn_weights = (q_pix @ k_pix.transpose(-2, -1)) * (self.dim ** -0.5) 
#         attn_weights = self.attn_act(attn_weights)
#         attn_out = attn_weights @ v_pix  # (n, p^2, h*w, c)
#         # 恢复窗口形状
#         out = rearrange(attn_out, "n (j i) (h w) c -> n (j h) (i w) (c)", j=self.n_win, i=self.n_win, h=H // self.n_win, w=W // self.n_win)
#         # 位置编码
#         # lepe = self.lepe(rearrange(k[..., self.dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
#         # lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)
#         out = out
#         return out
class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)
        self.topk=15

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        # 嵌入变换（线性变换）
        query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c)
        # 注意力分数矩阵计算Ar = Qr .* Kr^T
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)
        # print("hh",attn_logit[0][0])

        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1,sorted=True)  # (n, p^2, k), (n, p^2, k)
        # print("输出-1",topk_attn_logit[0][0])

        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index, attn_logit


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':  #添加加权操作
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v

import torch
import torch.nn as nn

# class BlockMLP(nn.Module):
#     def __init__(self, in_dim, n, mapratio, ratio, out, dropout=0.0):
#         super().__init__()

#         # --------- 维度计算 ---------
#         c1 = in_dim // n * mapratio[0]
#         c2 = in_dim // n * mapratio[0]
#         c3 = in_dim // n * (n - mapratio[1])

#         self.c_list = [c1, c2, c3]

#         h_list = [c // ratio for c in self.c_list]
#         self.h_list = h_list

#         self.in_dim = sum(self.c_list)
#         self.hidden_dim = sum(h_list)
#         self.out_dim = sum(out)

#         # --------- 统一 MLP ---------
#         self.fc1 = nn.Linear(self.in_dim, self.hidden_dim, bias=False)
#         self.fc2 = nn.Linear(self.hidden_dim, self.out_dim, bias=False)

#         self.act = nn.GELU()
#         self.drop = nn.Dropout(dropout)

#         self._init_block_weights(out)

#     def _init_block_weights(self, out):
#         with torch.no_grad():
#             # fc1: input -> hidden
#             self.fc1.weight.zero_()
#             i, o = 0, 0
#             for ci, hi in zip(self.c_list, self.h_list):
#                 self.fc1.weight[o:o+hi, i:i+ci].normal_(std=0.02)
#                 i += ci
#                 o += hi

#             # fc2: hidden -> output
#             self.fc2.weight.zero_()
#             i, o = 0, 0
#             for hi, oi in zip(self.h_list, out):
#                 self.fc2.weight[o:o+oi, i:i+hi].normal_(std=0.02)
#                 i += hi
#                 o += oi
#     def forward(self, x):
#         # x: (B, N, in_dim)
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
def build_mlp(in_dim, out_dim, hidden_dim=None, dropout=0.):
    if hidden_dim is None:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

class TopKDimMLP(nn.Module):
    def __init__(self, img_size,topk,usepool,mlp_ratio=0.5, dropout=0.1,v=2):
        super().__init__()
        img_size = 224
        self.topk = topk
        topk_to_base = {
            16: 8,
            12: 4,
            8: 2,
            6: 1
        }
        #优化，配置各个层的最终映射维度

        topk_to_base2 = {
            # 16: 10,  #8*6*8  64*6————10*7       #VTFormerv1.3更换
            16: 8,  #8*6*8  32*6————8*7
            12: 6,  #4*4*6   4*6————6*7
            8: 2,  #2*2*6   2*6————2*7
            6: 1  #1*1*6    1*6————1*7
        }
        topk_to_base3 = {
            16: 2,  #8*6*8  64*6————12*14
            12: 1,  #4*4*6   16*6————8*14
            8: 1,  #2*2*6   4*6————4*14
            6: 1  #1*1*6    1*6————1*14
        }
        self.usepool=usepool
        if self.usepool:
            topk_to_poolratio = {
                16: 2,  #8*6*8  64*6————12*14
                12: 2,  #4*4*6   16*6————8*14
                8: 2,  #2*2*6   4*6————4*14
                6: 1  #1*1*6    1*6————1*14
            }
            if topk in topk_to_poolratio:
                self.poolratio=topk_to_poolratio[topk]
        else:
            self.poolratio=1
        if topk in topk_to_base3:
            self.ratio=topk_to_base3[topk]
        if topk in topk_to_base2:
            z=topk_to_base2[topk]

        out=[3*z,2*z,1*z]
        self.basek=15
        if self.basek==21:
            self.n=6
            self.mapratio=[2,4]
        if self.basek==15:
            self.n=5
            self.mapratio=[2,4]
        if self.basek==10:
            self.n=4
            self.mapratio=[2,3]
        self.v=v

        # z=2
        # out=[32*z,16*z,8*z]

        if topk in topk_to_base:
            base = topk_to_base[topk]
            downsample = (32//base) * 7
            # self.in_dim = (img_size // downsample) ** 2 * 49  计算量太大117flop   
            self.in_dim=(img_size // downsample) **2 * self.n//self.poolratio

            base_dim = self.in_dim // self.n
            dim_a = base_dim * self.mapratio[0]
            dim_b = base_dim * (self.mapratio[1]-self.mapratio[0])
            dim_c = base_dim * (self.n - self.mapratio[1])

            hidden_a = dim_a // self.ratio
            hidden_b = dim_b // self.ratio
#VTformer1.2-轻量化mlp
            if self.v == 2:
                self.mlp1 = build_mlp(dim_a, out[0], dropout=dropout)
                self.mlp2 = build_mlp(dim_b, out[1], dropout=dropout)
                self.mlp3 = build_mlp(dim_c, out[2], dropout=dropout)
#VTformer1.1-mlp
            elif self.v == 1:
                self.mlp1 = build_mlp(dim_a, out[0], hidden_a, dropout)
                self.mlp2 = build_mlp(dim_b, out[1], hidden_a, dropout)
                self.mlp3 = build_mlp(dim_c, out[2], hidden_b, dropout)
        else:

            self.in_dim = None
            self.out_dim = None
   

    def downsample_avgpool(self, x, factor):
        """
        x: [B, N, C, L]
        factor: 下采样倍数
        """
        B, N, C, L = x.shape
        L_trim = (L // factor) * factor
        x = x[..., :L_trim]                # 保证能整除
        x = x.reshape(B * N * C, 1, L_trim)
        x = F.avg_pool1d(
            x,
            kernel_size=factor,
            stride=factor,
            ceil_mode=False
        )
        x = x.view(B, N, C, -1)
        return x

    def downsample_mean(self, x, factor):
        """
        x: [B, N, L, K]
        factor: 下采样倍数，比如2、7、14 
        """
        B, N, L, K = x.shape
        assert K % factor == 0, "最后一维度不能整除下采样倍数"
        x = x.view(B, N, L, K//factor, factor)  # 切成 factor 块
        x = x.mean(dim=4)  # 对 factor 块求均值
        return x
    def forward(self, x):
        """
        输入形状: (B, N, K)
        只有当 topk ∈ {1, 4, 16} 时进行 MLP 
        """
        B, N, K, L= x.shape
        assert K == self.in_dim*self.basek//self.n, f"输入形状与初始化不符: L={K}, in_dim={self.in_dim*self.basek//self.n}"
#分段自适应池化
        x = x.permute(0, 1,3,2).contiguous()  # -> (B, M, C, L)
        B, N, K, L= x.shape
        x0 = x[:, :, :,:L//self.basek]
        x1 = x[:, :, :,L//self.basek:L//self.basek*3]
        x2 = x[:, :, :,L//self.basek*3:L//self.basek*6]
        if self.basek>=10:
            x3 = x[:, :,:, L//self.basek*6:L//self.basek*10]
        if self.basek>=15:
            x4 = x[:, :, :,L//self.basek*10:L//self.basek*15]
        if self.basek>=21:
            x5 = x[:, :, :,L//self.basek*15:]
        # x1_ds = self.downsample_avgpool(x1, 2)
        # x2_ds = self.downsample_avgpool(x2, 3)
        # x3_ds = self.downsample_avgpool(x3, 4)
        # x4_ds = self.downsample_avgpool(x4, 5)
        # x5_ds = self.downsample_avgpool(x5, 6)
        features = [x0]
        x1_ds = self.downsample_mean(x1, 2)
        features.append(x1_ds)
        x2_ds = self.downsample_mean(x2, 3)   # 3倍下采样
        features.append(x2_ds)
        if self.basek>=10:
            x3_ds = self.downsample_mean(x3, 4)   # 4倍下采样
            features.append(x3_ds)
        if self.basek>=15:
            x4_ds = self.downsample_mean(x4, 5)
            features.append(x4_ds)
        if self.basek>=21:
            print(1)
            x5_ds = self.downsample_mean(x5, 6)  # 14倍下采样
            features.append(x5_ds)
        x_out = torch.cat(features, dim=-1)
        B, N, K, L= x_out.shape
        x1 = x[:, :, :, :L//self.n*self.mapratio[0]]
        x2 = x[:, :, :, L//self.n*self.mapratio[0]:L//self.n*(self.mapratio[1])]
        x3 = x[:, :, :,L//self.n*(self.mapratio[1]):L]
        x1 = self.mlp1(x1) 
        x2 = self.mlp2(x2) 
        x3 = self.mlp3(x3) 
        x_out = torch.cat([x1, x2, x3], dim=-1) 
        
        # x_out=self.mlp(x_out)
        # x_out = torch.cat([x1, x2, x3], dim=-1) 
        x_out = x_out.permute(0, 1,3,2).contiguous()  
        return x_out

    
class BiLevelRoutingAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights
    """

    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing)  # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':  # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':

            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')
        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)
        self.auto_pad = auto_pad
        self.usepool=False
        self.MLP=TopKDimMLP(224, topk, self.usepool,mlp_ratio=0.5, dropout=0.1)

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
        # NOTE: use padding for semantic segmentation
        # 输入填充处理
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0,  # dim=-1
                          pad_l, pad_r,  # dim=-2
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x.size()  # padded size
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0  #
        ###################################################

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x)

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        # 对值部分应用深度可分离卷积作为位置编码
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################

        # 路由机制
        r_weight, r_idx,self.GA = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors
        n, p2, hw, c = kv_pix.shape
        if self.usepool:
            if hw!=1:
                kv_pool = kv_pix.permute(0, 1, 3, 2)  
                kv_pool = kv_pool.reshape(n * p2, c, hw)
                    # (n, p², c, hw)
                kv_pool = F.avg_pool1d(
                    kv_pool,
                    kernel_size=2,
                    stride=2
                )  # (n*p², c, hw/2)
                kv_pix = kv_pool.reshape(n, p2, c, hw // 2).permute(0, 1, 3, 2)
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)


        ######### do attention as normal ####################
        # k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
        #                       m=self.num_heads)
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)   # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        k_pix_sel=self.MLP(k_pix_sel)
        k_pix_sel = k_pix_sel.permute(0, 1,3,2).contiguous()
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        v_pix_sel=self.MLP(v_pix_sel)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention    —— 注意力计算
        attn_weight = (q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        # else:F
        #     return out
        else:
            return out, self.GA