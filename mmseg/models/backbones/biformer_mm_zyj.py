from mmseg.registry import MODELS
import torch
import torch.nn as nn
from .biformer_hys import BiFormer
from timm.models.layers import LayerNorm2d
from mmengine.runner import load_checkpoint

@MODELS.register_module()
class BiFormer_mm_zyj(BiFormer):
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
        cnn_encoder_out=x
        for i in range(4):
            cnn_encoder_out = self.downsample_layers2[i](cnn_encoder_out)
            x = self.pool_layers[i](x)
            x = self.stages[i](x)
            x,cnn_encoder_out=self.FAM[i](x,cnn_encoder_out)
            # x = torch.cat((x, cnn_encoder_out), dim=1)
            # x = self.fusion[i](x)
            # DONE: check the inconsistency -> no effect on performance
            # in the version before submission:
            # x = self.extra_norms[i](x)
            # out.append(x)
            out.append(self.extra_norms[i](x))
        return tuple(out)

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)

    def train(self, mode=True):
        super(BiFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
