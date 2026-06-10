from mmseg.registry import MODELS
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# from .hys_Nex_21_6 import BiFormer
from .biformer import BiFormer
from timm.models.layers import LayerNorm2d
from mmengine.runner import load_checkpoint
import os
import numpy as np
import cv2 
from PIL import Image
@MODELS.register_module()
class BiFormer_mm(BiFormer):
    def __init__(self, pretrained=None, **kwargs):
        super().__init__(**kwargs)
        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))
        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)
        nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            print(f'Loading pretrained weights from {pretrained}')
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
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
        cnn_encoder_out = x

        # 保存图片的目录
        flag=0
        save_dir = 'cam/features_imgs2'
        os.makedirs(save_dir, exist_ok=True)
        for i in range(4):
            self._save_feature_channel_as_image(x, f'{save_dir}/stage{i}_xinput.png')
            cnn_encoder_out = self.downsample_layers2[i](cnn_encoder_out)
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if flag==1:
                # 保存 FAM 前特征图（第0通道）
                self._save_feature_channel_as_image(x, f'{save_dir}/stage{i}_before_FAM_x.png')
                self._save_feature_channel_as_image(cnn_encoder_out, f'{save_dir}/stage{i}_before_FAM_cnn.png')

            x, cnn_encoder_out = self.FAM[i](x, cnn_encoder_out)
            if flag==1:
            # 保存 FAM 后特征图（第0通道）
                self._save_feature_channel_as_image(x, f'{save_dir}/stage{i}_after_FAM_x.png')
                self._save_feature_channel_as_image(cnn_encoder_out, f'{save_dir}/stage{i}_after_FAM_cnn.png')

            out.append(self.extra_norms[i](x+cnn_encoder_out))
        return tuple(out)

    # def _save_feature_channel_as_image(self, feature_map, file_path, sigma=1.0):
    #     """
    #     feature_map: [B, C, H, W] 或 [C, H, W]
    #     file_path: 保存文件路径
    #     sigma: 高斯平滑的标准差
    #     """
    #     if feature_map.dim() == 4:
    #         feature_map = feature_map[0]  # 取 batch 第0个
    #     fmap = feature_map[0]  # 取第0通道
    #     fmap = fmap.detach().cpu().numpy()

    #     # 归一化到 [0,1]
    #     fmap = fmap - fmap.min()
    #     fmap = fmap / (fmap.max() + 1e-5)

    #     # 高斯平滑，减少颗粒感
    #     fmap = cv2.GaussianBlur(fmap, (0, 0), sigmaX=sigma, sigmaY=sigma)

    #     # 使用 matplotlib colormap 转彩色
    #     cmap = plt.get_cmap('viridis')
    #     img_color = (cmap(fmap)[:, :, :3] * 255).astype(np.uint8)

    #     # 保存图片
    #     Image.fromarray(img_color).save(file_path)
    def _save_feature_channel_as_image(self, feature_map, file_path):
        """
        feature_map: [B, C, H, W] 或 [C, H, W]
        file_path: 保存文件路径
        """
        if feature_map.dim() == 4:
            feature_map = feature_map[0]  # 取 batch 第0个
        fmap = feature_map[0]  # 取第0通道
        fmap = fmap.detach().cpu().numpy()

        # 归一化到 [0,1]
        fmap = fmap - fmap.min()
        fmap = fmap / (fmap.max() + 1e-5)

        # 轻度高斯平滑，核大小可调，小一些避免过度模糊
        fmap = cv2.GaussianBlur(fmap, ksize=(3, 3), sigmaX=0.5, sigmaY=0.5)

        # 使用 matplotlib colormap 转成彩色
        cmap = plt.get_cmap('viridis')
        img_color = (cmap(fmap)[:, :, :3] * 255).astype(np.uint8)

        # 保存图片
        Image.fromarray(img_color).save(file_path)

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)

    def train(self, mode=True):
        super(BiFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
