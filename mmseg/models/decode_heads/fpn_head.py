# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmseg.registry import MODELS
from ..utils import Upsample, resize
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class FPNHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.obstacle_head = nn.Sequential(
            nn.Conv2d(self.channels, self.channels // 2, 3, padding=1, groups=self.channels //2),
            nn.BatchNorm2d(self.channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels // 2, 1, 1)
        )

        self.use_obstacle_reweight = True
        self.obstacle_reweight_alpha = 0.5  # 可调 0.5 / 1.0 / 2.0

    def forward_seghead(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        # output = self.cls_seg(output)
        return output

    def forward(self, inputs):
        feat = self.forward_seghead(inputs)
        # ---------- A1 ----------
        obstacle_logits = self.obstacle_head(feat)   # [B,1,h,w]
        obstacle_prob = torch.sigmoid(obstacle_logits)

        # ---------- A2 ----------
        if self.use_obstacle_reweight and self.training:
            feat = feat * (1.0 + self.obstacle_reweight_alpha * obstacle_prob.detach())

        # ---------- segmentation ----------
        seg_logits = self.cls_seg(feat)

        return seg_logits

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------
    def loss(self, inputs, batch_data_samples, train_cfg):
        # forward
        seg_logits = self.forward(inputs)
        feat = self.forward_seghead(inputs)
        obstacle_logits = self.obstacle_head(feat)  # [B,1,h,w]

        losses = dict()

        # segmentation loss（标准 mmseg）
        losses.update(self.loss_by_feat(seg_logits, batch_data_samples))

        # ========= 1. segmentation GT =========
        seg_gt = self._stack_batch_gt(batch_data_samples)  # [B, 1, H, W]

        # ========= 2. obstacle GT（正确做法） =========
        # 注意：不要 unsqueeze！！！
        obstacle_gt = (seg_gt == 2).float()  # [B, 1, H, W]

        assert obstacle_gt.dim() == 4, f"obstacle_gt shape wrong: {obstacle_gt.shape}"

        # ========= 3. resize =========
        obstacle_gt = F.interpolate(
            obstacle_gt,
            size=obstacle_logits.shape[2:],  # (h, w)
            mode='nearest'
        )

        # ========= 4. loss =========
        losses['loss_obstacle_map'] = F.binary_cross_entropy_with_logits(
            obstacle_logits,
            obstacle_gt
        )

        return losses

    # --------------------------------------------------
    # Prediction (ONLY for visualization / analysis)
    # --------------------------------------------------
    def predict(self, inputs, batch_img_metas, test_cfg):
        """Predict segmentation logits."""
        feat = self.forward_seghead(inputs)
        seg_logits = self.cls_seg(feat)

        # obstacle_logits = self.obstacle_head(feat)
        # obstacle_prob = torch.sigmoid(obstacle_logits)

        # if self.use_obstacle_reweight:
        #     feat = feat * (1.0 + self.obstacle_reweight_alpha * obstacle_prob)

        # seg_logits = self.cls_seg(feat)

        # resize to original image size
        seg_logits = resize(
            seg_logits,
            size=batch_img_metas[0]['ori_shape'],
            mode='bilinear',
            align_corners=self.align_corners
        )

        return seg_logits
