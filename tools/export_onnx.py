import torch
import torch.nn as nn

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
from mmcv.cnn.utils import fuse_conv_bn
from mmcv.cnn import ConvModule
# 必须注册 mmseg 的所有模块（否则会报 SegDataPreProcessor 等错误）
register_all_modules()


# ===== ONNX wrapper（关键）=====
class MMSegONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # forward_dummy 是 mmseg 专门给部署用的
        return self.model.forward_dummy(x)


def main():
    # ================= 用户需要改的部分 =================

    config_file = './configs_l/segformer/segformer_mit-b0-gqy-256x256.py'
    checkpoint_file = './mmseg_log/segformer/yz/train_b2/best_mIoU_iter_140000.pth'
    onnx_file = './mmseg_log/onnx_file/segformer_b2.onnx'

    input_shape = (1, 3, 256, 256)   # ⭐ 固定 B C H W
    opset = 11
    # ====================================================

    # 1. build model
    cfg = Config.fromfile(config_file)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None  # 防止训练相关分支进入图

    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_file, map_location='cpu')
    model.eval()

    for m in model.modules():
        if isinstance(m, ConvModule):
            fuse_conv_bn(m)

    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = True

    # 2. wrap model
    wrapper = MMSegONNXWrapper(model)
    wrapper.eval()

    # 3. dummy input（静态 shape）
    dummy_input = torch.randn(input_shape, requires_grad=False)

    # 4. export ONNX（⚠️ 不要 dynamic_axes）
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_file,
        input_names=['input'],
        output_names=['logits'],
        opset_version=opset,
        do_constant_folding=True,
        verbose=False
    )

    print(f'✅ Static ONNX exported to: {onnx_file}')
    print(f'   Input shape : {input_shape}')
    print(f'   Output shape: (1, num_classes, 256, 256)')


if __name__ == '__main__':
    main()
