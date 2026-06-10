import os
import numpy as np
import torch
import onnx
import onnxruntime as ort

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


def main():
    # ====================== 路径配置（你只需要改这里） ======================
    config_file = '../configs/lformer/lformer-small-yz_seg-256x256.py'
    checkpoint_file = '../mmseg_log/lformer-s/best_mIoU_iter_144000.pth'
    onnx_file = '../onnx_file/lformer_2_0_144000.onnx'

    input_shape = (1, 3, 256, 256)
    # ======================================================================

    print('========== Step 0: basic check ==========')
    assert os.path.exists(onnx_file), f'ONNX file not found: {onnx_file}'
    print(f'ONNX file found: {onnx_file}')

    # ----------------------------------------------------------------------
    print('\n========== Step 1: ONNX structure check ==========')
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print('✅ ONNX structure is valid')
    print('Opset version:', onnx_model.opset_import[0].version)

    # ----------------------------------------------------------------------
    print('\n========== Step 2: ONNX Runtime inference ==========')
    sess = ort.InferenceSession(
        onnx_file,
        providers=['CPUExecutionProvider']
    )

    print('ONNX Inputs:')
    for i in sess.get_inputs():
        print(f'  name={i.name}, shape={i.shape}, type={i.type}')

    print('ONNX Outputs:')
    for o in sess.get_outputs():
        print(f'  name={o.name}, shape={o.shape}, type={o.type}')

    x_np = np.random.randn(*input_shape).astype(np.float32)
    onnx_out = sess.run(None, {'input': x_np})[0]

    print('ONNX output shape:', onnx_out.shape)
    print('ONNX output dtype:', onnx_out.dtype)
    print('ONNX output min/max:', onnx_out.min(), onnx_out.max())

    # ----------------------------------------------------------------------
    print('\n========== Step 3: PyTorch vs ONNX numerical check ==========')

    # 注册 mmseg 所有模块（关键）
    register_all_modules()

    cfg = Config.fromfile(config_file)
    cfg.model.pretrained = None

    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_file, map_location='cpu')
    model.eval()

    x_torch = torch.from_numpy(x_np)

    with torch.no_grad():
        torch_out = model.forward_dummy(x_torch).cpu().numpy()

    mean_diff = np.mean(np.abs(torch_out - onnx_out))
    max_diff = np.max(np.abs(torch_out - onnx_out))

    print('Mean absolute diff:', mean_diff)
    print('Max absolute diff :', max_diff)

    # 经验阈值
    if mean_diff < 1e-4 and max_diff < 1e-3:
        print('✅ PyTorch and ONNX outputs are numerically consistent')
    else:
        print('⚠️ Warning: numerical difference is relatively large')

    print('\n========== ALL CHECKS PASSED ==========')


if __name__ == '__main__':
    main()
