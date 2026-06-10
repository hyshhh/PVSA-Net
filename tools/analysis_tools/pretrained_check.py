import torch
# from mmseg.models import build_model
# from mmcv import Config

from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.apis import MMSegInferencer
import mmseg.models  
from mmengine.registry import init_default_scope

# 加载配置文件
cfg = Config.fromfile('./configs/lformer/lformer-b-GBA-256x256.py')  # 替换为你的配置文件路径

init_default_scope('mmseg')

# 构建模型
model = build_segmentor(cfg.model)

# 打印每个可训练层的名字和参数的形状
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name}, Shape: {param.shape}")

inferencer = MMSegInferencer(cfg)

inferencer('data/gqyyz/image/test/00261.jpg', show=True)                         
# 如果你想把参数保存到文件中，也可以使用以下方式
# with open('model_params.txt', 'w') as f:
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             f.write(f"Layer: {name}, Shape: {param.shape}\n")

print("模型参数打印完毕。")