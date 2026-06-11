# PVSA v2.0 说明

## 当前目标

`pvsa-v2.0` 是投票稀疏注意力的稳妥实现分支。当前已经完成第一版可验证实现：

- 默认模型路径仍然不启用新后端。
- `use_topp_flash=False` 仍是默认值。
- 投票自注意力、投票融合网络和普通 `kv_gather` 路径继续保留。
- 显式设置 `use_topp_flash=True` 后，会走块式注意力后端。
- 块式后端支持前向和反向传播，可用于小尺寸对齐和训练可用性检查。

当前还不是最终手写 `CUDA` 内核。由于开发机没有 `nvcc`，本分支先实现可训练的块式后端和验证脚本，把数学路径、接口和误差检查固定下来。后续有编译环境后，可以把内部块计算替换成真正的 `CUDA` 核。

## 关键文件

```text
mmseg/models/utils/top_p_bra.py
mmseg/models/utils/topp_flash_kernel.py
tests/test_models/test_utils/test_topp_flash_kernel.py
tools/analysis_tools/check_topp_flash_attention.py
```

## 注意力路径

普通路径仍然是：

```text
qkv -> 窗口路由 -> kv_gather -> 分头注意力 -> lepe -> wo
```

新后端入口位于路由之后、`kv_gather` 之前：

```text
r_weight, r_idx, r_mask = self.router(...)
if use_topp_flash:
    topp_flash_attention(...)
else:
    原普通路径
```

块式后端会按一批窗口一批窗口读取路由到的 `kv`，只在当前小块里临时形成规则矩阵，然后做注意力计算。这样避免一次性生成完整的 `kv_gather` 大张量。

## 反向传播

当前反向传播采用重算方式：

```text
前向：按块计算输出，不保存完整注意力矩阵
反向：根据保存的 q、kv、路由权重、路由索引和路由掩码重新计算，再求梯度
```

这个方式更接近以后写手工核时的接口形态，也可以先验证训练链路是否可用。

## 对齐检查

运行独立注意力后端检查：

```bash
python tools/analysis_tools/check_topp_flash_attention.py --device cuda
```

运行完整 `ToppAttention` 小尺寸检查：

```bash
python tools/analysis_tools/check_topp_flash_attention.py --device cuda --full-module
```

运行单元测试：

```bash
pytest tests/test_models/test_utils/test_topp_flash_kernel.py
```

脚本会输出：

```text
前向最大绝对误差
前向最大相对误差
梯度最大绝对误差
梯度最大相对误差
普通路径耗时
块式路径耗时
普通路径显存峰值
块式路径显存峰值
```

## 配置方式

默认不启用：

```python
use_topp_flash=False
```

启用块式后端：

```python
use_topp_flash=True
topp_flash_block_windows=64
```

`VTFormer` 和 `Block` 已经把这两个参数传到 `ToppAttention`，因此后续可以直接在模型配置里打开。

`topp_flash_block_windows` 控制每次处理多少个窗口。数值越大，计算更接近一次性张量计算；数值越小，峰值显存更低，但循环开销更大。

## 回退方式

如果测试中出现问题，直接关闭：

```python
use_topp_flash=False
```

关闭后会回到原来的 `kv_gather` 路径，不影响原模型结构和数据集配置。

## 后续真核路线

后续真正写 `CUDA` 核时，建议只替换 `topp_flash_kernel.py` 内部块计算，不改 `ToppAttention` 主体：

1. 实现前向核，只输出注意力结果。
2. 对齐当前块式后端和普通路径。
3. 实现反向核。
4. 用脚本比较误差、速度和显存。
5. 稳定后再在配置中打开 `use_topp_flash=True`。
