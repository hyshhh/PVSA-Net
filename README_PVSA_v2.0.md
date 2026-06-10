# PVSA v2.0 说明

## 分支目标

`pvsa-v2.0` 是投票稀疏注意力的稳妥演进分支。这个分支先建立可选的闪存注意力入口，不直接改动现有可运行路径，也不默认加载任何自定义核。

当前默认行为仍然是原来的 `ToppAttention` 张量实现：

- `use_topp_flash=False` 是默认值。
- 默认不会编译、导入或调用自定义 `CUDA` 核。
- 投票自注意力、投票融合网络和普通 `kv_gather` 路径继续保留。
- 如果显式设置 `use_topp_flash=True`，但自定义核不可用，会自动回退到普通实现。

## 已保留结构

主入口仍在：

```text
mmseg/models/utils/top_p_bra.py
```

其中 `ToppAttention` 的普通流程没有删除：

```text
qkv -> 窗口路由 -> kv_gather -> 分头注意力 -> lepe -> wo
```

新增的可选入口位于路由之后、`kv_gather` 之前：

```text
r_weight, r_idx, r_mask = self.router(...)
if self.use_topp_flash and is_topp_flash_available():
    topp_flash_attention(...)
else:
    继续走原来的 kv_gather 路径
```

这样未来实现自定义核时，可以直接复用当前路由结果，不需要重写模型主体。

## 新增文件

```text
mmseg/models/utils/topp_flash_kernel.py
```

这个文件现在是安全占位模块：

- `is_topp_flash_available()` 固定返回 `False`。
- `topp_flash_attention(...)` 只提供未来接口，不会在默认路径中执行。
- 现阶段不会触发任何编译动作，也不会改变本机环境。

## 为什么不会改崩环境

这个分支没有做以下事情：

- 没有重装 `torch`。
- 没有替换系统 `CUDA`。
- 没有改动训练脚本的数据集路径。
- 没有删除普通注意力路径。
- 没有把自定义核设为默认路径。

因此当前模型仍然按原来的方式运行。自定义核只是预留入口，后续可以在独立文件里逐步补齐。

## 后续实现路线

建议按下面顺序继续做：

1. 先实现前向自定义核，只计算输出，不接训练。
2. 用小尺寸输入对齐普通 `ToppAttention` 的输出误差。
3. 再实现反向传播，验证训练可用。
4. 对比显存、速度和精度。
5. 确认稳定后，再考虑在配置中打开 `use_topp_flash=True`。

## 关于分块补零方案

当前分支没有启用全局统一补零。后续自定义核可以采用更稳的局部方案：

```text
每个查询块读取路由到的 kv 块
每个小 kv 块在核内部临时补齐
补齐位置在 softmax 前屏蔽
只输出真实位置参与的注意力结果
```

这样能保持图形处理器友好的规则小块计算，同时避免全局大矩阵补零带来的额外显存浪费。

## 回退方式

如需使用完全原始路径，保持默认配置即可：

```python
use_topp_flash=False
```

如果后续测试自定义核出现问题，只要关闭这个参数，就会回到当前普通实现。
