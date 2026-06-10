# mmseg/engine/hooks/window_vis_hook.py
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmengine.hooks import Hook
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# -------------------------
# 辅助函数
# -------------------------
def to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()

def normalize_for_display(x: np.ndarray):
    x = x.astype(np.float32)
    x = x - x.min()
    if x.max() > 0:
        x = x / (x.max() + 1e-8)
    return x

def feature_map_to_image(feat: torch.Tensor, size=None):
    """
    feat: Tensor of shape [B, C, H, W] or [B, H, W, C] or [H, W]
    返回 numpy H x W (grayscale) 或 H x W x 3 (rgb)
    """
    if isinstance(feat, torch.Tensor):
        t = feat.detach().cpu()
        if t.dim() == 4:  # BCHW or BHWC
            if t.shape[1] <= 4:  # assume BCHW
                img = t[0].mean(0).numpy()
            else:
                img = t[0].mean(0).numpy()
        elif t.dim() == 3:
            # B H W or H W C
            if t.shape[0] <= 4:  # assume B H W (channels small) -> mean channels
                img = t[0].numpy()
            else:
                img = t.numpy().mean(-1) if t.shape[-1] > 4 else t[0].numpy()
        elif t.dim() == 2:
            img = t.numpy()
        else:
            img = t.squeeze().numpy()
    else:
        img = np.array(feat)

    if size is not None:
        img = Image.fromarray(normalize_for_display(img) * 255.0).convert("L")
        img = img.resize(size, Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
    else:
        img = normalize_for_display(img)
    return img

def save_gray(img_np, path):
    img = Image.fromarray((img_np * 255).astype(np.uint8)).convert("L")
    img.save(path)

def draw_window_grid_on_image(rgb_img, win_h, win_w, offset=(0,0), color=(255,0,0), line_width=2):
    """
    rgb_img: PIL.Image (RGB)
    win_h, win_w: pixel size of each window
    offset: (top_offset, left_offset) for padding
    """
    draw = ImageDraw.Draw(rgb_img)
    W, H = rgb_img.size  # Pillow returns (width, height)
    top_off, left_off = offset
    # draw vertical lines
    x = left_off
    while x < W:
        draw.line([(x, 0), (x, H)], fill=color, width=line_width)
        x += win_w
    # draw horizontal lines
    y = top_off
    while y < H:
        draw.line([(0, y), (W, y)], fill=color, width=line_width)
        y += win_h
    return rgb_img

def reconstruct_from_windows(windows: torch.Tensor, n_h, n_w, ph, pw, layout='nphwc'):
    """
    将 windows 重构为 (B, H, W, C) 或 (B, C, H, W)
    支持常见 window 格式输入：
    - (B, n_h*n_w, ph, pw, C)  -> NHWC
    - (B*n_w*n_h, C, ph, pw)  -> BCHW (where first dim is B*windows)
    - (B, n_h, n_w, ph, pw, C)
    返回 numpy 图像： (B, H, W) single-channel normalized
    """
    t = windows
    if t.dim() == 5:
        # (B, nwin, ph, pw, C)
        B, nwin, ph_, pw_, C = t.shape
        if nwin != n_h * n_w:
            # try (B*nwin, C, ph, pw)
            pass
        # reshape to (B, n_h, n_w, ph, pw, C)
        t = t.reshape(B, n_h, n_w, ph_, pw_, C)
        # move to (B, ph*n_h, pw*n_w, C)
        t = t.permute(0,1,3,2,4,5).reshape(B, ph_*n_h, pw_*n_w, C)
        # mean across channels
        img = t.mean(-1).cpu().numpy()
        return img  # (B, H, W)
    elif t.dim() == 4:
        # common case: (B*nwin, C, ph, pw)
        Bn, C, ph_, pw_ = t.shape
        # assume B=1 if ambiguous
        B = 1
        nwin = Bn // B
        if nwin != n_h*n_w:
            # try to infer n_h, n_w from provided args
            pass
        t = t.reshape(B, n_h, n_w, C, ph_, pw_)
        # reorder to (B, ph*n_h, pw*n_w, C)
        t = t.permute(0,1,4,2,5,3).reshape(B, ph_*n_h, pw_*n_w, C)
        img = t.mean(-1).cpu().numpy()
        return img
    else:
        # fallback
        return None

# -------------------------
# WindowVisHook
# -------------------------
class WindowVisHook(Hook):
    """
    Hook to visualize window partition / merge and attention.
    使用方法：在 config 的 custom_hooks 中添加
        dict(type='WindowVisHook', save_dir='work_dirs/vis_windows', every_n_iters=1)
    """
    def __init__(self, save_dir='work_dirs/vis_windows', every_n_iters=1, max_images=4):
        self.save_dir = save_dir
        self.every_n_iters = every_n_iters
        self.max_images = max_images
        self.handles = []
        self.cache = {}  # name -> latest tensor
        os.makedirs(self.save_dir, exist_ok=True)

    def before_run(self, runner):
        model = runner.model
        # 注册 hook：匹配常见名字（根据你的模型改名）
        for name, module in model.named_modules():
            lname = name.lower()
            if 'varyingwindow' in lname or 'varywindow' in lname or 'windowpartition' in lname or 'window' in lname and 'attention' in lname:
                # 仅叶子节点注册
                try:
                    h = module.register_forward_hook(self._make_hook(name))
                    self.handles.append(h)
                except Exception:
                    pass

    def _make_hook(self, name):
        def fn(module, inp, out):
            # 保存张量（避免保存非常大对象，多保留一份小型化数据）
            # 我们把 inp/out 转为 cpu tensor（注意：不在训练时使用会慢）
            try:
                # 优先保存 output
                val = out
                if isinstance(val, (list, tuple)):
                    val = val[0]
                # copy to cpu (no grad)
                self.cache[name] = val.detach().cpu().clone()
            except Exception as e:
                # fallback to input
                try:
                    val = inp[0]
                    self.cache[name] = val.detach().cpu().clone()
                except Exception:
                    self.cache[name] = None
        return fn

    def after_val_iter(self, runner, batch_idx, data_batch, outputs):
        # 每若干次写出
        iter_idx = runner.iter
        if iter_idx % self.every_n_iters != 0:
            return

        # 尝试从 data_batch/outputs 找到原始图片 (B,3,H,W)
        inputs = None
        try:
            if isinstance(data_batch, dict):
                if 'inputs' in data_batch:
                    inputs = data_batch['inputs']
                else:
                    # mmseg dataloader may wrap differently
                    for v in data_batch.values():
                        if torch.is_tensor(v) and v.dim() == 4 and v.shape[1] == 3:
                            inputs = v
                            break
        except Exception:
            inputs = None

        # 选择要保存的样本数量
        B = 1
        if inputs is not None:
            B = inputs.shape[0]

        # 保存每个捕获模块的可视化
        for name, tensor in self.cache.items():
            if tensor is None:
                continue
            try:
                # 我们仅处理前 self.max_images 个样本
                # tensor 可能是 CPU tensor
                t = tensor
                # case 1: (B, nwin, ph, pw, C)
                if t.dim() == 5:
                    B0, nwin, ph, pw, C = t.shape
                    # infer grid layout n_h, n_w from name if possible; fallback to sqrt
                    n_h = n_w = int(math.sqrt(nwin))
                    if n_h * n_w != nwin:
                        # try to check model image size from inputs
                        n_h = n_w = nwin  # fallback
                    imgs = reconstruct_from_windows(t, n_h=n_h, n_w=n_w, ph=ph, pw=pw)
                    # imgs: (B, H, W)
                    for b in range(min(B0, self.max_images)):
                        img = imgs[b]
                        png_name = os.path.join(self.save_dir, f"{runner.mode}_iter{runner.iter}_{name}_recon_b{b}.png")
                        save_gray(img, png_name)
                # case 2: (B*nwin, C, ph, pw)
                elif t.dim() == 4 and t.shape[1] > 1:
                    # try reconstruct
                    Bn, C, ph, pw = t.shape
                    # try B from inputs
                    B0 = inputs.shape[0] if inputs is not None else 1
                    nwin = Bn // B0
                    n_h = n_w = int(math.sqrt(nwin))
                    imgs = reconstruct_from_windows(t, n_h=n_h, n_w=n_w, ph=ph, pw=pw)
                    if imgs is not None:
                        for b in range(min(B0, self.max_images)):
                            img = imgs[b]
                            png_name = os.path.join(self.save_dir, f"{runner.mode}_iter{runner.iter}_{name}_recon_b{b}.png")
                            save_gray(img, png_name)
                # case: attention map (e.g., tensor dim 4 (B, heads, Lq, Lk))
                elif t.dim() == 4 and (t.shape[1] <= 16 and t.shape[2] == t.shape[3]):
                    # average heads
                    attn = t.mean(1)  # (B, L, L)
                    for b in range(min(attn.shape[0], self.max_images)):
                        a = attn[b]
                        # visualize a small central query-attn map (reshape L -> sqrt(L) x sqrt(L) for readability)
                        L = a.shape[-1]
                        s = int(math.sqrt(L))
                        if s*s == L:
                            # show the attn of the central token
                            qidx = L // 2
                            att = a[qidx].reshape(s, s).numpy()
                            att = normalize_for_display(att)
                            save_gray(att, os.path.join(self.save_dir, f"{runner.mode}_iter{runner.iter}_{name}_attn_b{b}.png"))
                        else:
                            # fallback: save matrix as image
                            save_gray(normalize_for_display(a.numpy()), os.path.join(self.save_dir, f"{runner.mode}_iter{runner.iter}_{name}_attn_b{b}.png"))
                else:
                    # generic: mean over channels and save
                    # convert to BCHW if NHWC
                    tt = t
                    if tt.dim() == 3:  # HWC
                        arr = tt.numpy().mean(-1)
                        save_gray(normalize_for_display(arr), os.path.join(self.save_dir, f"{runner.mode}_iter{runner.iter}_{name}_generic.png"))
                    elif tt.dim() == 2:
                        save_gray(normalize_for_display(tt.numpy()), os.path.join(self.save_dir, f"{runner.mode}_iter{runner.iter}_{name}_generic.png"))
                    elif tt.dim() == 4:  # B C H W or B H W C
                        if tt.shape[1] == 3:
                            # likely BCHW RGB -> save original resized
                            for b in range(min(tt.shape[0], self.max_images)):
                                img = tt[b].permute(1,2,0).numpy()
                                img = normalize_for_display(img)
                                im = Image.fromarray((img*255).astype(np.uint8)).convert("RGB")
                                im.save(os.path.join(self.save_dir, f"{runner.mode}_iter{runner.iter}_{name}_rgb_b{b}.png"))
                        else:
                            # mean channels
                            for b in range(min(tt.shape[0], self.max_images)):
                                img = tt[b].mean(0).numpy()
                                save_gray(normalize_for_display(img), os.path.join(self.save_dir, f"{runner.mode}_iter{runner.iter}_{name}_mean_b{b}.png"))
            except Exception as e:
                # print but do not crash
                print("WindowVisHook save failed for", name, "error:", e)

        # optionally draw overlay grid on original inputs
        if inputs is not None:
            for b in range(min(inputs.shape[0], self.max_images)):
                inp = inputs[b]  # (C,H,W)
                inp_img = inp.detach().cpu().permute(1,2,0).numpy()
                inp_img = normalize_for_display(inp_img)
                im = Image.fromarray((inp_img*255).astype(np.uint8)).convert("RGB")
                # guess window size from cache if possible (take any window size)
                for k, v in self.cache.items():
                    if v is None:
                        continue
                    if v.dim() == 5:
                        _, nwin, ph, pw, _ = v.shape
                        n_h = int(math.sqrt(nwin))
                        n_w = n_h
                        H, W = im.size[1], im.size[0]
                        # compute pixel window size roughly
                        win_h = ph * (H // (n_h * ph)) if (H % (n_h*ph) == 0) else H // n_h
                        win_w = pw * (W // (n_w * pw)) if (W % (n_w*pw) == 0) else W // n_w
                        im_grid = draw_window_grid_on_image(im.copy(), win_h, win_w)
                        im_grid.save(os.path.join(self.save_dir, f"{runner.mode}_iter{runner.iter}_overlay_{k}_b{b}.png"))
                        break

        # clear cache to avoid memory growth
        self.cache = {}

    def after_run(self, runner):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
