# Copyright (c) OpenMMLab. All rights reserved.
"""Optional block attention backend for PVSA ToppAttention.

The default backend in this file is a PyTorch block implementation with a
custom autograd wrapper. It is intended for correctness and training checks
before replacing the inner block with a hand-written CUDA kernel.
"""

import os
from typing import Optional, Tuple

import torch
from torch import Tensor


_TORCH_BACKENDS = {'torch', 'torch_block', 'block'}


def _normalize_backend(backend: Optional[str] = None) -> str:
    backend = backend or os.getenv('PVSA_TOPP_FLASH_BACKEND', 'torch_block')
    return backend.strip().lower()


def is_topp_flash_available(backend: Optional[str] = None) -> bool:
    """Return whether the requested optional backend is available."""
    backend = _normalize_backend(backend)
    if backend in _TORCH_BACKENDS:
        return True
    if backend == 'cuda':
        return False
    return False


def topp_attention_reference(q_pix: Tensor,
                             kv_pix: Tensor,
                             r_weight: Tensor,
                             r_idx: Tensor,
                             r_mask: Tensor,
                             num_heads: int,
                             qk_dim: int,
                             dim: int,
                             scale: float,
                             n_win: int,
                             H: int,
                             W: int) -> Tensor:
    """Reference implementation that matches the original kv_gather path."""
    _validate_inputs(q_pix, kv_pix, r_weight, r_idx, r_mask, num_heads,
                     qk_dim, dim, n_win, H, W)
    n, p2, kv_len, c_kv = kv_pix.shape
    topk = r_idx.size(-1)

    kv_pix_sel = torch.gather(
        kv_pix.view(n, 1, p2, kv_len, c_kv).expand(-1, p2, -1, -1, -1),
        dim=2,
        index=r_idx.long().view(n, p2, topk, 1, 1).expand(
            -1, -1, -1, kv_len, c_kv))
    kv_pix_sel = r_weight.view(n, p2, topk, 1, 1).to(kv_pix_sel.dtype) * kv_pix_sel
    k_pix_sel, v_pix_sel = kv_pix_sel.split([qk_dim, dim], dim=-1)

    head_q = qk_dim // num_heads
    head_v = dim // num_heads
    k_pix_sel = k_pix_sel.view(
        n, p2, topk, kv_len, num_heads, head_q).permute(
            0, 1, 4, 5, 2, 3).reshape(n * p2, num_heads, head_q,
                                      topk * kv_len)
    v_pix_sel = v_pix_sel.view(
        n, p2, topk, kv_len, num_heads, head_v).permute(
            0, 1, 4, 2, 3, 5).reshape(n * p2, num_heads, topk * kv_len,
                                      head_v)
    q = q_pix.view(n * p2, -1, num_heads, head_q).permute(0, 2, 1, 3)

    attn_weight = (q * scale) @ k_pix_sel
    route_mask = r_mask[..., None].expand(-1, -1, -1, kv_len)
    route_mask = route_mask.reshape(n * p2, 1, 1, topk * kv_len)
    attn_weight = attn_weight.masked_fill(
        ~route_mask, torch.finfo(attn_weight.dtype).min)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    out = attn_weight @ v_pix_sel
    out = out.permute(0, 2, 1, 3).reshape(n * p2, -1, dim)
    return _unflatten_windows(out, n, n_win, H, W, dim)


def topp_flash_attention(q_pix: Tensor,
                         kv_pix: Tensor,
                         r_weight: Tensor,
                         r_idx: Tensor,
                         r_mask: Tensor,
                         num_heads: int,
                         qk_dim: int,
                         dim: int,
                         scale: float,
                         n_win: int,
                         H: int,
                         W: int,
                         block_windows: int = 64,
                         backend: Optional[str] = None) -> Tensor:
    """Compute routed attention without materializing the full kv_gather tensor."""
    backend = _normalize_backend(backend)
    if backend not in _TORCH_BACKENDS:
        raise RuntimeError(
            f'topp flash backend {backend!r} is unavailable in this build.')
    return _ToppBlockAttentionFunction.apply(
        q_pix, kv_pix, r_weight, r_idx, r_mask, num_heads, qk_dim, dim,
        float(scale), n_win, H, W, int(block_windows))


class _ToppBlockAttentionFunction(torch.autograd.Function):
    """Autograd wrapper that recomputes the block attention in backward."""

    @staticmethod
    def forward(ctx, q_pix: Tensor, kv_pix: Tensor, r_weight: Tensor,
                r_idx: Tensor, r_mask: Tensor, num_heads: int, qk_dim: int,
                dim: int, scale: float, n_win: int, H: int, W: int,
                block_windows: int) -> Tensor:
        q_pix = q_pix.contiguous()
        kv_pix = kv_pix.contiguous()
        r_weight = r_weight.contiguous()
        r_idx = r_idx.contiguous()
        r_mask = r_mask.contiguous().bool()
        ctx.save_for_backward(q_pix, kv_pix, r_weight, r_idx, r_mask)
        ctx.params = (num_heads, qk_dim, dim, scale, n_win, H, W,
                      block_windows)
        with torch.no_grad():
            return _topp_attention_block_impl(
                q_pix, kv_pix, r_weight, r_idx, r_mask, num_heads, qk_dim,
                dim, scale, n_win, H, W, block_windows)

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tuple[Optional[Tensor], ...]:
        q_pix, kv_pix, r_weight, r_idx, r_mask = ctx.saved_tensors
        num_heads, qk_dim, dim, scale, n_win, H, W, block_windows = ctx.params

        needs = ctx.needs_input_grad
        with torch.enable_grad():
            q = q_pix.detach().requires_grad_(needs[0])
            kv = kv_pix.detach().requires_grad_(needs[1])
            rw = r_weight.detach().requires_grad_(needs[2])
            out = _topp_attention_block_impl(
                q, kv, rw, r_idx, r_mask, num_heads, qk_dim, dim, scale,
                n_win, H, W, block_windows)

        targets = []
        positions = []
        if needs[0]:
            targets.append(q)
            positions.append(0)
        if needs[1]:
            targets.append(kv)
            positions.append(1)
        if needs[2]:
            targets.append(rw)
            positions.append(2)

        grads = [None, None, None]
        if targets:
            computed = torch.autograd.grad(
                out, targets, grad_out.contiguous(), allow_unused=True)
            for pos, grad in zip(positions, computed):
                grads[pos] = grad

        return (grads[0], grads[1], grads[2], None, None, None, None, None,
                None, None, None, None, None)


def _topp_attention_block_impl(q_pix: Tensor,
                               kv_pix: Tensor,
                               r_weight: Tensor,
                               r_idx: Tensor,
                               r_mask: Tensor,
                               num_heads: int,
                               qk_dim: int,
                               dim: int,
                               scale: float,
                               n_win: int,
                               H: int,
                               W: int,
                               block_windows: int = 64) -> Tensor:
    _validate_inputs(q_pix, kv_pix, r_weight, r_idx, r_mask, num_heads,
                     qk_dim, dim, n_win, H, W)
    n, p2, q_len, _ = q_pix.shape
    _, _, kv_len, c_kv = kv_pix.shape
    topk = r_idx.size(-1)
    head_q = qk_dim // num_heads
    head_v = dim // num_heads
    flat_size = n * p2
    block_windows = flat_size if block_windows <= 0 else min(block_windows,
                                                             flat_size)

    q_flat = q_pix.reshape(flat_size, q_len, qk_dim)
    idx_flat = r_idx.reshape(flat_size, topk).long()
    weight_flat = r_weight.reshape(flat_size, topk).to(kv_pix.dtype)
    mask_flat = r_mask.reshape(flat_size, topk).bool()
    flat_out = []

    for start in range(0, flat_size, block_windows):
        end = min(start + block_windows, flat_size)
        batch = end - start
        flat_ids = torch.arange(start, end, device=q_pix.device)
        n_ids = torch.div(flat_ids, p2, rounding_mode='floor')

        kv_batch = kv_pix.index_select(0, n_ids)
        idx = idx_flat[start:end]
        kv_sel = torch.gather(
            kv_batch,
            dim=1,
            index=idx.view(batch, topk, 1, 1).expand(
                -1, -1, kv_len, c_kv))
        kv_sel = weight_flat[start:end].view(
            batch, topk, 1, 1) * kv_sel
        k_sel, v_sel = kv_sel.split([qk_dim, dim], dim=-1)

        k_sel = k_sel.view(batch, topk, kv_len, num_heads,
                           head_q).permute(0, 3, 4, 1, 2).reshape(
                               batch, num_heads, head_q, topk * kv_len)
        v_sel = v_sel.view(batch, topk, kv_len, num_heads,
                           head_v).permute(0, 3, 1, 2, 4).reshape(
                               batch, num_heads, topk * kv_len, head_v)
        q = q_flat[start:end].view(batch, q_len, num_heads,
                                   head_q).permute(0, 2, 1, 3)

        scores = (q * scale) @ k_sel
        route_mask = mask_flat[start:end, :, None].expand(
            -1, -1, kv_len).reshape(batch, 1, 1, topk * kv_len)
        scores = scores.masked_fill(~route_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v_sel
        flat_out.append(out.permute(0, 2, 1, 3).reshape(batch, q_len, dim))

    flat_out = torch.cat(flat_out, dim=0)
    return _unflatten_windows(flat_out, n, n_win, H, W, dim)


def _unflatten_windows(flat_out: Tensor, n: int, n_win: int, H: int, W: int,
                       dim: int) -> Tensor:
    q_h = H // n_win
    q_w = W // n_win
    return flat_out.view(n, n_win, n_win, q_h, q_w, dim).permute(
        0, 1, 3, 2, 4, 5).reshape(n, H, W, dim).contiguous()


def _validate_inputs(q_pix: Tensor, kv_pix: Tensor, r_weight: Tensor,
                     r_idx: Tensor, r_mask: Tensor, num_heads: int,
                     qk_dim: int, dim: int, n_win: int, H: int, W: int) -> None:
    if q_pix.dim() != 4 or kv_pix.dim() != 4:
        raise ValueError('q_pix and kv_pix must be 4D tensors.')
    n, p2, q_len, qk = q_pix.shape
    n_kv, p2_kv, _, c_kv = kv_pix.shape
    if (n_kv, p2_kv) != (n, p2):
        raise ValueError('q_pix and kv_pix must share n and p2 dimensions.')
    if r_idx.shape != r_weight.shape or r_idx.shape != r_mask.shape:
        raise ValueError('r_weight, r_idx and r_mask must share one shape.')
    if r_idx.shape[:2] != (n, p2):
        raise ValueError('routing tensors must match q_pix n and p2.')
    if p2 != n_win * n_win:
        raise ValueError('p2 must equal n_win * n_win.')
    if H % n_win != 0 or W % n_win != 0:
        raise ValueError('H and W must be divisible by n_win.')
    if q_len != (H // n_win) * (W // n_win):
        raise ValueError('q_len must match the window area.')
    if qk != qk_dim or c_kv != qk_dim + dim:
        raise ValueError('channel dimensions do not match qk_dim and dim.')
    if qk_dim % num_heads != 0 or dim % num_heads != 0:
        raise ValueError('qk_dim and dim must be divisible by num_heads.')
