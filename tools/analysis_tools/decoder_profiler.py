import argparse
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor
from fvcore.nn import FlopCountAnalysis, parameter_count_table


def build_dummy_inputs(in_channels, image_size=256, batch_size=1, device='cuda'):
    """构造 decode_head 所需的 feature list"""
    feats = []
    H, W = image_size, image_size

    for i, c in enumerate(in_channels):
        h = H // (2 ** (i + 2))    # 1/4, 1/8, 1/16, 1/32
        w = W // (2 ** (i + 2))
        feats.append(torch.randn(batch_size, c, h, w).to(device))

    return feats


def profile_decode_head(cfg_path, image_size=512, device='cuda'):
    print("\n================ Decode Head Profiling ================\n")

    # Load config
    cfg = Config.fromfile(cfg_path)

    # Build model
    model = build_segmentor(cfg.model)
    model.eval()
    model.to(device)

    decode_head = model.decode_head
    decode_head.eval()

    # build dummy feature maps
    feats = build_dummy_inputs(decode_head.in_channels, image_size, 1, device)

    print("Running FLOPs analysis...")
    flops = FlopCountAnalysis(decode_head, feats)

    print("\n------ FLOPs ------")
    print(f"Total decode_head FLOPs: {flops.total() / 1e6:.3f} M")

    print("\n------ Params ------")
    print(parameter_count_table(decode_head))

    print("\n------ Per-module FLOPs Breakdown ------")
    try:
        for name, val in flops.by_module().items():
            print(f"{name:<50}: {val/1e6:.4f} M")
    except:
        print("⚠ 部分算子不支持子模块 FLOPs 分解")

    print("\n====================== Done ======================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Decode head profiler")
    parser.add_argument("config", help="config file path")
    parser.add_argument("--img", type=int, default=256, help="input image resolution")
    parser.add_argument("--device", default='cuda', help="cuda or cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile_decode_head(args.config, args.img, args.device)
