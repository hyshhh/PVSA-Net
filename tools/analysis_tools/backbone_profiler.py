import argparse
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import warnings

warnings.filterwarnings("ignore")


def build_backbone_input(in_channels, image_size=256, batch_size=1, device="cuda"):
    """构造 backbone 输入（只需要最初的 RGB 输入即可）"""
    B = batch_size
    C = in_channels
    H = W = image_size
    return torch.randn(B, C, H, W).to(device)


def profile_backbone(cfg_path, image_size=256, device="cuda"):
    print("\n================ Backbone Profiling ================\n")

    # Load config
    cfg = Config.fromfile(cfg_path)

    # Build model
    model = build_segmentor(cfg.model)
    model.eval()
    model.to(device)

    backbone = model.backbone
    backbone.eval()

    # 构造输入
    input_tensor = build_backbone_input(
        in_channels=3,
        image_size=image_size,
        device=device
    )

    print("Running FLOPs analysis...")
    flops = FlopCountAnalysis(backbone, input_tensor)

    print("\n------ FLOPs ------")
    print(f"Total backbone FLOPs: {flops.total() / 1e9:.4f} GFLOPs")

    print("\n------ Params ------")
    print(parameter_count_table(backbone))

    print("\n------ Per-module FLOPs Breakdown ------")
    try:
        for name, val in flops.by_module().items():
            print(f"{name:<50}: {val/1e6:.4f} M")
    except:
        print("⚠  fvcore 不支持某些注意力算子的子模块 FLOPs 分解")

    print("\n====================== Done ======================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Backbone profiler")
    parser.add_argument("config", help="config file path")
    parser.add_argument("--img", type=int, default=256, help="input image resolution")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile_backbone(args.config, args.img, args.device)
