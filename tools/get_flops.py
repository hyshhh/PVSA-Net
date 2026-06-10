#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from mmengine.config import Config
from mmengine.analysis import get_model_complexity_info
from mmseg.models import build_segmentor
# 确保注册
from mmseg.datasets import SegDataPreProcessor

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute FLOPs and params for a MMSegmentation model')
    parser.add_argument('config', help='path to config file')
    parser.add_argument(
        '--shape', type=int, nargs=2, default=[512, 512],
        help='input image size in H W, default is 512 512')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    model = build_segmentor(cfg.model)
    model.eval()
    # 临时屏蔽数据预处理避免 registry 报错
    model.data_preprocessor = None

    input_shape = (3, args.shape[0], args.shape[1])
    flops, params = get_model_complexity_info(model, input_shape)

    print(f'Model: {args.config}')
    print(f'Input shape: {input_shape}')
    print(f'FLOPs: {flops}')
    print(f'Params: {params}')

if __name__ == '__main__':
    main()
