# Copyright (c) OpenMMLab. All rights reserved.
"""Use the pytorch-grad-cam tool to visualize Class Activation Maps (CAM).

requirement: pip install grad-cam
"""

from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from mmengine.model import revert_sync_batchnorm
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules
import cv2


class SemanticSegmentationTarget:
    """Wrap the model for semantic segmentation CAM."""

    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        model_output = torch.unsqueeze(model_output, dim=0)
        # 将输出 resize 到 mask 尺寸，保证相乘不报错
        model_output = F.interpolate(
            model_output, size=self.mask.shape, mode='bilinear', align_corners=False)
        model_output = torch.squeeze(model_output, dim=0)
        return (model_output[self.category, :, :] * self.mask).sum()


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='prediction.png', help='Path to output prediction file')
    parser.add_argument('--cam-file', default='vis_cam.png', help='Path to output cam file')
    parser.add_argument('--target-layers', default='backbone.stages[3]', help='Target layers to visualize CAM')
    parser.add_argument('--category-index', default='7', help='Category to visualize CAM')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    register_all_modules()
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # test a single image
    result = inference_model(model, args.img)

    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)

    # result data conversion
    prediction_data = result.pred_sem_seg.data
    pre_np_data = prediction_data.cpu().numpy().squeeze(0)

    target_layers = [eval(f'model.{args.target_layers}')]
    category = int(args.category_index)

    # 将 mask resize 到输入裁剪尺寸 224x224
    input_size = 224
    mask_float = np.float32(pre_np_data == category)
    mask_float_resized = cv2.resize(mask_float, (input_size, input_size), interpolation=cv2.INTER_NEAREST)

    # 数据处理
    image = np.array(Image.open(args.img).convert('RGB'))
    rgb_img = np.float32(image) / 255
    rgb_img_resized = np.array(Image.fromarray((rgb_img * 255).astype(np.uint8)).resize((input_size, input_size)))
    rgb_img_resized = np.float32(rgb_img_resized) / 255

    config = Config.fromfile(args.config)
    image_mean = config.data_preprocessor['mean']
    image_std = config.data_preprocessor['std']
    input_tensor = preprocess_image(
        rgb_img_resized,
        mean=[x / 255 for x in image_mean],
        std=[x / 255 for x in image_std]
    )

    # Grad CAM(Class Activation Maps)
    targets = [SemanticSegmentationTarget(category, mask_float_resized)]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img_resized, grayscale_cam, use_rgb=True)
        Image.fromarray(cam_image).save(args.cam_file)


if __name__ == '__main__':
    main()
