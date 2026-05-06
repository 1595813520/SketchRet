from __future__ import annotations

from typing import Dict, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def open_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def resize_long_edge_and_pad_square(
    img: Image.Image,
    target_size: int,
    fill=(255, 255, 255),
    resample=Image.BILINEAR,
) -> Tuple[Image.Image, Dict[str, object]]:
    img = img.convert('RGB')
    orig_w, orig_h = img.size
    long_edge = max(orig_w, orig_h)
    scale = float(target_size) / float(long_edge)
    scaled_w = max(1, int(round(orig_w * scale)))
    scaled_h = max(1, int(round(orig_h * scale)))
    resized = img.resize((scaled_w, scaled_h), resample=resample)
    canvas = Image.new('RGB', (target_size, target_size), fill)
    pad_x = (target_size - scaled_w) // 2
    pad_y = (target_size - scaled_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    meta = {
        'orig_size': (orig_w, orig_h),
        'scaled_size': (scaled_w, scaled_h),
        'scale_ratio': scale,
        'pad_offset': (pad_x, pad_y),
        'target_size': target_size,
    }
    return canvas, meta


def make_valid_mask(meta: Dict[str, object]) -> torch.Tensor:
    target_size = int(meta['target_size'])
    scaled_w, scaled_h = meta['scaled_size']
    pad_x, pad_y = meta['pad_offset']
    mask = torch.zeros(1, target_size, target_size, dtype=torch.float32)
    mask[:, pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w] = 1.0
    return mask


def preprocess_line_target(path: str, resolution: int) -> Tuple[torch.Tensor, Dict[str, object]]:
    img = open_image_rgb(path)
    padded, meta = resize_long_edge_and_pad_square(img, resolution, resample=Image.BILINEAR)
    x = TF.to_tensor(padded)  # [0,1]
    x = x * 2.0 - 1.0
    return x, meta


def preprocess_sketch_image(path: str, resolution: int) -> Tuple[torch.Tensor, Dict[str, object], torch.Tensor]:
    img = open_image_rgb(path)
    padded, meta = resize_long_edge_and_pad_square(img, resolution, resample=Image.BILINEAR)
    padded = padded.convert('L')
    x = TF.to_tensor(padded)  # [0,1]
    valid_mask = make_valid_mask(meta)
    return x, meta, valid_mask


def preprocess_ref_image(path: str, resolution: int = 224) -> torch.Tensor:
    img = open_image_rgb(path)
    padded, _ = resize_long_edge_and_pad_square(img, resolution, resample=Image.BICUBIC)
    x = TF.to_tensor(padded)
    mean = torch.tensor(CLIP_IMAGE_MEAN, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD, dtype=x.dtype).view(3, 1, 1)
    return (x - mean) / std


def denorm_line_tensor(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def apply_valid_mask_white(img_batch: torch.Tensor, valid_mask_batch: torch.Tensor) -> torch.Tensor:
    if img_batch.shape[1] == 1:
        valid_mask = valid_mask_batch
    else:
        valid_mask = valid_mask_batch.repeat(1, img_batch.shape[1], 1, 1)
    return img_batch * valid_mask + (1.0 - valid_mask)


def postprocess_pred_to_pil(pred: torch.Tensor, meta: Dict[str, object], unpad_back: bool = True):
    pred = pred[0].cpu()
    img = TF.to_pil_image(pred)
    if not unpad_back:
        return img
    scaled_w, scaled_h = meta['scaled_size']
    pad_x, pad_y = meta['pad_offset']
    orig_w, orig_h = meta['orig_size']
    cropped = img.crop((pad_x, pad_y, pad_x + scaled_w, pad_y + scaled_h))
    return cropped.resize((orig_w, orig_h), resample=Image.BILINEAR)
