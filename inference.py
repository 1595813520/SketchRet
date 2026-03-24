#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference script for panel-level sketch-to-line generation.

Input:
- required: sketch image
- optional: caption
- optional: refs + bbox_norm list

Output:
- line art image

Expected local modules:
- models/sketch_encoder.py
- models/attention_processor.py
- models/resampler.py
"""

from __future__ import annotations

import os
import json
import math
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from peft import LoraConfig
from peft.utils import set_peft_model_state_dict

from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.utils import convert_unet_state_dict_to_peft

from models.sketch_encoder import SketchEncoder
from models.attention_processor import MaskedRefAttentionProcessor
from models.resampler import Resampler


# =========================================================
# Helper modules
# =========================================================

class FrozenCLIPRefAdapter(nn.Module):
    """
    Frozen CLIP vision encoder + trainable projector + trainable resampler.
    Only projector/resampler weights are loaded from checkpoint.
    """
    def __init__(
        self,
        model_name_or_path: str,
        cross_attn_dim: int,
        num_queries: int = 8,
        resampler_dim: int = 1024,
        resampler_depth: int = 4,
        resampler_heads: int = 16,
        resampler_dim_head: int = 64,
    ):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(model_name_or_path)
        self.vision_encoder.requires_grad_(False)

        vision_hidden = self.vision_encoder.config.hidden_size
        self.proj = nn.Linear(vision_hidden, cross_attn_dim)
        self.resampler = Resampler(
            dim=resampler_dim,
            depth=resampler_depth,
            dim_head=resampler_dim_head,
            heads=resampler_heads,
            num_queries=num_queries,
            embedding_dim=cross_attn_dim,
            output_dim=cross_attn_dim,
            apply_pos_emb=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values=pixel_values, output_hidden_states=False)
            seq = outputs.last_hidden_state
        seq = self.proj(seq)
        return self.resampler(seq)

    def load_trainable_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]):
        self.proj.load_state_dict(state["proj"], strict=True)
        self.resampler.load_state_dict(state["resampler"], strict=True)


class SketchSpatialInjector:
    """
    Runtime-only hook injector.
    Does NOT modify persistent UNet structure.
    """
    def __init__(self, unet: UNet2DConditionModel):
        self.unet = unet
        self.cached_mid: Optional[torch.Tensor] = None
        self.cached_high: Optional[torch.Tensor] = None
        self.handles = []

    def set_features(self, mid_feat: torch.Tensor, high_feat: torch.Tensor):
        self.cached_mid = mid_feat
        self.cached_high = high_feat

    def clear(self):
        self.cached_mid = None
        self.cached_high = None

    def register(self):
        def hook_mid(module, inputs, output):
            if self.cached_mid is None:
                return output
            if not isinstance(output, torch.Tensor):
                return output
            if output.shape != self.cached_mid.shape:
                raise ValueError(
                    f"Mid feature shape mismatch: output={tuple(output.shape)} vs sketch_mid={tuple(self.cached_mid.shape)}"
                )
            return output + self.cached_mid

        def hook_high(module, inputs, output):
            if self.cached_high is None:
                return output
            if not isinstance(output, torch.Tensor):
                return output
            if output.shape != self.cached_high.shape:
                raise ValueError(
                    f"High feature shape mismatch: output={tuple(output.shape)} vs sketch_high={tuple(self.cached_high.shape)}"
                )
            return output + self.cached_high

        # 与训练时保持一致
        self.handles.append(self.unet.up_blocks[2].register_forward_hook(hook_mid))
        self.handles.append(self.unet.up_blocks[3].register_forward_hook(hook_high))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# =========================================================
# Image preprocess / postprocess
# =========================================================

def resize_long_edge_and_pad_square(
    img: Image.Image,
    target_size: int,
    fill=(255, 255, 255),
    resample=Image.BILINEAR,
):
    """
    保持比例，长边缩放到 target_size，再 pad 到正方形
    """
    img = img.convert("RGB")
    orig_w, orig_h = img.size
    long_edge = max(orig_w, orig_h)
    scale = float(target_size) / float(long_edge)

    scaled_w = max(1, int(round(orig_w * scale)))
    scaled_h = max(1, int(round(orig_h * scale)))

    resized = img.resize((scaled_w, scaled_h), resample=resample)

    canvas = Image.new("RGB", (target_size, target_size), fill)
    pad_x = (target_size - scaled_w) // 2
    pad_y = (target_size - scaled_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    meta = {
        "orig_size": (orig_w, orig_h),
        "scaled_size": (scaled_w, scaled_h),
        "scale_ratio": scale,
        "pad_offset": (pad_x, pad_y),
        "target_size": target_size,
    }
    return canvas, meta


def preprocess_sketch(sketch_path: str) -> Tuple[torch.Tensor, Dict]:
    img = Image.open(sketch_path).convert("RGB")
    padded, meta = resize_long_edge_and_pad_square(
        img,
        target_size=384,
        fill=(255, 255, 255),
        resample=Image.BILINEAR,
    )
    padded = padded.convert("L")
    x = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(padded.tobytes()))
                          .view(padded.size[1], padded.size[0], 1)
                          .numpy().copy())).float()  # fallback-safe
    # simpler PIL->tensor
    import torchvision.transforms.functional as TF
    x = TF.to_tensor(padded)  # [1,384,384], [0,1]
    return x.unsqueeze(0), meta


def preprocess_ref(ref_path: str) -> torch.Tensor:
    img = Image.open(ref_path).convert("RGB")
    padded, _ = resize_long_edge_and_pad_square(
        img,
        target_size=224,
        fill=(255, 255, 255),
        resample=Image.BICUBIC,
    )
    import torchvision.transforms.functional as TF
    x = TF.to_tensor(padded)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    x = (x - mean) / std
    return x


def denorm_line_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    [-1,1] -> [0,1]
    """
    return ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def postprocess_pred_to_pil(pred: torch.Tensor, meta: Dict, unpad_back: bool = True) -> Image.Image:
    """
    pred: [1,3,384,384], in [0,1]
    """
    import torchvision.transforms.functional as TF
    pred = pred[0].cpu()
    img = TF.to_pil_image(pred)

    if not unpad_back:
        return img

    scaled_w, scaled_h = meta["scaled_size"]
    pad_x, pad_y = meta["pad_offset"]
    orig_w, orig_h = meta["orig_size"]

    cropped = img.crop((pad_x, pad_y, pad_x + scaled_w, pad_y + scaled_h))
    restored = cropped.resize((orig_w, orig_h), resample=Image.BILINEAR)
    return restored


# =========================================================
# Model loading
# =========================================================

def get_unet_attention_hidden_size(unet: UNet2DConditionModel, name: str) -> int:
    if name.startswith("mid_block"):
        return unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name.split(".")[1])
        return list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name.split(".")[1])
        return unet.config.block_out_channels[block_id]
    else:
        return unet.config.cross_attention_dim


def load_models(
    pretrained_model_name_or_path: str,
    checkpoint_dir: str,
    lora_rank: int,
    ref_image_encoder_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
):
    # base
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # read extra_modules
    extra_path = os.path.join(checkpoint_dir, "extra_modules.pt")
    if not os.path.exists(extra_path):
        raise FileNotFoundError(f"extra_modules.pt not found in {checkpoint_dir}")
    extra = torch.load(extra_path, map_location="cpu")
    train_stage = extra.get("train_stage", "main")

    # LoRA
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg)

    # if ref stage, replace cross-attn processors first
    if train_stage == "ref":
        attn_procs = {}
        for name, old_proc in unet.attn_processors.items():
            if name.endswith("attn2.processor"):
                hidden_size = get_unet_attention_hidden_size(unet, name)
                attn_procs[name] = MaskedRefAttentionProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=unet.config.cross_attention_dim,
                    ref_cross_attention_dim=unet.config.cross_attention_dim,
                )
            else:
                attn_procs[name] = old_proc
        unet.set_attn_processor(attn_procs)

    # load LoRA
    lora_state_dict, _ = StableDiffusionPipeline.lora_state_dict(checkpoint_dir)
    unet_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items() if k.startswith("unet.")}
    unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
    set_peft_model_state_dict(unet, unet_state_dict, adapter_name="default")

    # sketch encoder
    sketch_encoder = SketchEncoder(
        pretrained_backbone=True,
        freeze_backbone=False,
        cross_attn_dim=unet.config.cross_attention_dim,
        num_sem_queries=8,
    )
    sketch_encoder.load_state_dict(extra["sketch_encoder"], strict=True)

    # ref adapter
    ref_adapter = None
    if train_stage == "ref":
        ref_adapter = FrozenCLIPRefAdapter(
            model_name_or_path=ref_image_encoder_name_or_path,
            cross_attn_dim=unet.config.cross_attention_dim,
            num_queries=8,
        )
        if "ref_adapter_trainable" in extra:
            ref_adapter.load_trainable_state_dict(extra["ref_adapter_trainable"])

        # load custom attn processors
        if "attn_processors" in extra:
            for name, sd in extra["attn_processors"].items():
                if name in unet.attn_processors and isinstance(unet.attn_processors[name], nn.Module):
                    unet.attn_processors[name].load_state_dict(sd, strict=True)

    # move
    text_encoder.to(device=device, dtype=dtype).eval()
    vae.to(device=device, dtype=dtype).eval()
    unet.to(device=device, dtype=dtype).eval()
    sketch_encoder.to(device=device, dtype=dtype).eval()
    if ref_adapter is not None:
        ref_adapter.to(device=device, dtype=dtype).eval()

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "scheduler": scheduler,
        "sketch_encoder": sketch_encoder,
        "ref_adapter": ref_adapter,
        "train_stage": train_stage,
    }


# =========================================================
# Inference
# =========================================================

@torch.no_grad()
def infer_one(
    models: Dict,
    sketch_path: str,
    caption: str = "",
    ref_items: Optional[List[Dict]] = None,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
    device: str = "cuda",
):
    device = torch.device(device)
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    unet = models["unet"]
    scheduler = models["scheduler"]
    sketch_encoder = models["sketch_encoder"]
    ref_adapter = models["ref_adapter"]
    train_stage = models["train_stage"]

    # preprocess sketch
    sketch_tensor, sketch_meta = preprocess_sketch(sketch_path)
    sketch_tensor = sketch_tensor.to(device=device, dtype=next(unet.parameters()).dtype)

    # text
    caption = caption if caption is not None else ""
    input_ids = tokenizer(
        [caption],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)

    text_tokens = text_encoder(input_ids, return_dict=False)[0]

    # sketch encoder
    latent_h = 48
    latent_w = 48
    sketch_out = sketch_encoder(
        sketch=sketch_tensor,
        high_size=(latent_h, latent_w),
        mid_size=(latent_h // 2, latent_w // 2),
    )
    encoder_hidden_states = torch.cat([text_tokens, sketch_out.sem_tokens], dim=1)

    # optional ref
    cross_attention_kwargs = None
    if train_stage == "ref" and ref_items is not None and len(ref_items) > 0:
        ref_imgs = []
        ref_bboxes = []

        for item in ref_items:
            if "image" not in item or "bbox_norm" not in item:
                raise ValueError("Each ref item must contain 'image' and 'bbox_norm'")
            ref_imgs.append(preprocess_ref(item["image"]))
            ref_bboxes.append(torch.tensor(item["bbox_norm"], dtype=torch.float))

        max_refs = len(ref_imgs)
        ref_imgs = torch.stack(ref_imgs, dim=0).unsqueeze(0).to(device=device, dtype=next(unet.parameters()).dtype)
        ref_bboxes = torch.stack(ref_bboxes, dim=0).unsqueeze(0).to(device=device)
        ref_valid_mask = torch.ones((1, max_refs), device=device, dtype=torch.float)

        B, N_ref, C, Hr, Wr = ref_imgs.shape
        ref_tokens = ref_adapter(ref_imgs.reshape(B * N_ref, C, Hr, Wr))
        K_ref, D_ref = ref_tokens.shape[1], ref_tokens.shape[2]
        ref_tokens = ref_tokens.reshape(B, N_ref, K_ref, D_ref)

        cross_attention_kwargs = {
            "ref_hidden_states": ref_tokens,
            "ref_bboxes": ref_bboxes,
            "ref_valid_mask": ref_valid_mask,
        }

    # runtime weak spatial injection
    injector = SketchSpatialInjector(unet)
    injector.register()
    injector.set_features(sketch_out.spatial_feats["mid"], sketch_out.spatial_feats["high"])

    # sample
    scheduler.set_timesteps(num_inference_steps, device=device)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)

    latents = torch.randn(
        (1, unet.config.in_channels, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=next(unet.parameters()).dtype,
    )

    try:
        for t in scheduler.timesteps:
            noise_pred = unet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            latents = scheduler.step(noise_pred, t, latents).prev_sample
    finally:
        injector.clear()
        injector.remove()

    pred = vae.decode(latents / vae.config.scaling_factor).sample.float().cpu()
    pred = denorm_line_tensor(pred)
    out_pil = postprocess_pred_to_pil(pred, sketch_meta, unpad_back=True)
    return out_pil


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for panel sketch-to-line model")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--sketch_path", type=str, required=True)
    parser.add_argument("--caption", type=str, default="")
    parser.add_argument("--ref_json", type=str, default=None, help="JSON file: [{'image':..., 'bbox_norm':[x0,y0,x1,y1]}, ...]")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lora_rank", type=int, default=4, help="Must match training LoRA rank")
    parser.add_argument("--ref_image_encoder_name_or_path", type=str, default="openai/clip-vit-large-patch14")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.device.startswith("cuda"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    models = load_models(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_dir=args.checkpoint_dir,
        lora_rank=args.lora_rank,
        ref_image_encoder_name_or_path=args.ref_image_encoder_name_or_path,
        device=torch.device(args.device),
        dtype=dtype,
    )

    ref_items = None
    if args.ref_json is not None:
        with open(args.ref_json, "r", encoding="utf-8") as f:
            ref_items = json.load(f)

    out = infer_one(
        models=models,
        sketch_path=args.sketch_path,
        caption=args.caption,
        ref_items=ref_items,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        device=args.device,
    )
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out.save(args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()