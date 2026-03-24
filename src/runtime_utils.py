import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)
from transformers import CLIPVisionModel

import diffusers
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from models.sketch_encoder import SketchEncoder
from models.resampler import Resampler

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torchvision.transforms.functional as TF
from PIL import Image

from transformers import CLIPVisionModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.utils import convert_unet_state_dict_to_peft

from .resampler import Resampler
from diffusers.training_utils import cast_training_params
from accelerate import Accelerator
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")
accelerator_global = None
args_global = None

# =========================================================
# Frozen CLIP + trainable ref adapter
# =========================================================

class FrozenCLIPRefAdapter(nn.Module):
    """
    Frozen CLIP vision backbone + trainable projector + trainable resampler.
    Save/load ONLY trainable parts.
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
            seq = outputs.last_hidden_state  # [B, 1+N, C]
        seq = self.proj(seq)
        return self.resampler(seq)

    def trainable_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "proj": self.proj.state_dict(),
            "resampler": self.resampler.state_dict(),
        }

    def load_trainable_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]):
        self.proj.load_state_dict(state["proj"], strict=True)
        self.resampler.load_state_dict(state["resampler"], strict=True)


# =========================================================
# Weak runtime sketch spatial injector
# =========================================================

class SketchSpatialInjector:
    """
    Runtime-only weak spatial injection.
    Does NOT change UNet structure or parameters.
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

        # SD1.5 with 384 input -> latent 48x48
        # up_blocks[2] output channels ~640, spatial 24x24
        # up_blocks[3] output channels ~320, spatial 48x48
        self.handles.append(self.unet.up_blocks[2].register_forward_hook(hook_mid))
        self.handles.append(self.unet.up_blocks[3].register_forward_hook(hook_high))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []



# =========================================================
# Save / Load
# =========================================================

def save_checkpoint_payload(
    output_dir: str,
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    sketch_encoder: SketchEncoder,
    ref_adapter: Optional[FrozenCLIPRefAdapter],
):
    os.makedirs(output_dir, exist_ok=True)

    # 1) save LoRA in diffusers format
    unwrapped_unet = accelerator.unwrap_model(unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unwrapped_unet)
    )
    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )

    # 2) save custom trainable modules
    payload = {
        "train_stage": args_global.train_stage,
        "sketch_encoder": accelerator.unwrap_model(sketch_encoder).state_dict(),
    }

    if ref_adapter is not None:
        payload["ref_adapter_trainable"] = accelerator.unwrap_model(ref_adapter).trainable_state_dict()

    # 3) save custom attention processor states
    proc_state = {}
    unwrapped_unet = accelerator.unwrap_model(unet)
    for name, proc in unwrapped_unet.attn_processors.items():
        if isinstance(proc, nn.Module):
            sd = proc.state_dict()
            if len(sd) > 0:
                proc_state[name] = sd
    payload["attn_processors"] = proc_state

    torch.save(payload, os.path.join(output_dir, "extra_modules.pt"))


def resize_long_edge_and_pad_square(
    img: Image.Image,
    target_size: int,
    fill=(255, 255, 255),
    resample=Image.BILINEAR,
):
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


def preprocess_sketch_384(sketch_path: str):
    img = Image.open(sketch_path).convert("RGB")
    padded, meta = resize_long_edge_and_pad_square(img, 384)
    padded = padded.convert("L")
    x = TF.to_tensor(padded)  # [1,384,384]
    return x.unsqueeze(0), meta


def preprocess_ref_224(ref_path: str):
    img = Image.open(ref_path).convert("RGB")
    padded, _ = resize_long_edge_and_pad_square(img, 224, resample=Image.BICUBIC)
    x = TF.to_tensor(padded)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    x = (x - mean) / std
    return x


def denorm_line_tensor(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def postprocess_pred_to_pil(pred: torch.Tensor, meta: Dict, unpad_back: bool = True) -> Image.Image:
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

def load_checkpoint_payload(
    input_dir: str,
    unet: UNet2DConditionModel,
    sketch_encoder: SketchEncoder,
    ref_adapter: Optional[FrozenCLIPRefAdapter],
    mixed_precision: Optional[str],
):
    # load LoRA
    lora_state_dict, _ = StableDiffusionPipeline.lora_state_dict(input_dir)
    unet_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items() if k.startswith("unet.")}
    unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
    incompatible = set_peft_model_state_dict(unet, unet_state_dict, adapter_name="default")
    if incompatible is not None:
        unexpected = getattr(incompatible, "unexpected_keys", None)
        if unexpected:
            logger.warning(f"Unexpected LoRA keys while loading: {unexpected}")

    if mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    extra_path = os.path.join(input_dir, "extra_modules.pt")
    if not os.path.exists(extra_path):
        logger.warning(f"extra_modules.pt not found under {input_dir}, skipping extra module load.")
        return

    extra = torch.load(extra_path, map_location="cpu")
    sketch_encoder.load_state_dict(extra["sketch_encoder"], strict=True)

    if ref_adapter is not None and "ref_adapter_trainable" in extra:
        ref_adapter.load_trainable_state_dict(extra["ref_adapter_trainable"])

    if "attn_processors" in extra:
        for name, sd in extra["attn_processors"].items():
            if name in unet.attn_processors and isinstance(unet.attn_processors[name], nn.Module):
                unet.attn_processors[name].load_state_dict(sd, strict=True)
