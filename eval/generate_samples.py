#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class SampleMeta:
    index: int
    sample_id: str
    raw: Dict[str, Any]
    pred_panel_path: Path
    gt_panel_path: Path
    sketch_panel_path: Path


def safe_name(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


def dtype_from_str(s: str) -> torch.dtype:
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    return torch.float32


def sha1_to_seed(text: str, base_seed: int) -> int:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return (int(h, 16) + int(base_seed)) % (2**31)


def ensure_project_imports(project_root: Path) -> Dict[str, Any]:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from data.panel_dataset_tr import MangaPanelIndexCollator, MangaPanelIndexDataset

    from models.attention_processor import IgnoreExtraKwargsAttnProcessor, MaskedRefAttentionProcessor
    from models.controlnext_injector import ControlNeXtInjector
    from models.controlnext_sketch import ControlNeXtSketchModel
    from models.local_sketch import LocalSketchQueryEncoder
    from models.ref_adapter import FrozenCLIPRefAdapter
    from models.ref_retargeter import RefTokenRetargeter
    from utils.checkpointing import load_checkpoint_payload
    from utils.image_ops import apply_valid_mask_white, denorm_line_tensor, postprocess_pred_to_pil

    return {
        "MangaPanelIndexCollator": MangaPanelIndexCollator,
        "MangaPanelIndexDataset": MangaPanelIndexDataset,
        "IgnoreExtraKwargsAttnProcessor": IgnoreExtraKwargsAttnProcessor,
        "MaskedRefAttentionProcessor": MaskedRefAttentionProcessor,
        "ControlNeXtInjector": ControlNeXtInjector,
        "ControlNeXtSketchModel": ControlNeXtSketchModel,
        "LocalSketchQueryEncoder": LocalSketchQueryEncoder,
        "FrozenCLIPRefAdapter": FrozenCLIPRefAdapter,
        "RefTokenRetargeter": RefTokenRetargeter,
        "load_checkpoint_payload": load_checkpoint_payload,
        "apply_valid_mask_white": apply_valid_mask_white,
        "denorm_line_tensor": denorm_line_tensor,
        "postprocess_pred_to_pil": postprocess_pred_to_pil,
    }


class FixedPromptCollator:
    def __init__(self, base_collator, fixed_prompt: Optional[str]):
        self.base_collator = base_collator
        self.fixed_prompt = fixed_prompt

    def __call__(self, batch):
        if self.fixed_prompt is None:
            return self.base_collator(batch)
        patched = []
        for item in batch:
            item2 = dict(item)
            item2["caption"] = self.fixed_prompt
            patched.append(item2)
        return self.base_collator(patched)


def get_unet_attention_hidden_size(unet: UNet2DConditionModel, name: str) -> int:
    if name.startswith("mid_block"):
        return unet.config.block_out_channels[-1]
    if name.startswith("up_blocks"):
        block_id = int(name.split(".")[1])
        return list(reversed(unet.config.block_out_channels))[block_id]
    if name.startswith("down_blocks"):
        block_id = int(name.split(".")[1])
        return unet.config.block_out_channels[block_id]
    return unet.config.cross_attention_dim


def read_index_rows(index_file: Path, num_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with index_file.open("r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= num_samples:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_sample_metas(index_rows: Sequence[Dict[str, Any]], output_root: Path) -> List[SampleMeta]:
    metas: List[SampleMeta] = []
    for idx, raw in enumerate(index_rows):
        sample_id = str(raw.get("sample_id", f"sample_{idx:06d}"))
        stem = safe_name(sample_id)
        metas.append(
            SampleMeta(
                index=idx,
                sample_id=sample_id,
                raw=raw,
                pred_panel_path=output_root / "pred_panel" / f"{stem}.png",
                gt_panel_path=output_root / "gt_panel" / f"{stem}.png",
                sketch_panel_path=output_root / "sketch_panel" / f"{stem}.png",
            )
        )
    return metas


def build_modules(args: argparse.Namespace, imports: Dict[str, Any], device: torch.device, weight_dtype: torch.dtype):
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    FrozenCLIPRefAdapter = imports["FrozenCLIPRefAdapter"]
    LocalSketchQueryEncoder = imports["LocalSketchQueryEncoder"]
    RefTokenRetargeter = imports["RefTokenRetargeter"]
    ControlNeXtSketchModel = imports["ControlNeXtSketchModel"]
    IgnoreExtraKwargsAttnProcessor = imports["IgnoreExtraKwargsAttnProcessor"]
    MaskedRefAttentionProcessor = imports["MaskedRefAttentionProcessor"]
    ControlNeXtInjector = imports["ControlNeXtInjector"]

    ref_adapter = FrozenCLIPRefAdapter(
        model_name_or_path=args.ref_image_encoder_name_or_path,
        cross_attn_dim=unet.config.cross_attention_dim,
        num_queries=args.num_ref_tokens,
    )
    local_sketch_query_encoder = LocalSketchQueryEncoder(
        out_dim=unet.config.cross_attention_dim,
        num_queries=args.num_local_queries,
    )
    ref_retargeter = RefTokenRetargeter(dim=unet.config.cross_attention_dim, depth=2, heads=8)
    controlnext_sketch = ControlNeXtSketchModel(
        controlnext_scale=args.controlnext_scale,
        cond_channels=2,
    )

    attn_procs = {}
    for name, old_proc in unet.attn_processors.items():
        if name.endswith("attn2.processor"):
            hidden_size = get_unet_attention_hidden_size(unet, name)
            attn_procs[name] = MaskedRefAttentionProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=unet.config.cross_attention_dim,
                ref_cross_attention_dim=unet.config.cross_attention_dim,
                text_scale=args.text_attn_scale,
            )
        else:
            attn_procs[name] = IgnoreExtraKwargsAttnProcessor(old_proc)
    unet.set_attn_processor(attn_procs)

    controlnext_injector = ControlNeXtInjector(unet)
    controlnext_injector.register()

    imports["load_checkpoint_payload"](
        args.checkpoint_path,
        unet=unet,
        ref_adapter=ref_adapter,
        local_sketch_query_encoder=local_sketch_query_encoder,
        ref_retargeter=ref_retargeter,
        controlnext_sketch=controlnext_sketch,
        mixed_precision={"fp16": "fp16", "bf16": "bf16", "fp32": None}[args.dtype],
    )

    vae.to(device=device, dtype=weight_dtype).eval()
    text_encoder.to(device=device, dtype=weight_dtype).eval()
    unet.to(device=device, dtype=weight_dtype).eval()
    ref_adapter.to(device=device, dtype=weight_dtype).eval()
    local_sketch_query_encoder.to(device=device, dtype=weight_dtype).eval()
    ref_retargeter.to(device=device, dtype=weight_dtype).eval()
    controlnext_sketch.to(device=device, dtype=weight_dtype).eval()

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "ref_adapter": ref_adapter,
        "local_sketch_query_encoder": local_sketch_query_encoder,
        "ref_retargeter": ref_retargeter,
        "controlnext_sketch": controlnext_sketch,
        "controlnext_injector": controlnext_injector,
    }


def _to_gray01_from_model_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[1] == 1:
        gray = x
    else:
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    if float(gray.min()) < -1e-6 or float(gray.max()) > 1.0 + 1e-6:
        gray = (gray + 1.0) / 2.0
    return gray.clamp(0.0, 1.0)


def _compute_ais_abstraction_scores(
    sketch_values: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    black_threshold: float,
    detail_ref: float,
    score_gamma: float,
) -> torch.Tensor:
    gray = _to_gray01_from_model_tensor(sketch_values)
    if valid_mask is None:
        vm = torch.ones_like(gray)
    else:
        vm = valid_mask[:, :1].float()
        if vm.shape[-2:] != gray.shape[-2:]:
            vm = F.interpolate(vm, size=gray.shape[-2:], mode="nearest")
    valid_area = vm.sum(dim=(1, 2, 3)).clamp(min=1.0)

    darkness = ((1.0 - gray) * vm).sum(dim=(1, 2, 3)) / valid_area
    line_ratio = (((gray < black_threshold).float()) * vm).sum(dim=(1, 2, 3)) / valid_area

    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
    grad_strength = (grad_mag * vm).sum(dim=(1, 2, 3)) / valid_area
    grad_strength = (grad_strength / 4.0).clamp(0.0, 1.0)

    detail_score = 0.45 * darkness + 0.40 * line_ratio + 0.15 * grad_strength
    detail_score = (detail_score / max(float(detail_ref), 1e-6)).clamp(0.0, 1.0)
    abstraction = (1.0 - detail_score).clamp(0.0, 1.0)
    if score_gamma != 1.0:
        abstraction = abstraction.pow(score_gamma)
    return abstraction


def _build_ais_timesteps(
    num_train_timesteps: int,
    num_inference_steps: int,
    abstraction_score: float,
    strength: float,
    power_min: float,
    power_max: float,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    score = float(np.clip(abstraction_score, 0.0, 1.0))
    strength = float(np.clip(strength, 0.0, 1.0))
    target_power = power_min + (power_max - power_min) * score
    power = 1.0 + strength * (target_power - 1.0)

    if num_inference_steps <= 1:
        return torch.tensor([num_train_timesteps - 1], device=device, dtype=torch.long), power

    u = np.linspace(0.0, 1.0, num_inference_steps, dtype=np.float64)
    mapped = 1.0 - np.power(u, power)

    timesteps = [int(num_train_timesteps - 1)]
    last = timesteps[0]
    for idx in range(1, num_inference_steps - 1):
        raw_t = int(round(mapped[idx] * (num_train_timesteps - 1)))
        remaining = (num_inference_steps - 1) - idx
        raw_t = min(raw_t, last - 1)
        raw_t = max(raw_t, remaining)
        timesteps.append(raw_t)
        last = raw_t
    timesteps.append(0)
    return torch.tensor(timesteps, device=device, dtype=torch.long), power


def _resolve_prev_timestep(
    scheduler: DDIMScheduler,
    timestep: torch.Tensor | int,
    timesteps_seq: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], dtype=torch.long)
    if timestep.ndim == 0:
        timestep = timestep[None]

    device = timestep.device
    cur = int(timestep[0].item())

    if timesteps_seq is not None:
        seq = timesteps_seq.to(device=device, dtype=torch.long).flatten()
        matches = (seq == cur).nonzero(as_tuple=False)
        if len(matches) == 0:
            raise ValueError(f"Current timestep {cur} not found in provided timesteps sequence.")
        idx = int(matches[0].item())
        if idx >= len(seq) - 1:
            return torch.full((1,), -1, device=device, dtype=torch.long)
        return seq[idx + 1 : idx + 2]

    step_ratio = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    return timestep - step_ratio


def _ddim_pred_x0_and_eps(scheduler: DDIMScheduler, model_output: torch.Tensor, timestep: torch.Tensor | int, sample: torch.Tensor):
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
    if timestep.ndim == 0:
        timestep = timestep[None]
    timestep = timestep.to(sample.device)

    alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    alpha_prod_t = alphas_cumprod[timestep].view(-1, 1, 1, 1)
    beta_prod_t = 1.0 - alpha_prod_t

    pred_type = scheduler.config.prediction_type
    if pred_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_epsilon = model_output
    elif pred_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()
    elif pred_type == "v_prediction":
        pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
    else:
        raise ValueError(f"Unsupported DDIM prediction_type: {pred_type}")

    return pred_original_sample, pred_epsilon


def _ddim_prev_sample_from_x0(
    scheduler: DDIMScheduler,
    pred_original_sample: torch.Tensor,
    pred_epsilon: torch.Tensor,
    timestep: torch.Tensor | int,
    timesteps_seq: Optional[torch.Tensor] = None,
):
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], device=pred_original_sample.device, dtype=torch.long)
    if timestep.ndim == 0:
        timestep = timestep[None]
    timestep = timestep.to(pred_original_sample.device)

    prev_timestep = _resolve_prev_timestep(scheduler, timestep, timesteps_seq=timesteps_seq).to(pred_original_sample.device)

    alphas_cumprod = scheduler.alphas_cumprod.to(device=pred_original_sample.device, dtype=pred_original_sample.dtype)
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0,
        alphas_cumprod[prev_timestep.clamp(min=0)],
        torch.full_like(prev_timestep, float(scheduler.final_alpha_cumprod), dtype=pred_original_sample.dtype),
    ).view(-1, 1, 1, 1)

    prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + (1.0 - alpha_prod_t_prev).sqrt() * pred_epsilon
    return prev_sample


def _apply_counter_regulation_guidance(
    pred_x0: torch.Tensor,
    vae: AutoencoderKL,
    valid_mask: Optional[torch.Tensor],
    gamma: float,
    grad_clip: float,
) -> torch.Tensor:
    pred_x0_leaf = pred_x0.detach().clone().requires_grad_(True)
    vae_dtype = next(vae.parameters()).dtype

    with torch.enable_grad():
        decoded = vae.decode(pred_x0_leaf.to(dtype=vae_dtype) / vae.config.scaling_factor).sample.float()
        img01 = ((decoded.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)

        if valid_mask is not None:
            vm = valid_mask.to(device=img01.device, dtype=img01.dtype)
            if vm.shape[1] == 1:
                vm = vm.repeat(1, img01.shape[1], 1, 1)
            img01 = img01 * vm + (1.0 - vm)

        gray = 0.299 * img01[:, 0:1] + 0.587 * img01[:, 1:2] + 0.114 * img01[:, 2:3]
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        grad_x = torch.clamp(grad_x, -grad_clip, grad_clip)
        grad_y = torch.clamp(grad_y, -grad_clip, grad_clip)

        edge_loss = -(grad_x.abs().mean() + grad_y.abs().mean())
        latent_grad = torch.autograd.grad(edge_loss, pred_x0_leaf, retain_graph=False, create_graph=False)[0]

    return (pred_x0 - gamma * latent_grad).detach()


def _should_apply_crg(args: argparse.Namespace, step_idx: int) -> bool:
    if not args.use_crg:
        return False
    if step_idx < args.crg_apply_from_step:
        return False
    if args.crg_apply_until_step >= 0 and step_idx > args.crg_apply_until_step:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate release-style panel outputs from ref-guided release benchmark jsonl.")
    parser.add_argument("--project_root", default="/data4/Sketch")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--crop_root", required=True)
    parser.add_argument("--index_file", required=True)
    parser.add_argument("--output_root", required=True)

    parser.add_argument('--fixed_prompt', type=str, default="Generate clean black-and-white manga line art, preserve sketch composition and character layout, black ink lines on white background, no color, no text. When reference images are provided, preserve the referenced character identity and appearance.", help='Use one fixed prompt for every sample. If set, per-sample captions are ignored everywhere.')
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--ref_resolution", type=int, default=224)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--text_attn_scale", type=float, default=0.35)
    parser.add_argument("--controlnext_scale", type=float, default=0.25)
    parser.add_argument("--ref_image_encoder_name_or_path", default="openai/clip-vit-large-patch14")
    parser.add_argument("--num_ref_tokens", type=int, default=8)
    parser.add_argument("--num_local_queries", type=int, default=4)
    parser.add_argument("--validation_num_inference_steps", type=int, default=30)
    parser.add_argument("--max_refs_per_panel", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--use_crg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Counter-based Regulation Guidance during inference.",
    )
    parser.add_argument(
        "--use_ais",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Abstraction-aware Importance Sampling during inference.",
    )

    parser.add_argument("--crg_gamma", type=float, default=60.0)
    parser.add_argument("--crg_grad_clip", type=float, default=0.001)
    parser.add_argument("--crg_apply_from_step", type=int, default=0)
    parser.add_argument("--crg_apply_until_step", type=int, default=-1)

    parser.add_argument("--ais_strength", type=float, default=1.0)
    parser.add_argument("--ais_power_min", type=float, default=0.70)
    parser.add_argument("--ais_power_max", type=float, default=2.00)
    parser.add_argument("--ais_black_threshold", type=float, default=0.85)
    parser.add_argument("--ais_detail_ref", type=float, default=0.20)
    parser.add_argument("--ais_score_gamma", type=float, default=1.0)
    return parser.parse_args()


def generate_release_outputs(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    weight_dtype = dtype_from_str(args.dtype)
    imports = ensure_project_imports(Path(args.project_root))

    index_rows = read_index_rows(Path(args.index_file), args.num_samples)
    metas = build_sample_metas(index_rows, Path(args.output_root))
    raw_map = {m.sample_id: m.raw for m in metas}

    MangaPanelIndexDataset = imports["MangaPanelIndexDataset"]
    MangaPanelIndexCollator = imports["MangaPanelIndexCollator"]
    apply_valid_mask_white = imports["apply_valid_mask_white"]
    denorm_line_tensor = imports["denorm_line_tensor"]
    postprocess_pred_to_pil = imports["postprocess_pred_to_pil"]

    modules = build_modules(args, imports, device, weight_dtype)

    dataset = MangaPanelIndexDataset(crop_root=args.crop_root, index_file=args.index_file, strict_exist_check=False)
    subset = Subset(dataset, list(range(len(metas))))
    base_collator = MangaPanelIndexCollator(
        tokenizer=modules["tokenizer"],
        resolution=args.resolution,
        caption_dropout_prob=0.0,
        sketch_dropout_prob=0.0,
        max_refs_per_panel=args.max_refs_per_panel,
        ref_resolution=args.ref_resolution,
        fixed_prompt=None,
    )
    collator = FixedPromptCollator(base_collator, args.fixed_prompt)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=False,
    )

    vae = modules["vae"]
    text_encoder = modules["text_encoder"]
    unet = modules["unet"]
    ref_adapter = modules["ref_adapter"]
    local_query = modules["local_sketch_query_encoder"]
    ref_retargeter = modules["ref_retargeter"]
    controlnext = modules["controlnext_sketch"]
    injector = modules["controlnext_injector"]

    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    scheduler.set_timesteps(args.validation_num_inference_steps, device=device)

    manifest_rows: List[Dict[str, Any]] = []
    offset = 0
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        for batch in loader:
            sample_ids = list(batch["sample_ids"])
            batch_metas = metas[offset : offset + len(sample_ids)]
            offset += len(sample_ids)

            need = args.overwrite or not all(m.pred_panel_path.exists() and m.gt_panel_path.exists() and m.sketch_panel_path.exists() for m in batch_metas)
            if not need:
                for meta in batch_metas:
                    row = raw_map[meta.sample_id]
                    manifest_rows.append({
                        "index": meta.index,
                        "sample_id": meta.sample_id,
                        "pred_panel_path": str(meta.pred_panel_path),
                        "gt_panel_path": str(meta.gt_panel_path),
                        "sketch_panel_path": str(meta.sketch_panel_path),
                        "ref_selected": row.get("ref_selected", []),
                    })
                continue

            with torch.no_grad():
                pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
                sketch_values = batch["sketch_values"].to(device=device, dtype=weight_dtype)
                valid_mask = batch["valid_mask"].to(device=device, dtype=weight_dtype)
                input_ids = batch["input_ids"].to(device=device)
                attention_mask = batch["attention_mask"].to(device=device)
                ref_imgs = batch["ref_pixel_values"].to(device=device, dtype=weight_dtype)
                ref_bboxes = batch["ref_bboxes"].to(device=device)
                ref_valid_mask = batch["ref_valid_mask"].to(device=device)
                panel_geoms = batch["panel_geoms"]

                text_tokens = text_encoder(input_ids, attention_mask=attention_mask, return_dict=False)[0]
                bsz, n_ref, c, hr, wr = ref_imgs.shape
                ref_tokens = ref_adapter(ref_imgs.reshape(bsz * n_ref, c, hr, wr))
                ref_tokens = ref_tokens.reshape(bsz, n_ref, ref_tokens.shape[1], ref_tokens.shape[2])

                role_q = local_query(sketch_values, valid_mask, ref_bboxes, ref_valid_mask)
                retargeted_ref_tokens = ref_retargeter(
                    text_tokens=text_tokens,
                    local_sketch_queries=role_q,
                    ref_tokens=ref_tokens,
                    text_attention_mask=attention_mask,
                    role_valid_mask=ref_valid_mask,
                )

            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            latent_h = pixel_values.shape[-2] // vae_scale_factor
            latent_w = pixel_values.shape[-1] // vae_scale_factor

            latent_list = []
            for sid in sample_ids:
                gen = torch.Generator(device=device).manual_seed(sha1_to_seed(sid, args.seed))
                latent_list.append(torch.randn((1, unet.config.in_channels, latent_h, latent_w), device=device, dtype=weight_dtype, generator=gen))
            latents = torch.cat(latent_list, dim=0)
            sample_generation_info: Dict[str, Dict[str, Any]] = {}
            final_latents: List[torch.Tensor] = []

            ais_scores = None
            if args.use_ais:
                ais_scores = _compute_ais_abstraction_scores(
                    sketch_values=sketch_values,
                    valid_mask=valid_mask,
                    black_threshold=args.ais_black_threshold,
                    detail_ref=args.ais_detail_ref,
                    score_gamma=args.ais_score_gamma,
                )

            for sample_idx, sid in enumerate(sample_ids):
                local_latents = latents[sample_idx:sample_idx + 1]
                local_text_tokens = text_tokens[sample_idx:sample_idx + 1]
                local_ref_bboxes = ref_bboxes[sample_idx:sample_idx + 1]
                local_ref_valid_mask = ref_valid_mask[sample_idx:sample_idx + 1]
                local_ref_hidden_states = retargeted_ref_tokens[sample_idx:sample_idx + 1]
                local_valid_mask = valid_mask[sample_idx:sample_idx + 1]
                local_sketch_values = sketch_values[sample_idx:sample_idx + 1]

                if args.use_ais:
                    abstraction_score = float(ais_scores[sample_idx].detach().cpu().item())
                    local_timesteps, local_ais_power = _build_ais_timesteps(
                        num_train_timesteps=scheduler.config.num_train_timesteps,
                        num_inference_steps=args.validation_num_inference_steps,
                        abstraction_score=abstraction_score,
                        strength=args.ais_strength,
                        power_min=args.ais_power_min,
                        power_max=args.ais_power_max,
                        device=device,
                    )
                else:
                    abstraction_score = None
                    local_ais_power = None
                    local_timesteps = scheduler.timesteps.to(device=device)

                local_cross_attention_kwargs = {
                    "ref_hidden_states": local_ref_hidden_states,
                    "ref_bboxes": local_ref_bboxes,
                    "ref_valid_mask": local_ref_valid_mask,
                }

                for step_idx, t in enumerate(local_timesteps):
                    if torch.is_tensor(t):
                        timestep_tensor = t.view(1) if t.ndim == 0 else t
                        timestep_scalar = int(timestep_tensor.flatten()[0].item())
                    else:
                        timestep_scalar = int(t)
                        timestep_tensor = torch.tensor([timestep_scalar], device=device, dtype=torch.long)

                    with torch.no_grad():
                        local_control = controlnext(torch.cat([local_sketch_values, local_valid_mask], dim=1), timestep=timestep_tensor)
                        injector.set_controls(local_control)
                        noise_pred = unet(
                            local_latents,
                            timestep_scalar,
                            encoder_hidden_states=local_text_tokens,
                            cross_attention_kwargs=local_cross_attention_kwargs,
                            return_dict=False,
                        )[0]
                        injector.clear()

                    pred_x0, pred_eps = _ddim_pred_x0_and_eps(scheduler, noise_pred, timestep_tensor, local_latents)
                    if _should_apply_crg(args, step_idx):
                        pred_x0 = _apply_counter_regulation_guidance(
                            pred_x0=pred_x0,
                            vae=vae,
                            valid_mask=local_valid_mask,
                            gamma=args.crg_gamma,
                            grad_clip=args.crg_grad_clip,
                        )
                    local_latents = _ddim_prev_sample_from_x0(
                        scheduler,
                        pred_x0,
                        pred_eps,
                        timestep_tensor,
                        timesteps_seq=local_timesteps,
                    ).to(dtype=weight_dtype)

                final_latents.append(local_latents)
                sample_generation_info[sid] = {
                    "use_crg": bool(args.use_crg),
                    "crg_gamma": float(args.crg_gamma),
                    "crg_grad_clip": float(args.crg_grad_clip),
                    "crg_apply_from_step": int(args.crg_apply_from_step),
                    "crg_apply_until_step": int(args.crg_apply_until_step),
                    "use_ais": bool(args.use_ais),
                    "ais_abstraction_score": abstraction_score,
                    "ais_power": local_ais_power,
                    "ais_strength": float(args.ais_strength),
                    "ais_power_min": float(args.ais_power_min),
                    "ais_power_max": float(args.ais_power_max),
                    "ais_black_threshold": float(args.ais_black_threshold),
                    "ais_detail_ref": float(args.ais_detail_ref),
                    "ais_score_gamma": float(args.ais_score_gamma),
                    "num_inference_steps": int(len(local_timesteps)),
                }

            latents = torch.cat(final_latents, dim=0)
            with torch.no_grad():
                vae_dtype = next(vae.parameters()).dtype
                pred_batch = vae.decode(latents.to(device=device, dtype=vae_dtype) / vae.config.scaling_factor).sample.float().cpu()
                pred_batch = denorm_line_tensor(pred_batch)
                pred_batch = apply_valid_mask_white(pred_batch, batch["valid_mask"].float().cpu())

                gt_batch = denorm_line_tensor(batch["pixel_values"].float().cpu())
                gt_batch = apply_valid_mask_white(gt_batch, batch["valid_mask"].float().cpu())

                sketch_batch = batch["sketch_values"].float().cpu().clamp(0.0, 1.0)
                sketch_batch = apply_valid_mask_white(sketch_batch, batch["valid_mask"].float().cpu())

            for i, meta in enumerate(batch_metas):
                pred_pil = postprocess_pred_to_pil(pred_batch[i:i+1], panel_geoms[i], unpad_back=True)
                gt_pil = postprocess_pred_to_pil(gt_batch[i:i+1], panel_geoms[i], unpad_back=True)
                sketch_pil = postprocess_pred_to_pil(sketch_batch[i:i+1], panel_geoms[i], unpad_back=True)

                meta.pred_panel_path.parent.mkdir(parents=True, exist_ok=True)
                meta.gt_panel_path.parent.mkdir(parents=True, exist_ok=True)
                meta.sketch_panel_path.parent.mkdir(parents=True, exist_ok=True)
                pred_pil.save(meta.pred_panel_path)
                gt_pil.save(meta.gt_panel_path)
                sketch_pil.save(meta.sketch_panel_path)

                row = raw_map[meta.sample_id]
                manifest_rows.append({
                    "index": meta.index,
                    "sample_id": meta.sample_id,
                    "pred_panel_path": str(meta.pred_panel_path),
                    "gt_panel_path": str(meta.gt_panel_path),
                    "sketch_panel_path": str(meta.sketch_panel_path),
                    "ref_selected": row.get("ref_selected", []),
                    "generation": sample_generation_info.get(meta.sample_id, {}),
                })
    finally:
        try:
            injector.remove()
        except Exception:
            pass

    manifest_path = output_root / "manifest_release_eval.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] release generation done -> {output_root}")
    print(f"[OK] manifest -> {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    args = parse_args()
    generate_release_outputs(args)
