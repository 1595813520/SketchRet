"""
Panel-level sketch-to-line fine-tuning for SD1.5 with:
- offline panel_index.jsonl dataset
- SketchEncoder (semantic tokens + weak spatial hints)
- optional masked ref cross-attention branch
- UNet LoRA
"""

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from tqdm.auto import tqdm

from src.runtime_utils import (
    FrozenCLIPRefAdapter,
    SketchSpatialInjector,
    save_checkpoint_payload,
    load_checkpoint_payload,
    get_unet_attention_hidden_size,
)

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module

# ===== local modules =====
from src.manga_dataset import (
    MangaPanelIndexDataset,
    MangaPanelIndexCollator,
    SameSpreadBatchSampler,
)
from models.sketch_encoder import SketchEncoder
from models.attention_processor import MaskedRefAttentionProcessor

if is_wandb_available():
    import wandb

check_min_version("0.37.0.dev0")
logger = get_logger(__name__, log_level="INFO")


# =========================================================
# Globals for unwrap helper
# 这个干嘛用的？
# =========================================================
accelerator_global = None
args_global = None


# =========================================================
# Helpers
# =========================================================

def unwrap_model(model):
    model = accelerator_global.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_sketch_gate(global_step: int, max_train_steps: int, warmup_ratio: float) -> float:
    warmup_steps = max(1, int(max_train_steps * warmup_ratio))
    if global_step >= warmup_steps:
        return 1.0
    return float(global_step) / float(warmup_steps)


# =========================================================
# Validation
# =========================================================

def denorm_line_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    [-1,1] -> [0,1]
    """
    return ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)

# 组织回整页
@torch.no_grad()
def save_validation_triplets(
    output_dir: str,
    global_step: int,
    sample_ids: List[str],
    sketch_batch: torch.Tensor,   # [B,1,384,384], [0,1]
    gt_batch: torch.Tensor,       # [B,3,384,384], [-1,1]
    pred_batch: torch.Tensor,     # [B,3,384,384], [-1,1] or [0,1]
):
    import os
    import torchvision.transforms.functional as TF

    save_dir = os.path.join(output_dir, "validation", f"step_{global_step:07d}")
    os.makedirs(save_dir, exist_ok=True)

    gt_batch = ((gt_batch.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
    if pred_batch.min() < 0:
        pred_batch = ((pred_batch.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
    else:
        pred_batch = pred_batch.clamp(0, 1)

    sketch_vis = sketch_batch.clamp(0, 1).repeat(1, 3, 1, 1)

    for i, sid in enumerate(sample_ids):
        triplet = torch.cat([sketch_vis[i], gt_batch[i], pred_batch[i]], dim=2)  # width concat
        pil = TF.to_pil_image(triplet.cpu())
        safe_name = sid.replace("/", "__").replace("::", "__")
        pil.save(os.path.join(save_dir, f"{safe_name}.png"))
        
# 单个panel的验证，生成sketch/gt/pred的拼接图，并保存到output_dir/validation/step_xxx/目录下
@torch.no_grad()
def run_validation(
    accelerator: Accelerator,
    args,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    sketch_encoder: SketchEncoder,
    ref_adapter: Optional[FrozenCLIPRefAdapter],
    spatial_injector: SketchSpatialInjector,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    weight_dtype: torch.dtype,
    fixed_val_examples: List[Dict[str, Any]],
    global_step: int,
):
    """
    Conditioned validation:
    log sketch / gt line / pred line triplets
    """
    if len(fixed_val_examples) == 0:
        return

    logger.info(f"Running conditioned validation at step={global_step} ...")

    # no dropout for validation
    val_collator = MangaPanelIndexCollator(
        tokenizer=tokenizer,
        train_stage=args.train_stage,
        caption_dropout_prob=0.0,
        sketch_dropout_prob=0.0,
        max_refs_per_panel=args.max_refs_per_panel,
    )
    batch = val_collator(fixed_val_examples)

    pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
    sketch_values = batch["sketch_values"].to(device=device, dtype=weight_dtype)
    input_ids = batch["input_ids"].to(device=device)

    # text
    text_tokens = text_encoder(input_ids, return_dict=False)[0]

    # sketch
    high_size = (48, 48)  # 384 / 8
    mid_size = (24, 24)
    role_bboxes = batch['ref_bboxes'].to(device=device) if args.train_stage == 'ref' else None
    role_valid_mask = batch['ref_valid_mask'].to(device=device) if args.train_stage == 'ref' else None
    sketch_out = sketch_encoder(
        sketch=sketch_values,
        high_size=high_size,
        mid_size=mid_size,
        role_bboxes=role_bboxes,
        role_valid_mask=role_valid_mask,
    )
    encoder_hidden_states = torch.cat([text_tokens, sketch_out.sem_tokens], dim=1)

    # ref (optional)
    cross_attention_kwargs = None
    if args.train_stage == "ref":
        ref_imgs = batch["ref_pixel_values"].to(device=device, dtype=weight_dtype)
        B, N_ref, C, Hr, Wr = ref_imgs.shape
        ref_imgs = ref_imgs.reshape(B * N_ref, C, Hr, Wr)

        ref_tokens = ref_adapter(ref_imgs)
        K_ref, D_ref = ref_tokens.shape[1], ref_tokens.shape[2]
        ref_tokens = ref_tokens.reshape(B, N_ref, K_ref, D_ref)

        if sketch_out.role_tokens is not None:
            role_tokens = sketch_out.role_tokens.to(device=device, dtype=ref_tokens.dtype)
            ref_tokens = torch.cat([role_tokens, ref_tokens], dim=2)

        cross_attention_kwargs = {
            "ref_hidden_states": ref_tokens,
            "ref_bboxes": batch["ref_bboxes"].to(device=device),
            "ref_valid_mask": batch["ref_valid_mask"].to(device=device),
        }

    # sampling scheduler
    val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    val_scheduler.set_timesteps(args.validation_num_inference_steps, device=device)

    B = pixel_values.shape[0]
    latents = torch.randn(
        (B, unet.config.in_channels, 48, 48),
        device=device,
        dtype=weight_dtype,
    )

    spatial_injector.set_features(sketch_out.spatial_feats["mid"], sketch_out.spatial_feats["high"])
    try:
        for t in val_scheduler.timesteps:
            noise_pred = unet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            latents = val_scheduler.step(noise_pred, t, latents).prev_sample
    finally:
        spatial_injector.clear()

    pred = vae.decode(latents / vae.config.scaling_factor).sample.float().cpu()
    gt = pixel_values.float().cpu()
    sk = sketch_values.float().cpu()

    pred = denorm_line_tensor(pred)
    gt = denorm_line_tensor(gt)
    sk = sk.clamp(0, 1).repeat(1, 3, 1, 1)  # [B,3,H,W]

    rows = []
    for i in range(B):
        row = torch.cat([sk[i], gt[i], pred[i]], dim=2)  # concatenate along width
        rows.append(row)
    vis = torch.stack(rows, dim=0)  # [B,3,H,3W]
    np_vis = (vis.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_images("validation_triplets", np_vis, global_step, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation_triplets": [
                        wandb.Image(img, caption=f"{batch['sample_ids'][i]} | sketch / gt / pred")
                        for i, img in enumerate(np_vis)
                    ]
                },
                step=global_step,
            )


# =========================================================
# Args
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Panel-level sketch-to-line SD1.5 LoRA training.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    parser.add_argument("--crop_root", type=str, required=True)
    parser.add_argument("--index_file", type=str, default="panel_index.jsonl")
    parser.add_argument("--train_stage", type=str, default="main", choices=["main", "ref"])
    parser.add_argument("--strict_exist_check", action="store_true")
    parser.add_argument("--max_refs_per_panel", type=int, default=3)

    parser.add_argument("--caption_dropout_prob", type=float, default=0.25)
    parser.add_argument("--sketch_dropout_prob", type=float, default=0.05)
    parser.add_argument("--sketch_gate_warmup_ratio", type=float, default=0.12)

    parser.add_argument("--num_sketch_sem_tokens", type=int, default=8)
    parser.add_argument("--freeze_sketch_backbone", action="store_true")

    parser.add_argument("--ref_image_encoder_name_or_path", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--num_ref_tokens", type=int, default=8)

    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned-panel-lora")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--noise_offset", type=float, default=0.0)
    parser.add_argument("--prediction_type", type=str, default=None)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # conditioned validation
    parser.add_argument("--validation_every_n_steps", type=int, default=1000)
    parser.add_argument("--num_validation_samples", type=int, default=4)
    parser.add_argument("--validation_num_inference_steps", type=int, default=30)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args

def main():
    global accelerator_global, args_global

    args = parse_args()
    args_global = args

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    accelerator_global = accelerator

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------
    # Base model
    # -----------------------------------------------------
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # freeze base modules
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # LoRA on UNet attention projections
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    if accelerator.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # -----------------------------------------------------
    # Conditioning modules
    # -----------------------------------------------------
    sketch_encoder = SketchEncoder(
        pretrained_backbone=True,
        freeze_backbone=args.freeze_sketch_backbone,
        cross_attn_dim=unet.config.cross_attention_dim,
        num_sem_queries=args.num_sketch_sem_tokens,
    )

    ref_adapter = None
    if args.train_stage == "ref":
        ref_adapter = FrozenCLIPRefAdapter(
            model_name_or_path=args.ref_image_encoder_name_or_path,
            cross_attn_dim=unet.config.cross_attention_dim,
            num_queries=args.num_ref_tokens,
        )

        # Replace only cross-attn processors
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

    # runtime spatial injector (no structure change)
    spatial_injector = SketchSpatialInjector(unet)
    spatial_injector.register()

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes for 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = []
    trainable_params += [p for p in unet.parameters() if p.requires_grad]
    trainable_params += list(sketch_encoder.parameters())
    if ref_adapter is not None:
        trainable_params += [p for p in ref_adapter.parameters() if p.requires_grad]

    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------
    train_dataset = MangaPanelIndexDataset(
        crop_root=args.crop_root,
        index_file=args.index_file,
        train_stage=args.train_stage,
        load_ref_images=True,
        strict_exist_check=args.strict_exist_check,
    )

    collator = MangaPanelIndexCollator(
        tokenizer=tokenizer,
        train_stage=args.train_stage,
        caption_dropout_prob=args.caption_dropout_prob,
        sketch_dropout_prob=args.sketch_dropout_prob,
        max_refs_per_panel=args.max_refs_per_panel,
    )

    batch_sampler = SameSpreadBatchSampler(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=False,
        seed=args.seed if args.seed is not None else 42,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # fixed validation subset
    fixed_val_examples = []
    for i in range(min(args.num_validation_samples, len(train_dataset))):
        fixed_val_examples.append(train_dataset[i])

    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # -----------------------------------------------------
    # Prepare
    # -----------------------------------------------------
    if ref_adapter is not None:
        unet, sketch_encoder, ref_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, sketch_encoder, ref_adapter, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, sketch_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, sketch_encoder, optimizer, train_dataloader, lr_scheduler
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("panel-sketch-to-line", config=vars(args))

    # -----------------------------------------------------
    # Resume
    # -----------------------------------------------------
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint != "latest":
            ckpt_path = args.resume_from_checkpoint
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            ckpt_path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None

        if ckpt_path is not None and os.path.exists(ckpt_path):
            accelerator.print(f"Resuming from checkpoint {ckpt_path}")
            accelerator.load_state(ckpt_path)
            load_checkpoint_payload(
                ckpt_path,
                unet=accelerator.unwrap_model(unet),
                sketch_encoder=accelerator.unwrap_model(sketch_encoder),
                ref_adapter=accelerator.unwrap_model(ref_adapter) if ref_adapter is not None else None,
                mixed_precision=args.mixed_precision,
            )
            global_step = int(os.path.basename(ckpt_path).split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found, training from scratch.")

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Train stage = {args.train_stage}")
    logger.info(f"  Batch size/device = {args.train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  UNet trainable params = {count_trainable_params(accelerator.unwrap_model(unet)):,}")
    logger.info(f"  SketchEncoder trainable params = {count_trainable_params(accelerator.unwrap_model(sketch_encoder)):,}")
    if ref_adapter is not None:
        logger.info(f"  RefAdapter trainable params = {count_trainable_params(accelerator.unwrap_model(ref_adapter)):,}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # -----------------------------------------------------
    # Train loop
    # -----------------------------------------------------
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        sketch_encoder.train()
        if ref_adapter is not None:
            ref_adapter.train()

        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # line_gt -> latent
                latents = vae.encode(batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )

                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # text
                text_tokens = text_encoder(batch["input_ids"].to(accelerator.device), return_dict=False)[0]

                # sketch
                latent_h, latent_w = noisy_latents.shape[-2:]   # 384 -> 48x48
                high_size = (latent_h, latent_w)
                mid_size = (latent_h // 2, latent_w // 2)

                role_bboxes = batch['ref_bboxes'].to(device=accelerator.device) if args.train_stage == 'ref' else None
                role_valid_mask = batch['ref_valid_mask'].to(device=accelerator.device) if args.train_stage == 'ref' else None
                sketch_out = sketch_encoder(
                    sketch=batch["sketch_values"].to(device=accelerator.device, dtype=weight_dtype),
                    high_size=high_size,
                    mid_size=mid_size,
                    role_bboxes=role_bboxes,
                    role_valid_mask=role_valid_mask,
                )

                sketch_keep = batch["sketch_keep_mask"].to(accelerator.device).view(-1, 1, 1)
                sketch_gate = get_sketch_gate(global_step, args.max_train_steps, args.sketch_gate_warmup_ratio)

                sem_tokens = sketch_out.sem_tokens * sketch_keep * sketch_gate
                spatial_mid = sketch_out.spatial_feats["mid"] * sketch_keep.view(-1, 1, 1, 1) * sketch_gate
                spatial_high = sketch_out.spatial_feats["high"] * sketch_keep.view(-1, 1, 1, 1) * sketch_gate

                encoder_hidden_states = torch.cat([text_tokens, sem_tokens], dim=1)

                # inject weak spatial hints
                spatial_injector.set_features(spatial_mid, spatial_high)

                # target
                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # ref branch
                cross_attention_kwargs = None
                if args.train_stage == "ref":
                    ref_imgs = batch["ref_pixel_values"].to(device=accelerator.device, dtype=weight_dtype)  # [B,N,3,224,224]
                    B, N_ref, C, Hr, Wr = ref_imgs.shape
                    ref_imgs = ref_imgs.reshape(B * N_ref, C, Hr, Wr)

                    ref_tokens = ref_adapter(ref_imgs)  # [B*N_ref, K_ref, D]
                    K_ref, D_ref = ref_tokens.shape[1], ref_tokens.shape[2]
                    ref_tokens = ref_tokens.reshape(B, N_ref, K_ref, D_ref)

                    cross_attention_kwargs = {
                        "ref_hidden_states": ref_tokens,
                        "ref_bboxes": batch["ref_bboxes"].to(device=accelerator.device),
                        "ref_valid_mask": batch["ref_valid_mask"].to(device=accelerator.device),
                    }

                try:
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                finally:
                    spatial_injector.clear()

                # diffusion loss only
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack(
                        [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                    ).min(dim=1)[0]

                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "sketch_gate": sketch_gate,
                    },
                    step=global_step,
                )
                train_loss = 0.0

                # conditioned validation
                if args.validation_every_n_steps > 0 and global_step % args.validation_every_n_steps == 0:
                    if accelerator.is_main_process:
                        run_validation(
                            accelerator=accelerator,
                            args=args,
                            vae=vae,
                            text_encoder=text_encoder,
                            unet=unet,
                            sketch_encoder=sketch_encoder,
                            ref_adapter=ref_adapter,
                            spatial_injector=spatial_injector,
                            tokenizer=tokenizer,
                            device=accelerator.device,
                            weight_dtype=weight_dtype,
                            fixed_val_examples=fixed_val_examples,
                            global_step=global_step,
                        )

                # checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing = checkpoints[:num_to_remove]
                                logger.info(f"Removing old checkpoints: {', '.join(removing)}")
                                for ckpt in removing:
                                    shutil.rmtree(os.path.join(args.output_dir, ckpt), ignore_errors=True)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        save_checkpoint_payload(
                            output_dir=save_path,
                            accelerator=accelerator,
                            unet=unet,
                            sketch_encoder=sketch_encoder,
                            ref_adapter=ref_adapter,
                        )
                        logger.info(f"Saved checkpoint to {save_path}")

            progress_bar.set_postfix(
                step_loss=float(loss.detach().item()),
                lr=float(lr_scheduler.get_last_lr()[0]),
            )

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_checkpoint_payload(
            output_dir=args.output_dir,
            accelerator=accelerator,
            unet=unet,
            sketch_encoder=sketch_encoder,
            ref_adapter=ref_adapter,
        )

    spatial_injector.remove()
    accelerator.end_training()


if __name__ == "__main__":
    main()