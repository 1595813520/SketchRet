from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, is_wandb_available
from peft import LoraConfig
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from data.panel_dataset_tr import MangaPanelIndexCollator, MangaPanelIndexDataset, SpreadGroupedSampler
from models.attention_processor import IgnoreExtraKwargsAttnProcessor, MaskedRefAttentionProcessor
from models.controlnext_injector import ControlNeXtInjector
from models.controlnext_sketch import ControlNeXtSketchModel
from models.local_sketch import LocalSketchQueryEncoder
from models.ref_adapter import FrozenCLIPRefAdapter
from models.ref_retargeter import RefTokenRetargeter
from utils.checkpointing import load_checkpoint_payload, save_checkpoint_payload
from utils.image_ops import denorm_line_tensor
from utils.losses import (
    bg_clean_loss,
    compute_masked_diffusion_mse_loss,
    fg_detail_loss,
    gradient_consistency_loss,
)
from utils.validation import run_validation

check_min_version('0.24.0')
logger = get_logger(__name__, log_level='INFO')

PHASE_A = 'phase_a_control_warmup'
PHASE_B = 'phase_b_retarget_warmup'
PHASE_C = 'phase_c_joint'

def get_unet_attention_hidden_size(unet: UNet2DConditionModel, name: str) -> int:
    if name.startswith('mid_block'):
        return unet.config.block_out_channels[-1]
    if name.startswith('up_blocks'):
        block_id = int(name.split('.')[1])
        return list(reversed(unet.config.block_out_channels))[block_id]
    if name.startswith('down_blocks'):
        block_id = int(name.split('.')[1])
        return unet.config.block_out_channels[block_id]
    return unet.config.cross_attention_dim


def count_trainable_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _set_params_trainable(params, flag: bool) -> None:
    for p in params:
        p.requires_grad = flag


def _unique_params(params):
    out = []
    seen = set()
    for p in params:
        if p is None:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def get_unet_lora_params(unet: UNet2DConditionModel):
    return [p for name, p in unet.named_parameters() if 'lora_' in name]


def get_unet_attn_processor_params(unet: UNet2DConditionModel):
    params = []
    for proc in unet.attn_processors.values():
        if isinstance(proc, torch.nn.Module):
            params.extend(list(proc.parameters()))
    return _unique_params(params)


def get_unet_trainable_subset(unet: UNet2DConditionModel):
    return _unique_params(get_unet_lora_params(unet) + get_unet_attn_processor_params(unet))


def set_unet_trainable_subset(unet: UNet2DConditionModel, flag: bool) -> None:
    unet.eval() if not flag else unet.train()
    for p in unet.parameters():
        p.requires_grad = False
    _set_params_trainable(get_unet_trainable_subset(unet), flag)


def get_ref_adapter_trainable_subset(ref_adapter):
    return _unique_params(list(ref_adapter.proj.parameters()) + list(ref_adapter.resampler.parameters()))


def set_ref_adapter_trainable_subset(ref_adapter, flag: bool) -> None:
    ref_adapter.eval() if not flag else ref_adapter.train()
    ref_adapter.vision_encoder.eval()
    ref_adapter.vision_encoder.requires_grad_(False)
    for p in ref_adapter.parameters():
        p.requires_grad = False
    _set_params_trainable(get_ref_adapter_trainable_subset(ref_adapter), flag)


def set_simple_module_trainable(module, flag: bool) -> None:
    module.train() if flag else module.eval()
    module.requires_grad_(flag)


def get_phase_boundaries(max_train_steps: int, phase_a_ratio: float, phase_b_ratio: float) -> Tuple[int, int]:
    phase_a_steps = int(max_train_steps * phase_a_ratio)
    phase_b_steps = int(max_train_steps * phase_b_ratio)
    phase_a_end = max(0, min(max_train_steps, phase_a_steps))
    phase_b_end = max(phase_a_end, min(max_train_steps, phase_a_steps + phase_b_steps))
    return phase_a_end, phase_b_end


def get_phase(global_step: int, phase_a_end: int, phase_b_end: int) -> str:
    if global_step < phase_a_end:
        return PHASE_A
    if global_step < phase_b_end:
        return PHASE_B
    return PHASE_C


def apply_training_phase(
    accelerator: Accelerator,
    phase_name: str,
    unet,
    ref_adapter,
    local_sketch_query_encoder,
    ref_retargeter,
    controlnext_sketch,
) -> None:
    unet_mod = accelerator.unwrap_model(unet)
    ref_adapter_mod = accelerator.unwrap_model(ref_adapter)
    local_query_mod = accelerator.unwrap_model(local_sketch_query_encoder)
    retarget_mod = accelerator.unwrap_model(ref_retargeter)
    control_mod = accelerator.unwrap_model(controlnext_sketch)

    # Shared full-sketch query encoder now serves both the gated control branch
    # and the reference retargeting path, so it should remain trainable in all phases.
    if phase_name == PHASE_A:
        set_unet_trainable_subset(unet_mod, True)
        set_simple_module_trainable(control_mod, True)
        set_ref_adapter_trainable_subset(ref_adapter_mod, False)
        set_simple_module_trainable(local_query_mod, True)
        set_simple_module_trainable(retarget_mod, False)
    elif phase_name == PHASE_B:
        set_unet_trainable_subset(unet_mod, False)
        set_simple_module_trainable(control_mod, False)
        set_ref_adapter_trainable_subset(ref_adapter_mod, True)
        set_simple_module_trainable(local_query_mod, True)
        set_simple_module_trainable(retarget_mod, True)
    elif phase_name == PHASE_C:
        set_unet_trainable_subset(unet_mod, True)
        set_simple_module_trainable(control_mod, True)
        set_ref_adapter_trainable_subset(ref_adapter_mod, True)
        set_simple_module_trainable(local_query_mod, True)
        set_simple_module_trainable(retarget_mod, True)
    else:
        raise ValueError(f'Unknown phase: {phase_name}')


def parse_args():
    parser = argparse.ArgumentParser(description='Sketch-first retargeted line-art training with query-gated ControlNeXt.')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True)
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--variant', type=str, default=None)

    parser.add_argument('--crop_root', type=str, required=True)
    parser.add_argument('--index_file', type=str, required=True)
    parser.add_argument('--validation_index_file', type=str, default=None, help='Optional preview/validation split (e.g. splits/test_ref.jsonl). If omitted and index_file is panel_index.jsonl, the script will auto-detect splits/test_ref.jsonl.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--logging_dir', type=str, default='logs')
    parser.add_argument('--report_to', type=str, default='tensorboard')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--strict_exist_check', action='store_true')

    parser.add_argument('--resolution', type=int, default=384)
    parser.add_argument('--ref_resolution', type=int, default=224)
    parser.add_argument('--max_refs_per_panel', type=int, default=3)
    parser.add_argument('--caption_dropout_prob', type=float, default=0.0)
    parser.add_argument('--fixed_prompt', type=str, default="Generate clean black-and-white manga line art, preserve sketch composition and character layout, black ink lines on white background, no color, no text. When reference images are provided, preserve the referenced character identity and appearance.", help='Use one fixed prompt for every sample. If set, per-sample captions are ignored everywhere.')
    parser.add_argument('--sketch_dropout_prob', type=float, default=0.0)
    parser.add_argument('--ref_image_encoder_name_or_path', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--num_ref_tokens', type=int, default=8)
    parser.add_argument('--num_local_queries', type=int, default=4)
    parser.add_argument('--local_sketch_roi_size', type=int, default=12)
    parser.add_argument('--local_sketch_input_downsample', type=int, default=4)
    parser.add_argument('--local_sketch_min_confidence', type=float, default=0.25)
    parser.add_argument('--local_sketch_residual_weight', type=float, default=4.0)

    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=200)
    parser.add_argument('--max_train_steps', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true')

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--scale_lr', action='store_true', default=False)
    parser.add_argument('--lr_scheduler', type=str, default='constant')
    parser.add_argument('--lr_warmup_steps', type=int, default=500)
    parser.add_argument('--snr_gamma', type=float, default=5.0)
    parser.add_argument('--use_8bit_adam', action='store_true')
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--mixed_precision', type=str, default=None, choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--noise_offset', type=float, default=0.0)
    parser.add_argument('--prediction_type', type=str, default=None)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--controlnext_scale', type=float, default=0.25)
    parser.add_argument('--text_attn_scale', type=float, default=0.35)

    parser.add_argument('--phase_a_ratio', type=float, default=0.10)
    parser.add_argument('--phase_b_ratio', type=float, default=0.15)

    parser.add_argument('--checkpointing_steps', type=int, default=1000)
    parser.add_argument('--checkpoints_total_limit', type=int, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    parser.add_argument('--validation_every_n_steps', type=int, default=1000)
    parser.add_argument('--num_validation_samples', type=int, default=8)
    parser.add_argument('--validation_num_inference_steps', type=int, default=30)
    parser.add_argument('--lambda_x0', type=float, default=0.0)
    parser.add_argument('--lambda_bg', type=float, default=0.0)
    parser.add_argument('--lambda_fg', type=float, default=0.0)
    parser.add_argument('--lambda_grad', type=float, default=0.0)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def compute_pred_x0(noise_scheduler, model_pred, noisy_latents, noise, timesteps):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sigma_t = (1 - alpha_t).sqrt()
    alpha_sqrt = alpha_t.sqrt()
    if noise_scheduler.config.prediction_type == 'epsilon':
        pred_x0 = (noisy_latents - sigma_t * model_pred) / alpha_sqrt
    elif noise_scheduler.config.prediction_type == 'v_prediction':
        pred_x0 = alpha_sqrt * noisy_latents - sigma_t * model_pred
    else:
        raise ValueError(f'Unsupported prediction_type: {noise_scheduler.config.prediction_type}')
    return pred_x0


def select_fixed_validation_examples(train_dataset, num_validation_samples: int, fixed_prompt: Optional[str] = None):
    """Prefer reference-supported examples for validation previews on the new dataset."""
    if num_validation_samples <= 0 or len(train_dataset) == 0:
        return []

    ref_first_indices = []
    fallback_indices = []
    for idx, ex in enumerate(train_dataset.examples):
        has_valid_ref = any(ref.get('valid', False) for ref in ex.refs)
        if has_valid_ref:
            ref_first_indices.append(idx)
        else:
            fallback_indices.append(idx)

    chosen_indices = (ref_first_indices + fallback_indices)[: min(num_validation_samples, len(train_dataset))]
    fixed_val_examples = []
    for idx in chosen_indices:
        ex = train_dataset[idx]
        if fixed_prompt is not None:
            ex = dict(ex)
            ex['caption'] = fixed_prompt
        fixed_val_examples.append(ex)
    return fixed_val_examples


def resolve_dataset_paths(index_file: str, validation_index_file: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Make split usage safer for the Turing dataset.

    If the user accidentally passes panel_index.jsonl, prefer splits/train.jsonl for
    training and, when available, use splits/test_ref.jsonl for validation previews.

    Important: this helper may run before Accelerator() is created, so it must not
    use `accelerate.logging.get_logger(...)`.
    """
    train_index_file = index_file
    val_index_file = validation_index_file

    index_path = Path(index_file)
    if index_path.name == 'panel_index.jsonl':
        split_root = index_path.parent / 'splits'
        train_candidate = split_root / 'train.jsonl'
        val_candidate = split_root / 'test_ref.jsonl'

        if train_candidate.exists():
            logging.warning(
                f"index_file points to panel_index.jsonl; switching training split to {train_candidate}."
            )
            train_index_file = str(train_candidate)

        if val_index_file is None and val_candidate.exists():
            val_index_file = str(val_candidate)
            logging.info(f"Using validation preview split from {val_candidate}.")

    return train_index_file, val_index_file


def main():
    args = parse_args()
    # args.index_file, args.validation_index_file = resolve_dataset_paths(args.index_file, args.validation_index_file)

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M',
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.fixed_prompt is not None and args.caption_dropout_prob > 0.0:
        logger.info(f"fixed_prompt is set; forcing caption_dropout_prob from {args.caption_dropout_prob} to 0.0.")
        args.caption_dropout_prob = 0.0

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer', revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder', revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae', revision=args.revision, variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='unet', revision=args.revision, variant=args.variant)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights='gaussian',
        target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
    )
    unet.add_adapter(unet_lora_config)

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    if accelerator.mixed_precision == 'fp16':
        cast_training_params(unet, dtype=torch.float32)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    ref_adapter = FrozenCLIPRefAdapter(
        model_name_or_path=args.ref_image_encoder_name_or_path,
        cross_attn_dim=unet.config.cross_attention_dim,
        num_queries=args.num_ref_tokens,
    )
    local_sketch_query_encoder = LocalSketchQueryEncoder(
        out_dim=unet.config.cross_attention_dim,
        num_queries=args.num_local_queries,
        roi_size=args.local_sketch_roi_size,
        input_downsample=args.local_sketch_input_downsample,
        min_confidence=args.local_sketch_min_confidence,
        residual_weight=args.local_sketch_residual_weight,
    )
    ref_retargeter = RefTokenRetargeter(dim=unet.config.cross_attention_dim, depth=2, heads=8)
    controlnext_sketch = ControlNeXtSketchModel(
        controlnext_scale=args.controlnext_scale,
        cond_channels=2,
    )

    attn_procs = {}
    for name, old_proc in unet.attn_processors.items():
        if name.endswith('attn2.processor'):
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

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError('Please install bitsandbytes for 8-bit Adam.') from e
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    potential_trainable_params = _unique_params(
        get_unet_trainable_subset(unet)
        + get_ref_adapter_trainable_subset(ref_adapter)
        + list(local_sketch_query_encoder.parameters())
        + list(ref_retargeter.parameters())
        + list(controlnext_sketch.parameters())
    )

    optimizer = optimizer_cls(
        potential_trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = MangaPanelIndexDataset(
        crop_root=args.crop_root,
        index_file=args.index_file,
        strict_exist_check=args.strict_exist_check,
    )
    val_dataset = None
    if args.validation_index_file is not None:
        val_dataset = MangaPanelIndexDataset(
            crop_root=args.crop_root,
            index_file=args.validation_index_file,
            strict_exist_check=args.strict_exist_check,
        )

    base_collator = MangaPanelIndexCollator(
        tokenizer=tokenizer,
        resolution=args.resolution,
        caption_dropout_prob=args.caption_dropout_prob,
        sketch_dropout_prob=args.sketch_dropout_prob,
        max_refs_per_panel=args.max_refs_per_panel,
        ref_resolution=args.ref_resolution,
    )

    if args.fixed_prompt is not None:
        class _FixedPromptCollator:
            def __init__(self, base_collator, fixed_prompt: str):
                self.base_collator = base_collator
                self.fixed_prompt = fixed_prompt
                self.tokenizer = base_collator.tokenizer

            def __call__(self, batch):
                patched_batch = []
                for item in batch:
                    item2 = dict(item)
                    item2['caption'] = self.fixed_prompt
                    patched_batch.append(item2)
                return self.base_collator(patched_batch)

        collator = _FixedPromptCollator(base_collator, args.fixed_prompt)
    else:
        collator = base_collator
    sampler = SpreadGroupedSampler(train_dataset, shuffle_spreads=True, shuffle_within_spread=False, seed=args.seed if args.seed is not None else 42)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    preview_dataset = val_dataset if val_dataset is not None else train_dataset
    fixed_val_examples = select_fixed_validation_examples(
        train_dataset=preview_dataset,
        num_validation_samples=args.num_validation_samples,
        fixed_prompt=args.fixed_prompt,
    )

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

    unet, ref_adapter, local_sketch_query_encoder, ref_retargeter, controlnext_sketch, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet,
        ref_adapter,
        local_sketch_query_encoder,
        ref_retargeter,
        controlnext_sketch,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    phase_a_end, phase_b_end = get_phase_boundaries(args.max_train_steps, args.phase_a_ratio, args.phase_b_ratio)

    if accelerator.is_main_process:
        accelerator.init_trackers('sketch-retarget-controlnext', config=vars(args))

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint != 'latest':
            ckpt_path = args.resume_from_checkpoint
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith('checkpoint-')]
            dirs = sorted(dirs, key=lambda x: int(x.split('-')[1]))
            ckpt_path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None

        if ckpt_path is not None and os.path.exists(ckpt_path):
            accelerator.print(f'Resuming from checkpoint {ckpt_path}')
            accelerator.load_state(ckpt_path)
            load_checkpoint_payload(
                ckpt_path,
                unet=accelerator.unwrap_model(unet),
                ref_adapter=accelerator.unwrap_model(ref_adapter),
                local_sketch_query_encoder=accelerator.unwrap_model(local_sketch_query_encoder),
                ref_retargeter=accelerator.unwrap_model(ref_retargeter),
                controlnext_sketch=accelerator.unwrap_model(controlnext_sketch),
                mixed_precision=args.mixed_precision,
            )
            global_step = int(os.path.basename(ckpt_path).split('-')[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found, training from scratch.")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info('***** Running training *****')
    logger.info(f'  Train index file = {args.index_file}')
    if args.validation_index_file is not None:
        logger.info(f'  Validation preview index file = {args.validation_index_file}')
    logger.info(f'  Num examples = {len(train_dataset)}')
    if hasattr(train_dataset, 'num_examples_with_ref'):
        logger.info(f'  Examples with at least one valid ref = {train_dataset.num_examples_with_ref}')
    if hasattr(train_dataset, 'num_groups'):
        logger.info(f'  Grouped units (side-page/page/spread) = {train_dataset.num_groups}')
    if val_dataset is not None:
        logger.info(f'  Num validation preview examples = {len(val_dataset)}')
    logger.info(f'  Num epochs = {args.num_train_epochs}')
    logger.info(f'  Batch size/device = {args.train_batch_size}')
    logger.info(f'  Total batch size = {total_batch_size}')
    logger.info(f'  Gradient accumulation = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    logger.info(f'  Phase A end step = {phase_a_end}')
    logger.info(f'  Phase B end step = {phase_b_end}')
    logger.info(f'  UNet trainable params = {count_trainable_params(accelerator.unwrap_model(unet)):,}')
    logger.info(f'  RefAdapter trainable params = {count_trainable_params(accelerator.unwrap_model(ref_adapter)):,}')
    logger.info(f'  LocalQuery trainable params = {count_trainable_params(accelerator.unwrap_model(local_sketch_query_encoder)):,}')
    logger.info(f'  Retargeter trainable params = {count_trainable_params(accelerator.unwrap_model(ref_retargeter)):,}')
    logger.info(f'  ControlNeXtSketch trainable params = {count_trainable_params(accelerator.unwrap_model(controlnext_sketch)):,}')

    current_phase = get_phase(global_step, phase_a_end, phase_b_end)
    apply_training_phase(
        accelerator=accelerator,
        phase_name=current_phase,
        unet=unet,
        ref_adapter=ref_adapter,
        local_sketch_query_encoder=local_sketch_query_encoder,
        ref_retargeter=ref_retargeter,
        controlnext_sketch=controlnext_sketch,
    )
    logger.info(f'  Starting phase = {current_phase}')

    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc='Steps', disable=not accelerator.is_local_main_process)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            phase_now = get_phase(global_step, phase_a_end, phase_b_end)
            if phase_now != current_phase:
                current_phase = phase_now
                apply_training_phase(
                    accelerator=accelerator,
                    phase_name=current_phase,
                    unet=unet,
                    ref_adapter=ref_adapter,
                    local_sketch_query_encoder=local_sketch_query_encoder,
                    ref_retargeter=ref_retargeter,
                    controlnext_sketch=controlnext_sketch,
                )
                logger.info(f'Switched training phase to {current_phase} at global_step={global_step}')

            with accelerator.accumulate(unet):
                pixel_values = batch['pixel_values'].to(device=accelerator.device, dtype=weight_dtype)
                sketch_values = batch['sketch_values'].to(device=accelerator.device, dtype=weight_dtype)
                valid_mask = batch['valid_mask'].to(device=accelerator.device, dtype=weight_dtype)
                input_ids = batch['input_ids'].to(device=accelerator.device)
                attention_mask = batch['attention_mask'].to(device=accelerator.device)
                ref_imgs = batch['ref_pixel_values'].to(device=accelerator.device, dtype=weight_dtype)
                ref_bboxes = batch['ref_bboxes'].to(device=accelerator.device)
                ref_valid_mask = batch['ref_valid_mask'].to(device=accelerator.device)

                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise = noise + args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                text_tokens = text_encoder(input_ids, attention_mask=attention_mask, return_dict=False)[0]
                control_inputs = torch.cat([sketch_values, valid_mask], dim=1)
                control = controlnext_sketch(control_inputs, timesteps)

                B, N_ref, C, Hr, Wr = ref_imgs.shape
                ref_tokens = ref_adapter(ref_imgs.reshape(B * N_ref, C, Hr, Wr))
                ref_tokens = ref_tokens.reshape(B, N_ref, ref_tokens.shape[1], ref_tokens.shape[2])
                character_queries = local_sketch_query_encoder(
                    sketch_values,
                    valid_mask,
                    ref_bboxes,
                    ref_valid_mask,
                    timesteps=timesteps,
                    num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                )
                retargeted_ref_tokens = ref_retargeter(
                    text_tokens=text_tokens,
                    local_sketch_queries=character_queries,
                    ref_tokens=ref_tokens,
                    text_attention_mask=attention_mask,
                    role_valid_mask=ref_valid_mask,
                )

                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f'Unknown prediction type {noise_scheduler.config.prediction_type}')

                cross_attention_kwargs = {
                    'ref_hidden_states': retargeted_ref_tokens,
                    'ref_bboxes': ref_bboxes,
                    'ref_valid_mask': ref_valid_mask,
                }

                try:
                    controlnext_injector.set_controls(control)
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_tokens,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                finally:
                    controlnext_injector.clear()

                loss, _ = compute_masked_diffusion_mse_loss(
                    model_pred=model_pred,
                    target=target,
                    valid_mask=valid_mask,
                    noise_scheduler=noise_scheduler,
                    timesteps=timesteps,
                    snr_gamma=args.snr_gamma,
                )

                if args.lambda_x0 > 0 or args.lambda_bg > 0 or args.lambda_fg > 0 or args.lambda_grad > 0:
                    pred_x0 = compute_pred_x0(noise_scheduler, model_pred, noisy_latents, noise, timesteps)
                    vae_dtype = next(vae.parameters()).dtype
                    pred_x0_for_decode = pred_x0.to(device=accelerator.device, dtype=vae_dtype)
                    pred_img = vae.decode(pred_x0_for_decode / vae.config.scaling_factor).sample
                    pred_img_01 = denorm_line_tensor(pred_img)
                    gt_img_01 = denorm_line_tensor(pixel_values)
                    if args.lambda_x0 > 0:
                        loss = loss + args.lambda_x0 * torch.nn.functional.l1_loss(pred_img_01 * valid_mask, gt_img_01 * valid_mask)
                    if args.lambda_bg > 0:
                        loss = loss + args.lambda_bg * bg_clean_loss(pred_img_01, gt_img_01, valid_mask)
                    if args.lambda_fg > 0:
                        loss = loss + args.lambda_fg * fg_detail_loss(pred_img_01, gt_img_01, valid_mask)
                    if args.lambda_grad > 0:
                        loss = loss + args.lambda_grad * gradient_consistency_loss(pred_img_01, gt_img_01, valid_mask)

                avg_loss = accelerator.gather(loss.detach().unsqueeze(0)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    'train_loss': train_loss,
                    'lr': lr_scheduler.get_last_lr()[0],
                    'phase': {'phase_a_control_warmup': 0, 'phase_b_retarget_warmup': 1, 'phase_c_joint': 2}[current_phase],
                }, step=global_step)
                train_loss = 0.0

                if args.validation_every_n_steps > 0 and global_step % args.validation_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        run_validation(
                            accelerator=accelerator,
                            args=args,
                            vae=vae,
                            text_encoder=text_encoder,
                            unet=unet,
                            ref_adapter=ref_adapter,
                            local_sketch_query_encoder=local_sketch_query_encoder,
                            ref_retargeter=ref_retargeter,
                            controlnext_sketch=controlnext_sketch,
                            controlnext_injector=controlnext_injector,
                            tokenizer=tokenizer,
                            device=accelerator.device,
                            weight_dtype=weight_dtype,
                            fixed_val_examples=fixed_val_examples,
                            collator=collator,
                            global_step=global_step,
                        )
                    accelerator.wait_for_everyone()

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith('checkpoint-')]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing = checkpoints[:num_to_remove]
                                logger.info(f"Removing old checkpoints: {', '.join(removing)}")
                                for ckpt in removing:
                                    shutil.rmtree(os.path.join(args.output_dir, ckpt), ignore_errors=True)
                        save_path = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                        accelerator.save_state(save_path)
                        save_checkpoint_payload(
                            output_dir=save_path,
                            accelerator=accelerator,
                            unet=unet,
                            ref_adapter=ref_adapter,
                            local_sketch_query_encoder=local_sketch_query_encoder,
                            ref_retargeter=ref_retargeter,
                            controlnext_sketch=controlnext_sketch,
                        )
                        logger.info(f'Saved checkpoint to {save_path}')

            progress_bar.set_postfix(step_loss=float(loss.detach().item()), lr=float(lr_scheduler.get_last_lr()[0]), phase=current_phase)
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
            ref_adapter=ref_adapter,
            local_sketch_query_encoder=local_sketch_query_encoder,
            ref_retargeter=ref_retargeter,
            controlnext_sketch=controlnext_sketch,
        )
    controlnext_injector.remove()
    accelerator.end_training()


if __name__ == '__main__':
    main()
