from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from accelerate import Accelerator
from diffusers import DDIMScheduler
from transformers import CLIPTextModel

from utils.image_ops import apply_valid_mask_white, denorm_line_tensor, postprocess_pred_to_pil


@torch.no_grad()
def save_validation_triplets(
    output_dir: str,
    global_step: int,
    sample_ids: List[str],
    sketch_batch: torch.Tensor,
    gt_batch: torch.Tensor,
    pred_batch: torch.Tensor,
    valid_mask_batch: torch.Tensor,
):
    import torchvision.transforms.functional as TF

    save_dir = os.path.join(output_dir, 'validation', f'step_{global_step:07d}')
    os.makedirs(save_dir, exist_ok=True)

    sketch_vis = sketch_batch.clamp(0, 1).repeat(1, 3, 1, 1)
    gt_vis = gt_batch.clamp(0, 1)
    pred_vis = pred_batch.clamp(0, 1)
    sketch_vis = apply_valid_mask_white(sketch_vis, valid_mask_batch)
    gt_vis = apply_valid_mask_white(gt_vis, valid_mask_batch)
    pred_vis = apply_valid_mask_white(pred_vis, valid_mask_batch)

    for i, sid in enumerate(sample_ids):
        triplet = torch.cat([sketch_vis[i], gt_vis[i], pred_vis[i]], dim=2)
        pil = TF.to_pil_image(triplet.cpu())
        safe_name = sid.replace('/', '__').replace('::', '__')
        pil.save(os.path.join(save_dir, f'{safe_name}_triplet.png'))


@torch.no_grad()
def save_unpadded_validation_preds(
    output_dir: str,
    global_step: int,
    sample_ids: List[str],
    pred_batch: torch.Tensor,
    panel_geoms: List[Dict[str, Any]],
):
    save_dir = os.path.join(output_dir, 'validation', f'step_{global_step:07d}', 'unpadded')
    os.makedirs(save_dir, exist_ok=True)
    for i, sid in enumerate(sample_ids):
        pred_i = pred_batch[i : i + 1]
        pil = postprocess_pred_to_pil(pred_i, panel_geoms[i], unpad_back=True)
        safe_name = sid.replace('/', '__').replace('::', '__')
        pil.save(os.path.join(save_dir, f'{safe_name}_pred.png'))


@torch.no_grad()
def run_validation(
    accelerator: Accelerator,
    args,
    vae,
    text_encoder: CLIPTextModel,
    unet,
    ref_adapter,
    local_sketch_query_encoder,
    ref_retargeter,
    controlnext_sketch,
    controlnext_injector,
    tokenizer,
    device: torch.device,
    weight_dtype: torch.dtype,
    fixed_val_examples: List[Dict[str, Any]],
    collator,
    global_step: int,
):
    del tokenizer
    if len(fixed_val_examples) == 0:
        return

    unet_eval = accelerator.unwrap_model(unet)
    ref_adapter_eval = accelerator.unwrap_model(ref_adapter)
    local_query_eval = accelerator.unwrap_model(local_sketch_query_encoder)
    ref_retargeter_eval = accelerator.unwrap_model(ref_retargeter)
    controlnext_eval = accelerator.unwrap_model(controlnext_sketch)

    unet_eval.eval()
    ref_adapter_eval.eval()
    local_query_eval.eval()
    ref_retargeter_eval.eval()
    controlnext_eval.eval()

    batch = collator(fixed_val_examples)
    pixel_values = batch['pixel_values'].to(device=device, dtype=weight_dtype)
    sketch_values = batch['sketch_values'].to(device=device, dtype=weight_dtype)
    valid_mask = batch['valid_mask'].to(device=device, dtype=weight_dtype)
    input_ids = batch['input_ids'].to(device=device)
    attention_mask = batch['attention_mask'].to(device=device)

    text_tokens = text_encoder(input_ids, attention_mask=attention_mask, return_dict=False)[0]
    ref_imgs = batch['ref_pixel_values'].to(device=device, dtype=weight_dtype)
    ref_bboxes = batch['ref_bboxes'].to(device=device)
    ref_valid_mask = batch['ref_valid_mask'].to(device=device)

    val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    val_scheduler.set_timesteps(args.validation_num_inference_steps, device=device)
    query_timestep = val_scheduler.timesteps[0].expand(sketch_values.shape[0])

    bsz, n_ref, c, hr, wr = ref_imgs.shape
    ref_tokens = ref_adapter_eval(ref_imgs.reshape(bsz * n_ref, c, hr, wr))
    ref_tokens = ref_tokens.reshape(bsz, n_ref, ref_tokens.shape[1], ref_tokens.shape[2])
    role_q = local_query_eval(
        sketch_values,
        valid_mask,
        ref_bboxes,
        ref_valid_mask,
        timesteps=query_timestep,
        num_train_timesteps=val_scheduler.config.num_train_timesteps,
    )
    retargeted_ref_tokens = ref_retargeter_eval(
        text_tokens=text_tokens,
        local_sketch_queries=role_q,
        ref_tokens=ref_tokens,
        text_attention_mask=attention_mask,
        role_valid_mask=ref_valid_mask,
    )

    control = controlnext_eval(torch.cat([sketch_values, valid_mask], dim=1), timestep=torch.zeros(bsz, device=device, dtype=torch.long))

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_h = pixel_values.shape[-2] // vae_scale_factor
    latent_w = pixel_values.shape[-1] // vae_scale_factor
    latents = torch.randn((bsz, unet_eval.config.in_channels, latent_h, latent_w), device=device, dtype=weight_dtype)
    cross_attention_kwargs = {
        'ref_hidden_states': retargeted_ref_tokens,
        'ref_bboxes': ref_bboxes,
        'ref_valid_mask': ref_valid_mask,
    }
    try:
        for t in val_scheduler.timesteps:
            controlnext_injector.set_controls(control)
            noise_pred = unet_eval(
                latents,
                t,
                encoder_hidden_states=text_tokens,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            controlnext_injector.clear()
            latents = val_scheduler.step(noise_pred, t, latents).prev_sample.to(dtype=weight_dtype)
    finally:
        controlnext_injector.clear()

    vae_dtype = next(vae.parameters()).dtype
    pred = vae.decode(latents.to(device=device, dtype=vae_dtype) / vae.config.scaling_factor).sample.float().cpu()
    gt = pixel_values.float().cpu()
    sk_raw = sketch_values.float().cpu()
    valid_mask_cpu = batch['valid_mask'].float().cpu()

    pred = denorm_line_tensor(pred)
    gt = denorm_line_tensor(gt)
    pred = apply_valid_mask_white(pred, valid_mask_cpu)
    gt = apply_valid_mask_white(gt, valid_mask_cpu)

    save_validation_triplets(
        output_dir=args.output_dir,
        global_step=global_step,
        sample_ids=batch['sample_ids'],
        sketch_batch=sk_raw,
        gt_batch=gt,
        pred_batch=pred,
        valid_mask_batch=valid_mask_cpu,
    )
    save_unpadded_validation_preds(
        output_dir=args.output_dir,
        global_step=global_step,
        sample_ids=batch['sample_ids'],
        pred_batch=pred,
        panel_geoms=batch['panel_geoms'],
    )
