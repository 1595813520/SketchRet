from __future__ import annotations

import os
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict


EXTRA_MODULES_NAME = 'extra_modules.pt'


def save_checkpoint_payload(
    output_dir: str,
    accelerator: Accelerator,
    unet,
    ref_adapter,
    local_sketch_query_encoder,
    ref_retargeter,
    controlnext_sketch,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    unwrapped_unet = accelerator.unwrap_model(unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )

    payload = {
        'ref_adapter_trainable': accelerator.unwrap_model(ref_adapter).trainable_state_dict(),
        'local_sketch_query_encoder': accelerator.unwrap_model(local_sketch_query_encoder).state_dict(),
        'ref_retargeter': accelerator.unwrap_model(ref_retargeter).state_dict(),
        'controlnext_sketch': accelerator.unwrap_model(controlnext_sketch).state_dict(),
    }

    proc_state: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, proc in unwrapped_unet.attn_processors.items():
        if isinstance(proc, torch.nn.Module):
            sd = proc.state_dict()
            if sd:
                proc_state[name] = sd
    payload['attn_processors'] = proc_state
    torch.save(payload, os.path.join(output_dir, EXTRA_MODULES_NAME))


def load_checkpoint_payload(
    input_dir: str,
    unet,
    ref_adapter,
    local_sketch_query_encoder,
    ref_retargeter,
    controlnext_sketch,
    mixed_precision: Optional[str],
) -> Dict[str, object]:
    lora_state_dict, _ = StableDiffusionPipeline.lora_state_dict(input_dir)
    unet_state_dict = {k.replace('unet.', ''): v for k, v in lora_state_dict.items() if k.startswith('unet.')}
    unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
    incompatible = set_peft_model_state_dict(unet, unet_state_dict, adapter_name='default')
    if incompatible is not None:
        unexpected = getattr(incompatible, 'unexpected_keys', None)
        if unexpected:
            print(f'[load_checkpoint_payload] Unexpected LoRA keys: {unexpected}')

    if mixed_precision == 'fp16':
        cast_training_params(unet, dtype=torch.float32)

    extra_path = os.path.join(input_dir, EXTRA_MODULES_NAME)
    if not os.path.exists(extra_path):
        print(f'[load_checkpoint_payload] {EXTRA_MODULES_NAME} not found in {input_dir}, skipping extra module load.')
        return {}

    extra = torch.load(extra_path, map_location='cpu')
    ref_adapter.load_trainable_state_dict(extra['ref_adapter_trainable'])
    local_sketch_query_encoder.load_state_dict(extra['local_sketch_query_encoder'], strict=True)
    ref_retargeter.load_state_dict(extra['ref_retargeter'], strict=True)
    controlnext_sketch.load_state_dict(extra['controlnext_sketch'], strict=True)

    if 'attn_processors' in extra:
        for name, sd in extra['attn_processors'].items():
            if name in unet.attn_processors and isinstance(unet.attn_processors[name], torch.nn.Module):
                unet.attn_processors[name].load_state_dict(sd, strict=False)
    return extra
