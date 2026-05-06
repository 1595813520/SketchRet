from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel


class ControlNeXtInjector:
    """Apply gated ControlNeXt injection with Cross Normalization."""

    def __init__(self, unet: UNet2DConditionModel):
        self.unet = unet
        self.current_controls: Optional[Dict[str, torch.Tensor]] = None
        self.handle = None

    def set_controls(self, controls: Optional[Dict[str, torch.Tensor]]) -> None:
        self.current_controls = controls

    def clear(self) -> None:
        self.current_controls = None

    @staticmethod
    def _cross_normalize(control: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        control_f = control.float()
        sample_f = sample.float()

        mean_latents = sample_f.mean(dim=(1, 2, 3), keepdim=True)
        std_latents = sample_f.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)

        mean_control = control_f.mean(dim=(1, 2, 3), keepdim=True)
        std_control = control_f.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)

        out = (control_f - mean_control) * (std_latents / std_control) + mean_latents
        return out.to(dtype=sample.dtype)

    def _hook(self, module, inputs, outputs):
        if self.current_controls is None:
            return outputs
        if not isinstance(outputs, tuple) or len(outputs) == 0:
            return outputs

        sample = outputs[0]
        control = self.current_controls['output']
        gate = self.current_controls.get('gate', None)
        scale = self.current_controls.get('scale', 1.0)
        if torch.is_tensor(scale):
            scale_value = float(scale.mean().item())
        else:
            scale_value = float(scale)

        control = F.adaptive_avg_pool2d(control, sample.shape[-2:]).to(device=sample.device, dtype=sample.dtype)
        control = self._cross_normalize(control, sample)

        if gate is not None:
            gate = F.adaptive_avg_pool2d(gate, sample.shape[-2:]).to(device=sample.device, dtype=sample.dtype)
            control = control * gate

        sample = sample + control * scale_value
        rest = outputs[1:]
        return (sample, *rest)

    def register(self) -> None:
        if self.handle is not None:
            return
        self.handle = self.unet.down_blocks[0].register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.clear()
