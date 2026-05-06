from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import Downsample2D, ResnetBlock2D


class ControlNeXtSketchModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        time_embed_dim: int = 256,
        in_channels: tuple[int, int] = (128, 128),
        out_channels: tuple[int, int] = (128, 256),
        groups: tuple[int, int] = (4, 8),
        controlnext_scale: float = 0.25,
        cond_channels: int = 2,
    ):
        super().__init__()
        self.time_proj = Timesteps(128, True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(128, time_embed_dim)
        self.embedding = nn.Sequential(
            nn.Conv2d(cond_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, 128),
            nn.ReLU(),
        )
        self.down_res = nn.ModuleList([])
        self.down_sample = nn.ModuleList([])
        for i in range(len(in_channels)):
            self.down_res.append(
                ResnetBlock2D(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    temb_channels=time_embed_dim,
                    groups=groups[i],
                )
            )
            self.down_sample.append(
                Downsample2D(
                    out_channels[i],
                    use_conv=True,
                    out_channels=out_channels[i],
                    padding=1,
                    name='op',
                )
            )
        self.mid_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels[-1], out_channels[-1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.GroupNorm(8, out_channels[-1]),
                nn.Conv2d(out_channels[-1], out_channels[-1], kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, out_channels[-1]),
            ),
            nn.Conv2d(out_channels[-1], 320, kernel_size=1, stride=1),
        ])
        self.gate_head = nn.Sequential(
            nn.Conv2d(out_channels[-1], out_channels[-1] // 2, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels[-1] // 2, 1, kernel_size=1, stride=1),
        )
        self.scale = controlnext_scale

    def _module_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        p = next(self.embedding.parameters())
        return p.device, p.dtype

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        global_sketch_queries: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del global_sketch_queries  # kept only for interface compatibility
        module_device, module_dtype = self._module_device_dtype()
        sample = sample.to(device=module_device, dtype=module_dtype)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == 'mps'
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        batch_size = sample.shape[0]
        timesteps = timesteps.expand(batch_size)
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.embedding(sample)
        for res, downsample in zip(self.down_res, self.down_sample):
            sample = res(sample, emb)
            sample = downsample(sample, emb)

        sample = self.mid_convs[0](sample) + sample
        control = self.mid_convs[1](sample)
        gate = torch.sigmoid(self.gate_head(sample))
        return {
            'output': control,
            'gate': gate,
            'scale': torch.tensor(self.scale, device=sample.device, dtype=sample.dtype),
        }
