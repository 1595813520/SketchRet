from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import CLIPVisionModel

from .resampler import PerceiverResampler


class FrozenCLIPRefAdapter(nn.Module):
    """
    Frozen CLIP vision backbone with a trainable projector + resampler.
    Only trainable layers are saved in checkpoints.
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
    ) -> None:
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(model_name_or_path)
        self.vision_encoder.requires_grad_(False)
        self.vision_encoder.eval()

        hidden = self.vision_encoder.config.hidden_size
        self.proj = nn.Linear(hidden, cross_attn_dim)
        self.resampler = PerceiverResampler(
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

    def trainable_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            'proj': self.proj.state_dict(),
            'resampler': self.resampler.state_dict(),
        }

    def load_trainable_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        self.proj.load_state_dict(state['proj'], strict=True)
        self.resampler.load_state_dict(state['resampler'], strict=True)
