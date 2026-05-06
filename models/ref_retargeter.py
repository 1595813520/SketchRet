from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class StateRetargetBlock(nn.Module):
    """State-first retargeting block.

    Local sketch queries act as the primary state query. They first retrieve
    detail evidence from reference tokens and then absorb text conditioning.
    The resulting state tokens are written back into the reference token set.
    """

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)

        self.norm_state_q1 = nn.LayerNorm(dim)
        self.norm_ref_kv = nn.LayerNorm(dim)
        self.attn_state_ref = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.norm_state_q2 = nn.LayerNorm(dim)
        self.norm_text_kv = nn.LayerNorm(dim)
        self.attn_state_text = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.state_ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

        self.norm_ref_q = nn.LayerNorm(dim)
        self.norm_state_kv = nn.LayerNorm(dim)
        self.attn_ref_state = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ref_ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        state_tokens: torch.Tensor,
        ref_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = state_tokens + self.attn_state_ref(
            self.norm_state_q1(state_tokens),
            self.norm_ref_kv(ref_tokens),
            self.norm_ref_kv(ref_tokens),
            need_weights=False,
        )[0]
        state = state + self.attn_state_text(
            self.norm_state_q2(state),
            self.norm_text_kv(text_tokens),
            self.norm_text_kv(text_tokens),
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )[0]
        state = state + self.state_ff(state)

        ref = ref_tokens + self.attn_ref_state(
            self.norm_ref_q(ref_tokens),
            self.norm_state_kv(state),
            self.norm_state_kv(state),
            need_weights=False,
        )[0]
        ref = ref + self.ref_ff(ref)
        return ref, state


class RefTokenRetargeter(nn.Module):
    """Rewrite reference tokens into state-compatible reference tokens."""

    def __init__(self, dim: int, depth: int = 2, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            StateRetargetBlock(dim=dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(dim)
        self.delta_gate = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        text_tokens: torch.Tensor,
        local_sketch_queries: torch.Tensor,
        ref_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        role_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n_role, k_ref, dim = ref_tokens.shape
        if n_role == 0:
            return ref_tokens

        ref = ref_tokens.reshape(b * n_role, k_ref, dim)
        state = local_sketch_queries.reshape(b * n_role, local_sketch_queries.shape[2], dim)
        text = text_tokens[:, None].expand(b, n_role, text_tokens.shape[1], dim).reshape(b * n_role, text_tokens.shape[1], dim)

        text_mask = None
        if text_attention_mask is not None:
            text_mask = ~text_attention_mask[:, None].expand(b, n_role, text_attention_mask.shape[1]).reshape(b * n_role, text_attention_mask.shape[1]).bool()

        out_ref = ref
        out_state = state
        for block in self.blocks:
            out_ref, out_state = block(out_state, out_ref, text, text_key_padding_mask=text_mask)

        out_ref = self.out_norm(out_ref)
        out_ref = ref + torch.tanh(self.delta_gate) * (out_ref - ref)
        out_ref = out_ref.reshape(b, n_role, k_ref, dim)
        if role_valid_mask is not None:
            out_ref = out_ref * role_valid_mask[:, :, None, None].to(out_ref.dtype)
        return out_ref
