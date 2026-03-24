# models/attention_processor.py
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedRefAttentionProcessor(nn.Module):
    """
    Global branch:
        original cross-attention over encoder_hidden_states = [text; sketch_sem_tokens]

    Local ref branch:
        per-character reference tokens + spatial mask
        only affects the corresponding query region

    Notes
    -----
    1) We do NOT directly add log(mask) to logits in code, because queries fully outside the mask
       can cause all -inf logits and lead to NaNs.
    2) Instead, we compute normal ref attention and multiply the output by normalized query weights.
    3) This processor is intended for cross-attention layers in UNet only.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        ref_cross_attention_dim: Optional[int] = None,
        num_ref_tokens: int = 8,
        use_learnable_scale: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.ref_cross_attention_dim = ref_cross_attention_dim or cross_attention_dim
        self.num_ref_tokens = num_ref_tokens

        # decoupled ref branch projections (IP-Adapter style idea)
        self.to_k_ref = nn.Linear(self.ref_cross_attention_dim, hidden_size, bias=False)
        self.to_v_ref = nn.Linear(self.ref_cross_attention_dim, hidden_size, bias=False)

        # start from zero so ref branch does not dominate early
        if use_learnable_scale:
            self.ref_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("ref_scale", torch.tensor(1.0), persistent=False)

    # -------------------------
    # helpers
    # -------------------------
    @staticmethod
    def _maybe_norm_encoder_hidden_states(attn, encoder_hidden_states):
        if encoder_hidden_states is not None and attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        return encoder_hidden_states

    @staticmethod
    def _reshape_heads_to_batch_dim(x: torch.Tensor, heads: int) -> torch.Tensor:
        # [B, N, C] -> [B, heads, N, d]
        b, n, c = x.shape
        d = c // heads
        return x.view(b, n, heads, d).transpose(1, 2)

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        # [B, heads, N, d] -> [B, N, C]
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d)

    @staticmethod
    def _build_mask_from_bboxes(
        bboxes: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        bboxes: [B, N_ref, 4], normalized xyxy in [0,1]
        returns mask: [B, N_ref, H, W], float {0,1}
        """
        # pixel centers in normalized coords
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        gy = ys.view(1, 1, h, 1)
        gx = xs.view(1, 1, 1, w)

        x0 = bboxes[..., 0].unsqueeze(-1).unsqueeze(-1)
        y0 = bboxes[..., 1].unsqueeze(-1).unsqueeze(-1)
        x1 = bboxes[..., 2].unsqueeze(-1).unsqueeze(-1)
        y1 = bboxes[..., 3].unsqueeze(-1).unsqueeze(-1)

        mask = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)
        return mask.to(dtype)

    @staticmethod
    def _resize_spatial_masks(
        spatial_masks: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        spatial_masks: [B, N_ref, H0, W0]
        returns:       [B, N_ref, h, w]
        """
        b, n, h0, w0 = spatial_masks.shape
        x = spatial_masks.reshape(b * n, 1, h0, w0)
        x = F.interpolate(x.float(), size=(h, w), mode="nearest")
        x = x.reshape(b, n, h, w)
        return x

    @staticmethod
    def _normalize_overlaps(
        role_masks: torch.Tensor,
        ref_valid_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        role_masks: [B, N_ref, H, W], float
        ref_valid_mask: [B, N_ref], bool/float, optional

        Returns normalized per-role weights [B, N_ref, H, W]
        so that overlapping regions do not explode in magnitude.
        """
        if ref_valid_mask is not None:
            role_masks = role_masks * ref_valid_mask[:, :, None, None].to(role_masks.dtype)

        denom = role_masks.sum(dim=1, keepdim=True).clamp(min=eps)
        # zero stays zero, overlapping areas become fractional weights
        normed = role_masks / denom
        normed = torch.where(role_masks > 0, normed, torch.zeros_like(normed))
        return normed

    def _prepare_role_query_weights(
        self,
        b: int,
        q_len: int,
        spatial_shape: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        ref_bboxes: Optional[torch.Tensor] = None,
        ref_spatial_masks: Optional[torch.Tensor] = None,
        ref_valid_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Returns:
            query_weights: [B, N_ref, Q]
        """
        h, w = spatial_shape
        assert h * w == q_len, f"spatial_shape {spatial_shape} incompatible with q_len={q_len}"

        if ref_spatial_masks is None and ref_bboxes is None:
            return None

        if ref_spatial_masks is not None:
            role_masks = self._resize_spatial_masks(ref_spatial_masks, h, w).to(device=device, dtype=dtype)
        else:
            role_masks = self._build_mask_from_bboxes(ref_bboxes.to(device=device, dtype=dtype), h, w, device, dtype)

        role_masks = self._normalize_overlaps(role_masks, ref_valid_mask=ref_valid_mask)
        query_weights = role_masks.reshape(b, role_masks.shape[1], q_len)  # [B, N_ref, Q]
        return query_weights

    # -------------------------
    # main
    # -------------------------
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        # custom kwargs for ref branch
        ref_hidden_states: Optional[torch.Tensor] = None,   # [B, N_ref, K_ref, D_ref]
        ref_bboxes: Optional[torch.Tensor] = None,          # [B, N_ref, 4], normalized xyxy
        ref_spatial_masks: Optional[torch.Tensor] = None,   # [B, N_ref, H0, W0], optional
        ref_valid_mask: Optional[torch.Tensor] = None,      # [B, N_ref], bool/float
        spatial_shape: Optional[Tuple[int, int]] = None,    # required if hidden_states is 3D
    ) -> torch.Tensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # ---- flatten 4D -> 3D if needed
        if input_ndim == 4:
            bsz, channels, h, w = hidden_states.shape
            hidden_states = hidden_states.view(bsz, channels, h * w).transpose(1, 2)
            current_spatial_shape = (h, w)
        else:
            bsz, q_len, _ = hidden_states.shape
            if spatial_shape is None:
                # best effort: infer square shape if possible
                side = int(math.sqrt(q_len))
                if side * side != q_len:
                    raise ValueError(
                        "spatial_shape must be provided for 3D hidden_states when q_len is not a perfect square."
                    )
                current_spatial_shape = (side, side)
            else:
                current_spatial_shape = spatial_shape

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # ---- queries
        query = attn.to_q(hidden_states)

        # ---- global branch: standard cross-attention over [text; sketch_sem_tokens]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        encoder_hidden_states = self._maybe_norm_encoder_hidden_states(attn, encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        heads = attn.heads
        head_dim = query.shape[-1] // heads

        query_h = self._reshape_heads_to_batch_dim(query, heads)
        key_h = self._reshape_heads_to_batch_dim(key, heads)
        value_h = self._reshape_heads_to_batch_dim(value, heads)

        # use attention_mask if provided, same as normal cross-attn
        attn_bias = None
        if attention_mask is not None:
            attn_bias = attn.prepare_attention_mask(
                attention_mask, target_length=key.shape[1], batch_size=batch_size, out_dim=4
            )  # [B, heads, Q, K]

        global_out = F.scaled_dot_product_attention(
            query_h, key_h, value_h,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
        )  # [B, heads, Q, d]

        # ---- local ref branch
        ref_out = torch.zeros_like(global_out)

        if ref_hidden_states is not None:
            # ref_hidden_states: [B, N_ref, K_ref, D_ref]
            if ref_hidden_states.ndim != 4:
                raise ValueError("ref_hidden_states must be [B, N_ref, K_ref, D_ref]")

            b, n_ref, k_ref, _ = ref_hidden_states.shape
            if b != batch_size:
                raise ValueError(f"Batch mismatch: hidden_states batch={batch_size}, ref batch={b}")

            query_weights = self._prepare_role_query_weights(
                b=batch_size,
                q_len=sequence_length,
                spatial_shape=current_spatial_shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                ref_bboxes=ref_bboxes,
                ref_spatial_masks=ref_spatial_masks,
                ref_valid_mask=ref_valid_mask,
            )

            if query_weights is not None:
                # [B, N_ref, Q] -> used to gate each role output on query positions
                pass

            for i in range(n_ref):
                if ref_valid_mask is not None:
                    # skip empty padded role slots
                    valid_i = ref_valid_mask[:, i].float()  # [B]
                    if valid_i.max().item() == 0:
                        continue
                else:
                    valid_i = None

                role_tokens = ref_hidden_states[:, i]  # [B, K_ref, D_ref]

                key_ref = self.to_k_ref(role_tokens)
                value_ref = self.to_v_ref(role_tokens)

                key_ref_h = self._reshape_heads_to_batch_dim(key_ref, heads)
                value_ref_h = self._reshape_heads_to_batch_dim(value_ref, heads)

                # manual attention for ref branch
                scores = torch.matmul(query_h, key_ref_h.transpose(-1, -2)) / math.sqrt(head_dim)
                probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
                role_out = torch.matmul(probs, value_ref_h)  # [B, heads, Q, d]

                if query_weights is not None:
                    # [B, Q] -> [B, 1, Q, 1]
                    qw = query_weights[:, i].unsqueeze(1).unsqueeze(-1).to(role_out.dtype)
                    role_out = role_out * qw

                if valid_i is not None:
                    # [B] -> [B,1,1,1]
                    role_valid = valid_i.view(batch_size, 1, 1, 1).to(role_out.dtype)
                    role_out = role_out * role_valid

                ref_out = ref_out + role_out

        # ---- merge global + ref branch
        scale = self.ref_scale.tanh() if isinstance(self.ref_scale, torch.Tensor) else self.ref_scale
        hidden_states = global_out + scale * ref_out

        # ---- merge heads back
        hidden_states = self._merge_heads(hidden_states)
        hidden_states = hidden_states.to(query.dtype)

        # attn output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # ---- restore 4D if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).reshape(bsz, channels, h, w)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states