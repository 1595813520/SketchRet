from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class IgnoreExtraKwargsAttnProcessor(nn.Module):
    """Wrapper for default diffusers attention processors that silently drops
    task-specific `cross_attention_kwargs` not consumed by self-attention paths.
    """

    def __init__(self, base_processor):
        super().__init__()
        self.base_processor = base_processor

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        ref_hidden_states: Optional[torch.Tensor] = None,
        ref_bboxes: Optional[torch.Tensor] = None,
        ref_valid_mask: Optional[torch.Tensor] = None,
        structure_gate: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del ref_hidden_states, ref_bboxes, ref_valid_mask, structure_gate, kwargs
        return self.forward(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            scale=scale,
        )

    def forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        import inspect

        call_target = getattr(self.base_processor, 'forward', None)
        if call_target is None:
            call_target = getattr(self.base_processor, '__call__')

        candidate_kwargs = {
            'encoder_hidden_states': encoder_hidden_states,
            'attention_mask': attention_mask,
            'temb': temb,
            'scale': scale,
        }

        try:
            sig = inspect.signature(call_target)
            has_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
            if has_var_kw:
                filtered = candidate_kwargs
            else:
                filtered = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
        except (TypeError, ValueError):
            filtered = candidate_kwargs

        return self.base_processor(attn, hidden_states, **filtered)


class MaskedRefAttentionProcessor(nn.Module):
    """Decoupled text / reference cross-attention.

    Text attention remains the standard cross-attention branch, but is explicitly
    weakened through a fixed scale so that renderer-time text guidance is always
    secondary to sketch control and retargeted reference detail guidance.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        ref_cross_attention_dim: int,
        ref_scale_init: float = 1.0,
        text_scale: float = 0.35,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.ref_cross_attention_dim = ref_cross_attention_dim
        self.to_k_ref = nn.Linear(ref_cross_attention_dim, hidden_size, bias=False)
        self.to_v_ref = nn.Linear(ref_cross_attention_dim, hidden_size, bias=False)
        self.ref_scale = nn.Parameter(torch.tensor(float(ref_scale_init)))
        self.text_scale = float(text_scale)

    @staticmethod
    def _infer_hw(query_length: int) -> Optional[Tuple[int, int]]:
        root = int(math.sqrt(query_length))
        if root * root == query_length:
            return root, root
        return None

    @staticmethod
    def _build_bbox_mask(
        boxes: torch.Tensor,
        valid_roles: torch.Tensor,
        query_len: int,
        tokens_per_role: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        hw = MaskedRefAttentionProcessor._infer_hw(query_len)
        if hw is None:
            return None
        h, w = hw
        ys = (torch.arange(h, device=device, dtype=boxes.dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=boxes.dtype) + 0.5) / w
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        gx = gx.reshape(1, 1, query_len)
        gy = gy.reshape(1, 1, query_len)

        x0 = boxes[..., 0:1].clamp(0.0, 1.0)
        y0 = boxes[..., 1:2].clamp(0.0, 1.0)
        x1 = boxes[..., 2:3].clamp(0.0, 1.0)
        y1 = boxes[..., 3:4].clamp(0.0, 1.0)
        inside = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
        inside = inside & valid_roles[..., None]
        fallback = valid_roles[..., None].expand_as(inside)
        any_hit = inside.any(dim=1, keepdim=True)
        inside = torch.where(any_hit, inside, fallback)
        inside = inside.permute(0, 2, 1).contiguous()
        inside = inside[:, :, :, None].expand(-1, -1, -1, tokens_per_role).reshape(boxes.shape[0], query_len, -1)
        return inside

    def _reference_attention(
        self,
        attn,
        query_states: torch.Tensor,
        ref_hidden_states: torch.Tensor,
        ref_bboxes: Optional[torch.Tensor],
        ref_valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, query_len, _ = query_states.shape
        b2, n_role, k_ref, d = ref_hidden_states.shape
        assert b == b2, 'Batch size mismatch between query and ref states.'
        ref_flat = ref_hidden_states.reshape(b, n_role * k_ref, d)

        q = attn.to_q(query_states)
        k = self.to_k_ref(ref_flat)
        v = self.to_v_ref(ref_flat)
        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        ref_attn_mask = None
        if ref_bboxes is not None and ref_valid_mask is not None:
            valid_roles = ref_valid_mask.to(dtype=torch.bool)
            bbox_valid = self._build_bbox_mask(ref_bboxes, valid_roles, query_len, k_ref, device=query_states.device)
            if bbox_valid is not None:
                heads = q.shape[0] // b
                ref_attn_mask = bbox_valid[:, None].expand(b, heads, query_len, n_role * k_ref)
                ref_attn_mask = ref_attn_mask.reshape(b * heads, query_len, n_role * k_ref)
                ref_attn_mask = torch.where(
                    ref_attn_mask,
                    torch.zeros_like(ref_attn_mask, dtype=q.dtype),
                    torch.full_like(ref_attn_mask, -1e4, dtype=q.dtype),
                )

        scores = attn.get_attention_scores(q, k, ref_attn_mask)
        out = torch.bmm(scores, v)
        out = attn.batch_to_head_dim(out)
        return out

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        ref_hidden_states: Optional[torch.Tensor] = None,
        ref_bboxes: Optional[torch.Tensor] = None,
        ref_valid_mask: Optional[torch.Tensor] = None,
        structure_gate: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del structure_gate, scale, kwargs
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        else:
            b = hidden_states.shape[0]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        text_attention_mask = None
        if attention_mask is not None:
            text_attention_mask = attn.prepare_attention_mask(attention_mask, encoder_hidden_states.shape[1], b)

        attention_probs = attn.get_attention_scores(query, key, text_attention_mask)
        hidden_states_out = torch.bmm(attention_probs, value)
        hidden_states_out = attn.batch_to_head_dim(hidden_states_out)
        hidden_states_out = self.text_scale * hidden_states_out

        if ref_hidden_states is not None and ref_hidden_states.numel() > 0:
            ref_out = self._reference_attention(
                attn=attn,
                query_states=hidden_states,
                ref_hidden_states=ref_hidden_states,
                ref_bboxes=ref_bboxes,
                ref_valid_mask=ref_valid_mask,
            )
            hidden_states_out = hidden_states_out + torch.tanh(self.ref_scale) * ref_out

        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)

        if input_ndim == 4:
            hidden_states_out = hidden_states_out.transpose(-1, -2).reshape(b, c, h, w)

        if attn.residual_connection:
            hidden_states_out = hidden_states_out + residual

        hidden_states_out = hidden_states_out / attn.rescale_output_factor
        return hidden_states_out
