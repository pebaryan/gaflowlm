"""
Minimal mock of flash_attn for environments without CUDA flash attention.

Provides rotary embedding functions and attention fallbacks used by
S-FLM's DiT backbone. Flash attention kernels are replaced with
PyTorch's scaled_dot_product_attention.
"""
import torch
import torch.nn.functional as F
import math


def _apply_rotary_single(x, cos_half, sin_half):
    """Apply rotary embedding to x.

    Args:
        x: [B, L, H, D] or [L, H, D] or [L, D]
        cos_half: [L, D//2]
        sin_half: [L, D//2]
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    # cos_half is [L, D//2]. x1 has leading dims [..., L, ..., D//2].
    # Insert dimensions so cos_half broadcasts correctly:
    #   For x [B, L, H, D//2]: need cos as [1, L, 1, D//2]
    #   For x [L, H, D//2]:      need cos as [L, 1, D//2]
    #   For x [L, D//2]:         need cos as [L, D//2] (no change)
    # Pattern: for each dim before the L dim in x, insert a singleton.
    # L is always at dim[-(n_leading+1)] where n_leading = ndim-2 for
    # 4D, ndim-2 for 3D, etc. Actually L is always dim 1 for >=3D,
    # dim 0 for 2D.
    if x1.ndim == 4:
        # x1: [B, L, H, D//2], cos: [L, D//2] -> [1, L, 1, D//2]
        cos_half = cos_half.unsqueeze(0).unsqueeze(2)
        sin_half = sin_half.unsqueeze(0).unsqueeze(2)
    elif x1.ndim == 3:
        # x1: [L, H, D//2], cos: [L, D//2] -> [L, 1, D//2]
        cos_half = cos_half.unsqueeze(1)
        sin_half = sin_half.unsqueeze(1)
    # else: 2D, no change needed

    out1 = x1 * cos_half - x2 * sin_half
    out2 = x2 * cos_half + x1 * sin_half
    return torch.cat([out1, out2], dim=-1)


def apply_rotary_emb_torch(q, cos, sin):
    """Apply rotary embedding to query or key tensor.

    Args:
        q: [..., head_dim]
        cos: [..., head_dim // 2] (broadcastable)
        sin: [..., head_dim // 2]
    """
    return _apply_rotary_single(q, cos, sin)


def apply_rotary_emb_qkv_(qkv, cos, sin):
    """Apply rotary embedding to qkv tensor in-place.

    Args:
        qkv: [batch, seqlen, 3, heads, head_dim]
        cos: [seqlen, head_dim // 2]
        sin: [seqlen, head_dim // 2]
    """
    q = qkv[:, :, 0]  # [B, L, H, D]
    k = qkv[:, :, 1]  # [B, L, H, D]

    qkv[:, :, 0] = _apply_rotary_single(q, cos, sin)
    qkv[:, :, 1] = _apply_rotary_single(k, cos, sin)
    return qkv


def flash_attn_func(q, k, v, causal=False, softmax_scale=None):
    """Drop-in for flash_attn.flash_attn_func using PyTorch SDPA."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if q.ndim == 4:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=softmax_scale)
        return out.transpose(1, 2)
    else:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=softmax_scale)


def flash_attn_qkvpacked_func(qkv, causal=False, softmax_scale=None):
    """Drop-in for flash_attn.flash_attn_qkvpacked_func."""
    q = qkv[:, :, 0].transpose(1, 2)
    k = qkv[:, :, 1].transpose(1, 2)
    v = qkv[:, :, 2].transpose(1, 2)
    out = flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
    return out.transpose(1, 2)


class layers:
    """Namespace for flash_attn.layers compatibility."""
    class rotary:
        apply_rotary_emb_torch = staticmethod(apply_rotary_emb_torch)
        apply_rotary_emb_qkv_ = staticmethod(apply_rotary_emb_qkv_)