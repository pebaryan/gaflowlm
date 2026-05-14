"""
Clifford Frame Attention (CFA) for Cl(k,0,0) multivector sequences.

Replaces standard dot-product attention with geometric bilinear attention:
- Q, K, V are multivector linear projections of the input multivector
- Attention scores: scalar part of geometric product via grade-weighted dot product
- Output: geometric bilinear Q_i * V_agg_i (geometric product of query and aggregated value)

The geometric bilinear is the DEFAULT behavior. It encodes both inner (grade-lowering)
and outer (grade-raising) products between query and value, enabling geometric message
passing that standard attention cannot replicate.

Reference:
    Wagner et al. "Generating Highly Designable Proteins with Geometric Algebra
    Flow Matching" (GAFL), arXiv 2411.05238, Section 3.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CliffordFrameAttention(nn.Module):
    """Clifford Frame Attention layer."""

    def __init__(
        self,
        mv_dim: int,
        n_heads: int = 8,
        engine=None,
        bilinear: bool = True,
        dropout: float = 0.0,
        use_higher_order: bool = False,
    ):
        super().__init__()
        self.mv_dim = mv_dim
        self.n_heads = n_heads
        self.head_dim = mv_dim
        self.hidden_dim = n_heads * mv_dim
        self.engine = engine
        self.bilinear = bilinear or use_higher_order
        self.use_higher_order = use_higher_order
        self.dropout = dropout

        dt = engine.cayley.dtype if engine is not None else torch.float32

        self.W_q = nn.Linear(mv_dim, self.hidden_dim, bias=False, dtype=dt)
        self.W_k = nn.Linear(mv_dim, self.hidden_dim, bias=False, dtype=dt)
        self.W_v = nn.Linear(mv_dim, self.hidden_dim, bias=False, dtype=dt)
        self.W_o = nn.Linear(self.hidden_dim, mv_dim, bias=False, dtype=dt)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        if self.engine is not None:
            self.register_buffer("_grade_signs", self.engine.reverse_signs.squeeze(-1))
        else:
            self.register_buffer("_grade_signs", torch.ones(mv_dim, dtype=dt))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """CFA forward pass."""
        B, L, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q_h = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2).reshape(
            B * self.n_heads, L, self.head_dim
        )
        K_h = K.view(B, L, self.n_heads, self.head_dim).transpose(1, 2).reshape(
            B * self.n_heads, L, self.head_dim
        )
        V_h = V.view(B, L, self.n_heads, self.head_dim).transpose(1, 2).reshape(
            B * self.n_heads, L, self.head_dim
        )

        K_w = K_h * self._grade_signs.to(K_h.device)
        attn = torch.matmul(Q_h, K_w.transpose(-2, -1)) * self.scale

        row_valid = None
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1).reshape(-1, L, L)
            row_valid = mask.any(dim=-1, keepdim=True)
            attn = attn.masked_fill(~mask, -1e9)
            attn = torch.where(row_valid, attn, torch.zeros_like(attn))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        V_agg = torch.matmul(attn_weights, V_h)

        if self.bilinear:
            Q_flat = Q_h.reshape(-1, self.head_dim)
            V_flat = V_agg.reshape(-1, self.head_dim)
            gp = self.engine.geometric_product(Q_flat, V_flat)
            bilinear = gp.view(B * self.n_heads, L, self.head_dim)
            out = bilinear + V_agg

            if self.use_higher_order:
                hk = self.engine.geometric_product(Q_flat, K_h.reshape(-1, self.head_dim))
                higher_order = hk.view(B * self.n_heads, L, self.head_dim)
                out = out + 0.25 * higher_order
        else:
            out = V_agg

        if row_valid is not None:
            out = out * row_valid.to(dtype=out.dtype)

        out = out.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2).reshape(
            B, L, self.hidden_dim
        )
        out = self.W_o(out)

        return out


class CFSTransformerBlock(nn.Module):
    """CFS Transformer block: CFA + FFN with pre-norm residual."""

    def __init__(
        self,
        mv_dim: int,
        n_heads: int = 8,
        ff_dim: int = 1024,
        engine=None,
        dropout: float = 0.0,
        use_higher_order: bool = False,
    ):
        super().__init__()
        self.mv_dim = mv_dim
        self.hidden_dim = mv_dim * n_heads

        dt = engine.cayley.dtype if engine is not None else torch.float32

        self.attention = CliffordFrameAttention(
            mv_dim=mv_dim,
            n_heads=n_heads,
            engine=engine,
            bilinear=True,
            dropout=dropout,
            use_higher_order=use_higher_order,
        )

        self.norm1 = nn.LayerNorm(mv_dim, dtype=dt)
        self.norm2 = nn.LayerNorm(mv_dim, dtype=dt)

        self.ffn = nn.Sequential(
            nn.Linear(mv_dim, ff_dim, dtype=dt),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, mv_dim, dtype=dt),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask=mask)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x
