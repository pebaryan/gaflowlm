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
    """Clifford Frame Attention layer.

    Args:
        mv_dim: Multivector dimension (2^k for Cl(k,0,0)).
        n_heads: Number of attention heads.
        engine: CliffordEngine instance for geometric product ops.
        bilinear: If True (default), use geometric bilinear Q * V_agg.
                  If False, fall back to standard attention output.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        mv_dim: int,
        n_heads: int = 8,
        engine=None,
        bilinear: bool = True,
        dropout: float = 0.0,
        use_higher_order: bool = False,  # Legacy alias
    ):
        super().__init__()
        self.mv_dim = mv_dim
        self.n_heads = n_heads
        self.head_dim = mv_dim
        self.hidden_dim = n_heads * mv_dim
        self.engine = engine
        self.bilinear = bilinear or use_higher_order
        self.dropout = dropout

        # Dtype from engine
        dt = engine.cayley.dtype if engine is not None else torch.float32

        # Multivector linear projections for Q, K, V
        self.W_q = nn.Linear(mv_dim, self.hidden_dim, bias=False, dtype=dt)
        self.W_k = nn.Linear(mv_dim, self.hidden_dim, bias=False, dtype=dt)
        self.W_v = nn.Linear(mv_dim, self.hidden_dim, bias=False, dtype=dt)

        # Output projection: concat heads -> mv_dim
        self.W_o = nn.Linear(self.hidden_dim, mv_dim, bias=False, dtype=dt)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Precompute grade signs for attention scores
        # For Cl(k,0,0): <e_a * e_b>_0 is nonzero only when a==b,
        # with sign = (-1)^(r*(r-1)/2) where r = grade(a).
        # These are the reverse signs from the engine.
        if self.engine is not None:
            self.register_buffer('_grade_signs', self.engine.reverse_signs.squeeze(-1))
        else:
            self.register_buffer('_grade_signs', torch.ones(mv_dim, dtype=dt))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """CFA forward pass.

        Args:
            x: [B, L, mv_dim] multivector sequence.
            mask: [B, L, L] attention mask (True = allowed).

        Returns:
            [B, L, mv_dim] after geometric bilinear attention.
        """
        B, L, _ = x.shape

        # Linear projections: [B, L, hidden_dim]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape to heads: [B*H, L, head_dim]
        Q_h = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2).reshape(
            B * self.n_heads, L, self.head_dim
        )
        K_h = K.view(B, L, self.n_heads, self.head_dim).transpose(1, 2).reshape(
            B * self.n_heads, L, self.head_dim
        )
        V_h = V.view(B, L, self.n_heads, self.head_dim).transpose(1, 2).reshape(
            B * self.n_heads, L, self.head_dim
        )

        # --- Attention scores: grade-weighted dot product ---
        # Equivalent to scalar part of geometric product in Cl(k,0,0)
        K_w = K_h * self._grade_signs.to(K_h.device)
        attn = torch.matmul(Q_h, K_w.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1).reshape(-1, L, L)
            attn = attn.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # --- Aggregate values ---
        V_agg = torch.matmul(attn_weights, V_h)  # [B*H, L, head_dim]

        # --- Geometric bilinear output ---
        if self.bilinear:
            # Geometric bilinear: Q_i * V_agg_i
            # Encodes both inner (grade-lowering) and outer (grade-raising) products
            Q_flat = Q_h.reshape(-1, self.head_dim)
            V_flat = V_agg.reshape(-1, self.head_dim)
            gp = self.engine.geometric_product(Q_flat, V_flat)  # [B*H*L, head_dim]
            bilinear = gp.view(B * self.n_heads, L, self.head_dim)
            # Residual: bilinear + V_agg (lets model recover standard attention)
            out = bilinear + V_agg
        else:
            # Standard attention fallback
            out = V_agg

        # Reshape back: [B, L, hidden_dim] -> [B, L, mv_dim]
        out = out.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2).reshape(
            B, L, self.hidden_dim
        )
        out = self.W_o(out)

        return out


class CFSTransformerBlock(nn.Module):
    """CFS Transformer block: CFA + FFN with pre-norm residual.

    Args:
        mv_dim: Multivector dimension (2^k).
        n_heads: Number of attention heads.
        ff_dim: Feed-forward hidden dimension.
        engine: CliffordEngine instance.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        mv_dim: int,
        n_heads: int = 8,
        ff_dim: int = 1024,
        engine=None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mv_dim = mv_dim
        self.hidden_dim = mv_dim * n_heads

        dt = engine.cayley.dtype if engine is not None else torch.float32

        # CFA layer (bilinear=True by default)
        self.attention = CliffordFrameAttention(
            mv_dim=mv_dim,
            n_heads=n_heads,
            engine=engine,
            bilinear=True,
            dropout=dropout,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(mv_dim, dtype=dt)
        self.norm2 = nn.LayerNorm(mv_dim, dtype=dt)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(mv_dim, ff_dim, dtype=dt),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, mv_dim, dtype=dt),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """CFS block forward.

        Args:
            x: [B, L, mv_dim] multivector sequence.
            mask: [B, L, L] attention mask.

        Returns:
            [B, L, mv_dim] processed multivector sequence.
        """
        # Pre-norm + CFA
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask=mask)
        x = residual + x

        # Pre-norm + FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x
