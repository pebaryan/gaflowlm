"""
CARE — Clifford Algebra Rotary Encoding.

Rotor-based positional encoding for multivector sequences in Cl(k,0,0).

For each position i, we construct a rotor:
    R_i = exp(½ θ_x(i) B_x + ½ θ_y(i) B_y)

Where B_x, B_y are learned bivectors and θ_x(i), θ_y(i) are position-dependent
angles (sinusoidal by default, can be learned).

The position encoding is applied as a rotor sandwich:
    M̃_i = R_i · M_i · R̃_i

This preserves the multivector norm and the geometric structure, unlike
additive position embeddings which break the algebraic properties.
"""

import math
import torch
import torch.nn as nn


class CAREPositionEncoding(nn.Module):
    """CARE position encoding via rotor sandwich.

    Applies a position-dependent rotation to multivector tokens using
    learned bivectors and sinusoidal/complex-exponential angle functions.

    Args:
        k: Clifford dimension Cl(k,0,0). Multivector size = 2^k.
        max_len: Maximum sequence length.
        engine: CliffordEngine instance for geometric product ops.
        learned_angles: If True, theta_x/theta_y are learned per position.
                        If False (default), use sinusoidal angles.
        freq: Base frequency for sinusoidal angles (only when not learned).
    """

    def __init__(
        self,
        k: int,
        max_len: int = 2048,
        engine=None,
        learned_angles: bool = False,
        freq: float = 10000.0,
    ):
        super().__init__()
        self.k = k
        self.max_len = max_len
        self.engine = engine
        self.mv_dim = 1 << k  # 2^k

        if engine is not None:
            self.cayley = engine.cayley
            self.grade_masks = engine.grade_masks
            self._dtype = engine.cayley.dtype
            self._has_engine = True
        else:
            self._has_engine = False
            self._dtype = torch.float32

        # Learned bivectors: two independent bivectors in Cl(k,0,0)
        # Shape: [1, 1, mv_dim] for broadcasting with [B, L, mv_dim]
        biv_mask = self.grade_masks[2].squeeze(0) if engine is not None else None

        B_x = torch.zeros(1, 1, self.mv_dim, dtype=self._dtype)
        B_y = torch.zeros(1, 1, self.mv_dim, dtype=self._dtype)
        if biv_mask is not None:
            # Initialize with small random bivector components
            nn.init.normal_(B_x, mean=0.0, std=0.02)
            nn.init.normal_(B_y, mean=0.0, std=0.02)
            B_x = B_x * biv_mask
            B_y = B_y * biv_mask
        self.B_x = nn.Parameter(B_x)
        self.B_y = nn.Parameter(B_y)

        # Angle table: theta_x[i], theta_y[i] for each position
        dtype = self._dtype
        if learned_angles:
            theta = torch.zeros(1, max_len, 2, dtype=dtype)
            nn.init.uniform_(theta, -math.pi, math.pi)
            self.theta = nn.Parameter(theta)
            self._learned = True
        else:
            # Sinusoidal angles: geometric progression of frequencies
            freqs = torch.exp(-math.log(freq) * torch.arange(0, 2, dtype=dtype) / 2)
            pos = torch.arange(max_len, dtype=dtype).unsqueeze(1)  # [L, 1]
            theta = pos * freqs.unsqueeze(0)  # [L, 2]
            self.register_buffer('theta', theta.unsqueeze(0))  # [1, L, 2]
            self._learned = False

    def _build_rotor(self, theta_x: torch.Tensor, theta_y: torch.Tensor) -> torch.Tensor:
        """Build rotor R = exp(½(θ_x B_x + θ_y B_y)).

        Args:
            theta_x: [..., 1] angle for B_x
            theta_y: [..., 1] angle for B_y

        Returns:
            rotor: [..., mv_dim]
        """
        # Combine bivectors: ½(θ_x B_x + θ_y B_y)
        B = 0.5 * (theta_x * self.B_x + theta_y * self.B_y)

        # Compute B² scalar: ⟨B²⟩₀  [..., 1]
        BB = self._geometric_product(B, B)
        bb_scalar = BB[..., :1]

        # Rotor exponential with numerically safe gradient
        # R = cos(θ) + (B/θ) sin(θ)   where θ² = -⟨B²⟩₀
        # For B ≈ 0: R ≈ 1 + B (first-order Taylor)
        theta_sq = (-bb_scalar).clamp(min=0)
        theta = (theta_sq + 1e-16).sqrt()

        # Use Taylor expansion for small theta to avoid NaN gradients
        is_small = theta_sq < 1e-10

        # Safe computation for non-small theta
        theta_safe = theta + (theta < 1e-10).float() * 1e-10
        sin_over_theta = torch.sin(theta_safe) / theta_safe

        # For small theta: sin(θ)/θ ≈ 1 - θ²/6 ≈ 1
        # Use a smooth blend
        sin_over_theta = torch.where(
            is_small,
            torch.ones_like(theta),
            sin_over_theta,
        )

        cos_theta = torch.where(
            is_small,
            torch.ones_like(theta),
            torch.cos(theta_safe),
        )

        # R = cos(θ) + (sin(θ)/θ) * B
        scalar_part = cos_theta  # [..., 1]
        biv_part = sin_over_theta * B  # [..., mv_dim]

        # Combine: add scalar to first component
        rotor = torch.cat([scalar_part, biv_part[..., 1:]], dim=-1)
        return rotor

    def _geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Geometric product via Cayley tensor einsum.

        Args:
            A: [..., mv_dim]
            B: [..., mv_dim]

        Returns:
            [..., mv_dim]
        """
        if self._has_engine:
            return self.engine.geometric_product(A, B)
        # Fallback: direct einsum
        return torch.einsum('...i,...j,ijk->...k', A, B, self.cayley)

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """Apply CARE position encoding.

        Args:
            x: [B, L, mv_dim] multivector sequence.
            pos: [B, L] position indices (default: 0..L-1 arange).

        Returns:
            [B, L, mv_dim] position-encoded multivectors.
        """
        B, L, D = x.shape
        if pos is None:
            pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        # Get angles for positions
        if self._learned:
            # Learned angles: index into theta table
            theta = self.theta.expand(B, -1, -1)  # [B, L, 2]
            # Gather by position (clamp to max_len)
            pos_clamped = pos.clamp(0, self.max_len - 1)
            theta = torch.gather(
                theta, 1,
                pos_clamped.unsqueeze(-1).expand(-1, -1, 2)
            )
        else:
            # Sinusoidal: index into buffer
            pos_clamped = pos.clamp(0, self.max_len - 1).long()
            theta = self.theta.expand(B, -1, -1)  # [B, L, 2]
            theta = torch.gather(
                theta, 1,
                pos_clamped.unsqueeze(-1).expand(-1, -1, 2)
            )

        theta_x = theta[..., 0:1]  # [B, L, 1]
        theta_y = theta[..., 1:2]  # [B, L, 1]

        # Build rotor for each position
        R = self._build_rotor(theta_x, theta_y)  # [B, L, mv_dim]

        # Apply rotor sandwich: R · x · R̃
        # First: R · x
        Rx = self._geometric_product(R, x)
        # Compute reverse of R
        if self._has_engine:
            R_rev = self.engine.reverse_mv(R)
        else:
            # Fallback: negate grade-2 via direct mask
            biv_mask = self.grade_masks[2].squeeze(0)
            R_rev = R - 2 * R * biv_mask
        # R · x · R̃
        out = self._geometric_product(Rx, R_rev)

        return out
