"""
Clifford Algebra Engine for Cl(k,0,0) — Euclidean signature.

Implements geometric product, outer/inner products, reverse, grade projection,
rotor exponential/logarithm/application, and spinor norm.

All operations are batched and @torch.compile compatible.

References:
- Dorst, Fontijne, Mann — "Geometric Algebra for Computer Science" (2007)
- Alesiani, Maruyama — "Clifford Flows" (NeurIPS 2024 ML4PS)
- Wagner et al. — "Generating Highly Designable Proteins with GAFL" (arXiv 2411.05238)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Cayley tensor construction for Cl(k,0,0)
# ---------------------------------------------------------------------------

def _basis_blade_index(k: int) -> dict:
    """Map bitmask of basis blades to flat index in 2^k multivector."""
    return {i: 1 << i for i in range(k)}


def build_cayley_tensor(
    k: int,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the 3-index Cayley tensor C_{ijk} for Cl(k,0,0).

    The geometric product of basis blades e_a * e_b = sign * e_c
    is encoded as C[a,b,c] = sign.

    For Cl(k,0,0): e_i^2 = +1 for all i (Euclidean signature).

    Args:
        k: Dimension of the Clifford algebra.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        cayley: [2^k, 2^k, 2^k] tensor. cayley[a,b,c] gives the coefficient
                of e_c in the product e_a * e_b.
    """
    n = 1 << k  # 2^k basis blades
    cayley = torch.zeros(n, n, n, device=device, dtype=dtype)

    for a in range(n):
        for b in range(n):
            a_bits = a
            b_bits = b

            # Count swaps for anti-commutation:
            # each pair (i in a, j in b) where i > j contributes a sign flip
            sign = 1
            a_ones = a_bits
            count = 0
            while a_ones:
                i = a_ones & (-a_ones)  # lowest set bit
                mask_below_i = i - 1
                count += bin(b_bits & mask_below_i).count('1')
                a_ones &= a_ones - 1  # clear lowest set bit
            if count % 2 == 1:
                sign = -1

            result = a_bits ^ b_bits  # XOR gives the result blade
            if result < n:
                cayley[a, b, result] += sign

    return cayley


def build_grade_masks(
    k: int,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build grade projection masks for Cl(k,0,0).

    Returns:
        grade_masks: [k+1, 2^k] binary mask. grade_masks[r] selects grade-r components.
    """
    n = 1 << k
    grade_masks = torch.zeros(k + 1, n, device=device, dtype=dtype)
    for i in range(n):
        grade = bin(i).count('1')
        grade_masks[grade, i] = 1.0
    return grade_masks


def build_reverse_signs(
    k: int,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build sign table for reverse operation.

    The reverse of a grade-r component gets sign (-1)^(r(r-1)/2).

    Returns:
        signs: [2^k] tensor of signs for each basis blade.
    """
    n = 1 << k
    signs = torch.ones(n, device=device, dtype=dtype)
    for i in range(n):
        grade = bin(i).count('1')
        if (grade * (grade - 1) // 2) % 2 == 1:
            signs[i] = -1.0
    return signs


# ---------------------------------------------------------------------------
# Clifford Algebra Operations
# ---------------------------------------------------------------------------

class CliffordEngine(nn.Module):
    """Geometric algebra engine for Cl(k,0,0).

    Precomputes and caches the Cayley tensor, grade masks, and reverse signs.
    All operations are @torch.compile compatible.

    Args:
        k: Dimension of the Clifford algebra. Default: 8.
           k=8 gives Cl(8,0,0) with 256 basis elements per multivector.
        device: Torch device.
        dtype: Computation dtype (use float64 for numerical stability in rotors).
    """

    def __init__(self, k: int = 8, device: str = 'cpu', dtype: torch.dtype = torch.float64):
        super().__init__()
        self.k = k
        self.n = 1 << k  # 2^k basis blades
        self._device = device
        self._dtype = dtype

        # Precompute and register as buffers (not parameters)
        cayley = build_cayley_tensor(k, device, dtype)
        grade_masks = build_grade_masks(k, device, dtype)
        reverse_signs = build_reverse_signs(k, device, dtype)

        self.register_buffer('cayley', cayley)
        self.register_buffer('grade_masks', grade_masks)
        self.register_buffer('reverse_signs', reverse_signs)

        # Commonly used masks
        self.register_buffer('scalar_mask', grade_masks[0:1])      # [1, n]
        self.register_buffer('vector_mask', grade_masks[1:2])      # [1, n]
        self.register_buffer('bivector_mask', grade_masks[2:3])    # [1, n]
        self.register_buffer('pseudoscalar_mask', grade_masks[k:k+1])  # [1, n]

    # -------------------------------------------------------------------
    # Core algebra
    # -------------------------------------------------------------------

    def geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute geometric product AB in Cl(k,0,0).

        Uses precomputed Cayley tensor: C_{ijk} where e_i * e_j = C_{ijk} e_k.

        Args:
            A: [..., n] multivector coefficients.
            B: [..., n] multivector coefficients.

        Returns:
            AB: [..., n] multivector coefficients.
        """
        return torch.einsum('...j,...k,jki->...i', A, B, self.cayley)

    def reverse_mv(self, M: torch.Tensor) -> torch.Tensor:
        """Reverse of multivector: M̃.

        Each grade-r component gets sign (-1)^(r(r-1)/2).

        Args:
            M: [..., n] multivector coefficients.

        Returns:
            M̃: [..., n] reversed multivector.
        """
        return M * self.reverse_signs

    def scalar_part(self, M: torch.Tensor) -> torch.Tensor:
        """Extract scalar (grade-0) component."""
        return M[..., 0:1]

    def grade_project(self, M: torch.Tensor, grade: int) -> torch.Tensor:
        """Project to specific grade.

        Args:
            M: [..., n] multivector.
            grade: Grade to project to (0=scalar, 1=vector, 2=bivector, etc.)

        Returns:
            [..., n] projected multivector.
        """
        return M * self.grade_masks[grade]

    def vector_part(self, M: torch.Tensor) -> torch.Tensor:
        """Extract vector (grade-1) components."""
        return M * self.vector_mask.squeeze(0)

    def bivector_part(self, M: torch.Tensor) -> torch.Tensor:
        """Extract bivector (grade-2) components."""
        return M * self.bivector_mask.squeeze(0)

    def outer_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Outer (wedge) product A ∧ B = (AB - BA) / 2."""
        AB = self.geometric_product(A, B)
        BA = self.geometric_product(B, A)
        return (AB - BA) / 2

    def inner_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Inner (dot) product A · B = (AB + BA) / 2."""
        AB = self.geometric_product(A, B)
        BA = self.geometric_product(B, A)
        return (AB + BA) / 2

    def commutator(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Commutator [A, B] = AB - BA (Lie bracket)."""
        AB = self.geometric_product(A, B)
        BA = self.geometric_product(B, A)
        return AB - BA

    def spinor_norm(self, M: torch.Tensor) -> torch.Tensor:
        """Compute spinor norm ⟨MM̃⟩₀."""
        Mrev = self.reverse_mv(M)
        MMrev = self.geometric_product(M, Mrev)
        return self.scalar_part(MMrev)

    def normalize_multivector(self, M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize multivector so ⟨MM̃⟩₀ = 1."""
        norm_sq = self.spinor_norm(M).clamp(min=eps)
        return M / norm_sq.sqrt()

    # -------------------------------------------------------------------
    # Rotor operations
    # -------------------------------------------------------------------

    def bivector_from_vectors(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute bivector encoding rotation from b to a.

        B = (1/2) a ∧ b

        Args:
            a: [..., n] vector (grade-1 multivector, other grades zero).
            b: [..., n] vector (grade-1 multivector, other grades zero).

        Returns:
            [..., n] bivector (grade-2 multivector).
        """
        return self.outer_product(a, b) / 2

    def rotor_exp(self, B: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Exponential of a bivector: R = exp(B).

        For a bivector B in Cl(k,0,0):
            R = cos(θ) + (B/θ) sin(θ)   where θ = √(-⟨B²⟩₀)

        If B ≈ 0 (small angle):
            R = 1 + B   (first-order approximation)

        Args:
            B: [..., n] bivector (grade-2 multivector).
            eps: Small constant for numerical stability.

        Returns:
            [..., n] rotor (even-grade multivector).
        """
        BB = self.geometric_product(B, B)
        bb_scalar = self.scalar_part(BB)  # [..., 1]

        # For Euclidean Cl(k,0,0): bivectors square to negative scalars
        # θ² = -⟨B²⟩₀
        theta_sq = (-bb_scalar).clamp(min=0)
        theta = theta_sq.sqrt()
        theta_safe = theta.clamp(min=eps)

        cos_theta = torch.cos(theta)
        sin_over_theta = torch.where(
            theta_sq > eps * eps,
            torch.sin(theta) / theta_safe,
            torch.ones_like(theta),
        )

        # R = cos(θ) + (sin(θ)/θ) * B
        result = cos_theta * self.scalar_mask.squeeze(0) + sin_over_theta * B
        return result

    def rotor_from_vectors(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Compute rotor R such that R b R̃ rotates b toward a by angle 2α.

        R = exp(α · (1/2)(a ∧ b)) for unit vectors a, b.

        When α=0: R=1 (identity, no rotation).
        When α=0.5: rotates b to a fully (SLERP equivalent at α=0.5).

        Args:
            a: [..., n] vector on sphere (target).
            b: [..., n] vector on sphere (source).
            alpha: [..., 1] interpolation parameter. Default: 0.5 (midpoint).
            eps: Small constant.

        Returns:
            [..., n] rotor R.
        """
        B = self.bivector_from_vectors(a, b)
        if alpha is not None:
            B = alpha * B
        return self.rotor_exp(B, eps=eps)

    def rotor_apply(self, R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply rotor R to multivector x via sandwich product: x' = R x R̃.

        Preserves norm, grade, and algebraic structure.

        Args:
            R: [..., n] rotor (even-grade multivector).
            x: [..., n] multivector to rotate.

        Returns:
            [..., n] rotated multivector R x R̃.
        """
        R_rev = self.reverse_mv(R)
        Rx = self.geometric_product(R, x)
        return self.geometric_product(Rx, R_rev)

    def bivector_log(self, R: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Logarithm of a rotor: B = log(R).

        For R = cos(θ) + (B/θ)sin(θ):
            B = (θ/sin(θ)) · ⟨R⟩₂

        Args:
            R: [..., n] rotor (even-grade multivector).
            eps: Small constant.

        Returns:
            [..., n] bivector B = log(R).
        """
        sp = self.scalar_part(R)
        bp = self.bivector_part(R)

        cos_theta = sp.clamp(-1 + eps, 1 - eps)
        theta = torch.acos(cos_theta)
        sin_theta = theta.sin().clamp(min=eps)

        scale = theta / sin_theta
        return scale * bp

    # -------------------------------------------------------------------
    # Embedding ↔ Clifford projections
    # -------------------------------------------------------------------

    def embed_to_clifford(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a d-dimensional vector into Cl(k,0,0) as a grade-1 multivector.

        Pads or truncates x to k dimensions and places in the vector slots
        of the multivector representation.

        Args:
            x: [..., d] vector tensor.

        Returns:
            [..., n] multivector with only grade-1 components nonzero.
        """
        d = x.shape[-1]
        batch_shape = x.shape[:-1]
        mv = torch.zeros(*batch_shape, self.n, device=x.device, dtype=x.dtype)

        if d >= self.k:
            mv[..., 1:self.k + 1] = x[..., :self.k]
        else:
            mv[..., 1:d + 1] = x[..., :d]

        return mv

    def clifford_to_embed(self, mv: torch.Tensor, d: int) -> torch.Tensor:
        """Extract d-dimensional vector from grade-1 components of multivector.

        Args:
            mv: [..., n] multivector.
            d: Output dimension.

        Returns:
            [..., d] vector tensor.
        """
        return mv[..., 1:d + 1]

    def _bivector_indices(self) -> list:
        """Get flat indices of grade-2 basis blades."""
        indices = []
        for i in range(self.n):
            if bin(i).count('1') == 2:
                indices.append(i)
        return indices


# ---------------------------------------------------------------------------
# Projection modules (nn.Module) for integration with neural networks
# ---------------------------------------------------------------------------

class EmbedToClifford(nn.Module):
    """Project d-dimensional embedding to Cl(k,0,0) multivector.

    Decomposes the embedding into:
    - Grade-0 (scalar): learned bias
    - Grade-1 (vector): linear projection d -> k
    - Grade-2 (bivector): linear projection d -> k*(k-1)/2

    Higher grades are set to zero for efficiency.
    """

    def __init__(self, d_embed: int, k: int = 8, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_embed = d_embed
        self.k = k
        self.n = 1 << k
        self.n_bvec = k * (k - 1) // 2

        self.scalar_bias = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.vec_proj = nn.Linear(d_embed, k, bias=False, dtype=dtype)
        self.biv_proj = nn.Linear(d_embed, self.n_bvec, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, engine: CliffordEngine) -> torch.Tensor:
        """Project embedding to multivector.

        Args:
            x: [..., d_embed] embedding.
            engine: CliffordEngine instance.

        Returns:
            [..., n] multivector.
        """
        batch_shape = x.shape[:-1]
        mv = torch.zeros(*batch_shape, engine.n, device=x.device, dtype=x.dtype)

        # Grade-0: learned scalar bias
        mv[..., 0] = self.scalar_bias.squeeze()

        # Grade-1: linear projection
        vec = self.vec_proj(x)  # [..., k]
        k = engine.k
        mv[..., 1:k + 1] = vec[..., :k]

        # Grade-2: bivector projection
        biv = self.biv_proj(x)  # [..., n_bvec]
        biv_indices = engine._bivector_indices()
        n_fill = min(len(biv_indices), biv.shape[-1])
        for i in range(n_fill):
            mv[..., biv_indices[i]] = biv[..., i]

        return mv


class CliffordToEmbed(nn.Module):
    """Project Cl(k,0,0) multivector back to d-dimensional embedding.

    Concatenates projections from grades 0, 1, and 2.
    """

    def __init__(self, k: int = 8, d_embed: int = 768, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.k = k
        self.n_bvec = k * (k - 1) // 2
        self.n = 1 << k
        self.d_embed = d_embed

        self.scalar_proj = nn.Linear(1, d_embed, bias=True, dtype=dtype)
        self.vec_proj = nn.Linear(k, d_embed, bias=False, dtype=dtype)
        self.biv_proj = nn.Linear(self.n_bvec, d_embed, bias=False, dtype=dtype)

    def forward(self, mv: torch.Tensor, engine: CliffordEngine) -> torch.Tensor:
        """Project multivector back to embedding.

        Args:
            mv: [..., n] multivector.
            engine: CliffordEngine instance.

        Returns:
            [..., d_embed] embedding.
        """
        k = engine.k

        # Grade-0: scalar
        g0 = mv[..., 0:1]  # [..., 1]

        # Grade-1: vector
        g1 = mv[..., 1:k + 1]  # [..., k]

        # Grade-2: bivector
        biv_indices = engine._bivector_indices()
        n_fill = min(len(biv_indices), self.n_bvec)
        g2 = torch.zeros(*mv.shape[:-1], self.n_bvec, device=mv.device, dtype=mv.dtype)
        for i in range(n_fill):
            g2[..., i] = mv[..., biv_indices[i]]

        return self.scalar_proj(g0) + self.vec_proj(g1) + self.biv_proj(g2)