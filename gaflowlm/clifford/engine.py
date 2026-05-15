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


def _extract_sparse_cayley(dense: torch.Tensor) -> tuple:
    """Extract (j, k, i, sign) triples from a dense Cayley-shaped tensor.

    Returns int64 index buffers plus a float sign buffer suitable for
    a scatter_add-based contraction:
        product[..., m] = A[..., j[m]] * B[..., k[m]] * sign[m]
        out[..., i[m]]  += product[..., m]
    """
    nz = dense.nonzero(as_tuple=False)  # [nnz, 3]
    j_idx = nz[:, 0].contiguous()
    k_idx = nz[:, 1].contiguous()
    i_idx = nz[:, 2].contiguous()
    signs = dense[j_idx, k_idx, i_idx].contiguous()
    return j_idx, k_idx, i_idx, signs


def build_wedge_cayley(
    cayley: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Wedge Cayley: keep entries where grade(i) == grade(j) + grade(k).

    For homogeneous A_r, B_s in Cl(k,0,0):
        A_r ∧ B_s = ⟨A_r B_s⟩_{r+s}
    Linear extension covers mixed-grade multivectors.
    """
    n = 1 << k
    grade = torch.tensor([bin(i).count('1') for i in range(n)])
    g_j = grade.view(n, 1, 1)
    g_k = grade.view(1, n, 1)
    g_i = grade.view(1, 1, n)
    mask = (g_i == g_j + g_k).to(cayley.dtype)
    return cayley * mask


def build_inner_cayley(
    cayley: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Inner Cayley: keep entries where grade(i) == |grade(j) - grade(k)|.

    For homogeneous A_r, B_s in Cl(k,0,0):
        A_r · B_s = ⟨A_r B_s⟩_{|r-s|}
    Linear extension covers mixed-grade multivectors.
    """
    n = 1 << k
    grade = torch.tensor([bin(i).count('1') for i in range(n)])
    g_j = grade.view(n, 1, 1)
    g_k = grade.view(1, n, 1)
    g_i = grade.view(1, 1, n)
    mask = (g_i == (g_j - g_k).abs()).to(cayley.dtype)
    return cayley * mask


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
        wedge_cayley = build_wedge_cayley(cayley, k)
        inner_cayley = build_inner_cayley(cayley, k)

        self.register_buffer('cayley', cayley)
        self.register_buffer('grade_masks', grade_masks)
        self.register_buffer('reverse_signs', reverse_signs)

        # Commonly used masks
        self.register_buffer('scalar_mask', grade_masks[0:1])      # [1, n]
        self.register_buffer('vector_mask', grade_masks[1:2])      # [1, n]
        self.register_buffer('bivector_mask', grade_masks[2:3])    # [1, n]
        self.register_buffer('pseudoscalar_mask', grade_masks[k:k+1])  # [1, n]

        # Sparse (j, k, i, sign) triples for geometric / wedge / inner products.
        # geometric_product, outer_product, inner_product all go through
        # _sparse_gp using these. Far cheaper than einsum over a dense [n,n,n]
        # tensor at k>=4.
        gp_j, gp_k, gp_i, gp_s = _extract_sparse_cayley(cayley)
        we_j, we_k, we_i, we_s = _extract_sparse_cayley(wedge_cayley)
        in_j, in_k, in_i, in_s = _extract_sparse_cayley(inner_cayley)
        self.register_buffer('_gp_j', gp_j)
        self.register_buffer('_gp_k', gp_k)
        self.register_buffer('_gp_i', gp_i)
        self.register_buffer('_gp_signs', gp_s)
        self.register_buffer('_wedge_j', we_j)
        self.register_buffer('_wedge_k', we_k)
        self.register_buffer('_wedge_i', we_i)
        self.register_buffer('_wedge_signs', we_s)
        self.register_buffer('_inner_j', in_j)
        self.register_buffer('_inner_k', in_k)
        self.register_buffer('_inner_i', in_i)
        self.register_buffer('_inner_signs', in_s)

        # Flat index buffers for grade-1 (vector) and grade-2 (bivector) basis
        # blades, used by projection layers. Grade-r blades sit at flat
        # indices whose bitmask has popcount r — *not* at the first
        # binomial(k, r) sequential slots. Earlier versions of this engine
        # used `mv[..., 1:k+1] = ...` for grade-1 which silently writes into
        # bivector slots once k ≥ 3.
        vec_idx = torch.tensor(
            [i for i in range(self.n) if bin(i).count('1') == 1],
            dtype=torch.long,
        )
        biv_idx = torch.tensor(
            [i for i in range(self.n) if bin(i).count('1') == 2],
            dtype=torch.long,
        )
        self.register_buffer('vector_indices', vec_idx)
        self.register_buffer('bivector_indices', biv_idx)

    # -------------------------------------------------------------------
    # Core algebra
    # -------------------------------------------------------------------

    def _sparse_product(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        j_idx: torch.Tensor,
        k_idx: torch.Tensor,
        i_idx: torch.Tensor,
        signs: torch.Tensor,
    ) -> torch.Tensor:
        """Generic sparse Cayley contraction shared by GP / wedge / inner.

        Given precomputed nonzero triples (j_idx, k_idx, i_idx, signs):
            out[..., i_idx[m]] += signs[m] * A[..., j_idx[m]] * B[..., k_idx[m]]
        """
        # Index-select keeps A and B's batch shape; the elementwise multiply
        # below broadcasts those shapes together. Size the output from the
        # broadcast result, not from A alone — A may be broadcasted against
        # a larger B (e.g. positional rotor vs token batch in CARE).
        products = A.index_select(-1, j_idx) * B.index_select(-1, k_idx) * signs
        out = products.new_zeros(*products.shape[:-1], self.n)
        return out.index_add_(-1, i_idx, products)

    def geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute geometric product AB in Cl(k,0,0).

        Uses precomputed sparse Cayley triples (~n² nonzeros for k≥1, vs the
        n³ cost of a dense einsum).

        Args:
            A: [..., n] multivector coefficients.
            B: [..., n] multivector coefficients.

        Returns:
            AB: [..., n] multivector coefficients.
        """
        return self._sparse_product(
            A, B, self._gp_j, self._gp_k, self._gp_i, self._gp_signs,
        )

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
        """Wedge product A ∧ B.

        For homogeneous A_r, B_s: A_r ∧ B_s = ⟨A_r B_s⟩_{r+s}.
        Extended bilinearly to multigrade inputs via the precomputed
        grade-(r+s)-filtered Cayley tensor.

        Note: (AB - BA)/2 equals A∧B only when both inputs are grade-1.
        For mixed grades this method is the correct definition; the
        commutator is exposed separately as `commutator`.
        """
        return self._sparse_product(
            A, B, self._wedge_j, self._wedge_k, self._wedge_i, self._wedge_signs,
        )

    def inner_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Inner product A · B (Hestenes/symmetric convention).

        For homogeneous A_r, B_s: A_r · B_s = ⟨A_r B_s⟩_{|r-s|}.
        Extended bilinearly to multigrade inputs via the precomputed
        grade-||r-s||-filtered Cayley tensor.
        """
        return self._sparse_product(
            A, B, self._inner_j, self._inner_k, self._inner_i, self._inner_signs,
        )

    def commutator(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Commutator [A, B] = AB - BA (Lie bracket).

        Distinct from the wedge product for non-vector inputs.
        """
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
        """Exponential of a *simple* bivector: R = exp(B).

        For a simple bivector B (i.e. B² ∈ ℝ) in Cl(k,0,0):
            R = cos(θ) + (B/θ) sin(θ)   where θ = √(-⟨B²⟩₀)

        If B ≈ 0 (small angle):
            R = 1 + B   (first-order approximation)

        WARNING: This closed form requires B to be *simple*, i.e. to lie
        in a single rotation plane (B∧B = 0). For Cl(k,0,0) with k≥4 a
        generic grade-2 element is a sum of independent simple bivectors
        and the formula above is wrong — a polar/Cartan decomposition
        into independent planes is needed. Bivectors produced by
        `bivector_from_vectors` (from two grade-1 inputs) are always
        simple, so the standard RHF / S-FLM training path is safe; CFS
        code that constructs general bivectors from network outputs is
        not. See https://en.wikipedia.org/wiki/Bivector#Higher_dimensions

        Args:
            B: [..., n] bivector (grade-2 multivector), assumed simple.
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
        """Logarithm of a *simple* rotor: B = log(R).

        For R = cos(θ) + (B/θ)sin(θ):
            B = (θ/sin(θ)) · ⟨R⟩₂

        WARNING: Only valid when R is the exponential of a simple bivector
        (single rotation plane). For Cl(k,0,0) with k≥4 a general rotor
        is a product of independent simple rotors and requires a polar
        decomposition; the formula above silently returns the wrong
        bivector in that case. Numerical precision also degrades near
        θ ≈ π (the acos branch); a polar-form log is more robust there.

        Args:
            R: [..., n] rotor (even-grade multivector), assumed simple.
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

        Writes the first min(d, k) components of x into the grade-1 blade
        slots (indices 1, 2, 4, 8, ..., 2^(k-1)). Other grades stay zero.

        Args:
            x: [..., d] vector tensor.

        Returns:
            [..., n] multivector with only grade-1 components nonzero.
        """
        d = x.shape[-1]
        n_fill = min(d, self.k)
        mv = x.new_zeros(*x.shape[:-1], self.n)
        idx = self.vector_indices[:n_fill]
        mv.index_copy_(-1, idx, x[..., :n_fill])
        return mv

    def clifford_to_embed(self, mv: torch.Tensor, d: int) -> torch.Tensor:
        """Extract a d-dim vector from the grade-1 slots of a multivector.

        Reads from the grade-1 blade indices (1, 2, 4, ..., 2^(k-1)).
        If d > k, the trailing (d - k) entries are zero.
        """
        n_fill = min(d, self.k)
        idx = self.vector_indices[:n_fill]
        head = mv.index_select(-1, idx)
        if d == n_fill:
            return head
        tail = mv.new_zeros(*mv.shape[:-1], d - n_fill)
        return torch.cat([head, tail], dim=-1)

    def _bivector_indices(self) -> list:
        """Flat indices of grade-2 basis blades (Python list, for back-compat).

        New code should prefer the `bivector_indices` buffer (a long tensor),
        which can be passed directly to index_select / index_copy_.
        """
        return self.bivector_indices.tolist()


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

        # Grade-1: scatter into grade-1 blade slots (indices 1, 2, 4, 8, ...).
        vec = self.vec_proj(x)  # [..., k]
        n_vec = min(engine.vector_indices.numel(), vec.shape[-1])
        mv.index_copy_(-1, engine.vector_indices[:n_vec], vec[..., :n_vec])

        # Grade-2: scatter into grade-2 blade slots.
        biv = self.biv_proj(x)  # [..., n_bvec]
        n_biv = min(engine.bivector_indices.numel(), biv.shape[-1])
        mv.index_copy_(-1, engine.bivector_indices[:n_biv], biv[..., :n_biv])

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
        # Grade-0: scalar
        g0 = mv[..., 0:1]  # [..., 1]

        # Grade-1: gather grade-1 blades into [..., k].
        g1 = mv.index_select(-1, engine.vector_indices)

        # Grade-2: gather grade-2 blades into [..., n_bvec]. Pad with zeros
        # if the projection layer is wider than the algebra provides.
        n_fill = min(engine.bivector_indices.numel(), self.n_bvec)
        if n_fill == self.n_bvec:
            g2 = mv.index_select(-1, engine.bivector_indices[:n_fill])
        else:
            g2 = mv.new_zeros(*mv.shape[:-1], self.n_bvec)
            g2[..., :n_fill] = mv.index_select(-1, engine.bivector_indices[:n_fill])

        return self.scalar_proj(g0) + self.vec_proj(g1) + self.biv_proj(g2)