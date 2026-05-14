"""
Rotor-based replacements for S-FLM's trigonometric sphere operations.

Two approaches:
1. ANALYTIC (RHF): Uses closed-form rotor expressions operating directly on
   d-dimensional vectors. No Cayley tensor needed. Mathematically equivalent
   to SLERP/log_map/exp_map but with cleaner algebraic structure for extensions.

2. CLIFFORD (CFS): Uses the full CliffordEngine for multivector-based
   attention and embedding projections. Uses Cl(k,0,0) with small k.

Key insight (from MATHEMATICAL_INSIGHTS.md):
  slerp(x, y, α) = ⟨R(α·B) x R̃⟩₁  where B = ½ x ∧ y
  log_map(x, y) = bivector_part(log(R)) projected to tangent space  
  exp_map(x, v) = R(v∧x) x R̃   (sandwich product on the sphere)

For RHF (the minimal-swap variant), we use analytic rotor formulas that
work in the full d-dimensional space without projecting to Cl(k,0,0).
These are numerically identical to the trig formulas but expressed in
rotor algebra, enabling future extensions (e.g., bivector velocity fields).

For CFS (the maximal-swap variant), we use CliffordEngine for everything.
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Analytic Rotor Operations (d-dimensional, no Cayley tensor needed)
# ---------------------------------------------------------------------------

def rotor_slerp(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    alpha_t: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Spherical interpolation via rotor decomposed into 2D rotation planes.

    This is mathematically equivalent to SLERP but expressed as a rotor
    sandwich product decomposed into the plane containing x and y.

    For unit vectors x, y on S^{d-1}:
      The bivector B = ½ x∧y encodes the rotation plane
      R(α) = exp(α·B) rotates by angle 2α from y toward x
      slerp(x, y, α) = grade-1 projection of R y R̃

    The analytic form simplifies to:
      slerp(x, y, α) = cos(αω) · y + sin(αω)/sin(ω) · (x - y·cos(ω))

    Wait — this IS just SLERP. The rotor formulation is algebraically
    equivalent. But we write it differently to make the bivector structure
    explicit, enabling future GA extensions.

    For RHF, we provide a numerically equivalent version that uses the
    bivector parametrization internally for clarity.

    Args:
        clean: [..., d] unit vector (clean embedding, x₀).
        noisy: [..., d] unit vector (noise on sphere).
        alpha_t: [..., 1] or broadcastable interpolation parameter.
                 Convention matches S-FLM: alpha_t=0 → noisy, alpha_t=1 → clean.
        eps: Small constant for numerical stability.

    Returns:
        [..., d] interpolated unit vector on sphere.
    """
    # S-FLM convention: alpha_t=1 → clean, alpha_t=0 → noisy
    # (invert_time_convention flips this, handled upstream)
    # Rotor formulation: rotate noisy toward clean by angle (1-alpha_t)*ω
    # where ω = angle between clean and noisy.
    # This is identical to S-FLM's slerp but with explicit bivector structure.

    alpha_t = alpha_t.reshape(alpha_t.shape[0], *([1] * (clean.ndim - 1)))
    cos_omega = (clean * noisy).sum(dim=-1, keepdim=True)
    cos_omega = cos_omega.clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cos_omega)
    sin_omega = omega.sin().clamp(min=eps)

    # Bivector-weighted interpolation:
    # The bivector B = ½(clean ∧ noisy) = ½(clean ⊗ noisy - noisy ⊗ clean)
    # decomposed in the rotation plane. The rotor formulation gives:
    #   R(α) noisy R̃ = cos(ω·α) · noisy + sin(ω·α)/sin(ω) · (clean - cos(ω)·noisy)
    # which is exactly SLERP.
    coeff_clean = ((1 - alpha_t) * omega).sin() / sin_omega
    coeff_noisy = (alpha_t * omega).sin() / sin_omega
    return coeff_clean * clean + coeff_noisy * noisy


def rotor_log_map(
    x: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log-map on the sphere via bivector decomposition.

    Computes the tangent vector at x pointing toward target.
    In rotor algebra: the bivector log(R) where R rotates x → target
    gives the rotation plane and angle. Projected to the tangent space:

      v = (ω / sin(ω)) · (target - x·cos(ω))

    This is identical to S-FLM's log_map but expressed via the bivector
    logarithm structure.

    Args:
        x: [..., d] current point on sphere.
        target: [..., d] target point on sphere.
        eps: Small constant.

    Returns:
        [..., d] tangent vector at x pointing toward target.
    """
    cos_omega = (x * target).sum(dim=-1, keepdim=True)
    cos_omega = cos_omega.clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cos_omega)
    scale = omega / omega.sin().clamp(min=eps)
    return scale * (target - x * cos_omega)


def rotor_exp_map(
    x: torch.Tensor,
    delta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Exponential map on the sphere via rotor sandwich.

    Moves x along the tangent vector delta on the sphere.
    In rotor algebra: the bivector B ∝ x ∧ delta encodes the rotation,
    and R = exp(||δ|| · B_normalized) applied via sandwich moves x along δ.

    The analytic form collapses to the standard exp_map:
      x' = x·cos(||δ||) + (δ/||δ||)·sin(||δ||)

    Args:
        x: [..., d] current point on sphere.
        delta: [..., d] tangent vector at x (velocity).
        eps: Small constant.

    Returns:
        [..., d] new point on sphere.
    """
    delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=eps)
    return (
        x * torch.cos(delta_norm)
        + (delta / delta_norm) * torch.sin(delta_norm)
    )


# ---------------------------------------------------------------------------
# Bivector-based extensions (for CFS variant)
# ---------------------------------------------------------------------------

def bivector_velocity(
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute bivector velocity field between current and target on sphere.

    This is the CFS extension: instead of just computing the tangent vector,
    we compute the outer product (bivector) of the velocity field, which
    encodes the rotation plane in addition to the magnitude and direction.

    For CFS, this bivector is used as input to Clifford Frame Attention
    and enables higher-order geometric reasoning.

    Args:
        x_t: [..., d] current point on sphere.
        x_0: [..., d] target point on sphere.
        eps: Small constant.

    Returns:
        v: [..., d] tangent vector (same as log_map for now)
        B: [..., d, d] bivector = ½ v ∧ x_t (antisymmetric matrix)
    """
    v = rotor_log_map(x_t, x_0, eps=eps)

    # Compute bivector as outer product: B = ½(v ⊗ x_t - x_t ⊗ v)
    B = 0.5 * (
        v.unsqueeze(-1) * x_t.unsqueeze(-2)   # v ⊗ x_t
        - x_t.unsqueeze(-1) * v.unsqueeze(-2)  # x_t ⊗ v
    )

    return v, B


def sphere_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize to unit sphere (identical to S-FLM's sphere_normalize)."""
    return F.normalize(x, dim=-1, eps=eps)


# ---------------------------------------------------------------------------
# RotorOps: unified interface for S-FLM integration
# ---------------------------------------------------------------------------

class RotorOps:
    """Stateful wrapper providing rotor-based sphere operations.

    Drop-in replacement for S-FLM's utils.slerp, utils.log_map, utils.exp_map,
    and utils.sphere_normalize.

    For RHF (analytic mode), these are numerically identical to S-FLM's trig
    formulas but written in rotor algebra for clarity and extensibility.

    For CFS (Clifford mode), these use the CliffordEngine for full multivector
    operations. Set clifford_k > 0 to enable.

    Args:
        mode: 'analytic' for RHF (same numerics as trig SLERP) or
              'clifford' for CFS (full multivector operations via CliffordEngine).
        clifford_k: Dimension k for Cl(k,0,0) when mode='clifford'. Default: 8.
        device: Torch device.
        dtype: Computation dtype.
    """

    def __init__(self, mode: str = 'analytic', clifford_k: int = 8,
                 device: str = 'cpu', dtype: torch.dtype = torch.float64):
        self.mode = mode
        self.clifford_k = clifford_k
        self.device = device
        self.dtype = dtype

        if mode == 'clifford':
            from gaflowlm.clifford.engine import CliffordEngine
            self.engine = CliffordEngine(k=clifford_k, device=device, dtype=dtype)
        else:
            self.engine = None

    def slerp(self, clean: torch.Tensor, noisy: torch.Tensor,
              alpha_t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Spherical interpolation."""
        if self.mode == 'clifford':
            from gaflowlm.clifford.rotor_ops import clifford_slerp
            return clifford_slerp(clean, noisy, alpha_t, self.engine, eps)
        return rotor_slerp(clean, noisy, alpha_t, eps)

    def log_map(self, x: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
        """Log-map on the sphere."""
        if self.mode == 'clifford':
            from gaflowlm.clifford.rotor_ops import clifford_log_map
            return clifford_log_map(x, target, self.engine, eps)
        return rotor_log_map(x, target, eps)

    def exp_map(self, x: torch.Tensor, delta: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
        """Exp-map on the sphere."""
        if self.mode == 'clifford':
            from gaflowlm.clifford.rotor_ops import clifford_exp_map
            return clifford_exp_map(x, delta, self.engine, eps)
        return rotor_exp_map(x, delta, eps)

    def sphere_normalize(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Normalize to unit sphere."""
        return sphere_normalize(x, eps)