"""
Clifford-mode rotor operations using the full Cl(k,0,0) engine.

These operate by projecting d-dimensional embeddings into Cl(k,0,0),
performing rotor operations in the algebra, and projecting back.

Used by CFS (Clifford Flow on Sphere) variant only.
RHF uses the analytic rotor_utils.py which works in full d-dimensional space.
"""

import torch
import torch.nn.functional as F

from gaflowlm.clifford.engine import CliffordEngine


def clifford_slerp(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    alpha_t: torch.Tensor,
    engine: CliffordEngine,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SLERP via rotor sandwich in Cl(k,0,0).

    Projects vectors into Clifford algebra, computes rotor bivector from
    the outer product, and applies sandwich product.

    NOTE: This projects d-dimensional vectors to k-dimensional before
    rotor operations. Information beyond k dims is lost. For RHF,
    use the analytic rotor_slerp instead which works in full d-dim space.

    Args:
        clean: [..., d] unit vector.
        noisy: [..., d] unit vector.
        alpha_t: interpolation parameter.
        engine: CliffordEngine instance.
        eps: Small constant.

    Returns:
        [..., d] interpolated unit vector.
    """
    alpha_t = alpha_t.reshape(alpha_t.shape[0], *([1] * (clean.ndim - 1)))
    orig_dtype = clean.dtype
    d = clean.shape[-1]

    # Embed in Cl(k,0,0) — project to first k dims
    clean_mv = engine.embed_to_clifford(clean.to(engine._dtype))
    noisy_mv = engine.embed_to_clifford(noisy.to(engine._dtype))

    # Compute bivector: B = ½ clean ∧ noisy
    B = engine.bivector_from_vectors(clean_mv, noisy_mv)

    # Scale by interpolation parameter
    B_scaled = (1 - alpha_t.to(engine._dtype)) * B

    # Compute rotor and apply
    R = engine.rotor_exp(B_scaled, eps=eps)
    result_mv = engine.rotor_apply(R, noisy_mv)

    # Extract vector part and project back to d dimensions
    result = engine.clifford_to_embed(result_mv, d).to(orig_dtype)

    # Re-normalize (rotor sandwich preserves norm in exact arithmetic,
    # but we're working in a k-dimensional projection, so re-normalize)
    return F.normalize(result, dim=-1, eps=eps)


def clifford_log_map(
    x: torch.Tensor,
    target: torch.Tensor,
    engine: CliffordEngine,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log-map via bivector logarithm in Cl(k,0,0).

    NOTE: Falls back to analytic formula for the tangent vector,
    since the log-map is the same regardless of the algebra representation.
    The Clifford mode adds value when used with multivector embeddings
    (CFS variant), not for the log-map itself.

    Args:
        x: [..., d] current point on sphere.
        target: [..., d] target point on sphere.
        engine: CliffordEngine instance.
        eps: Small constant.

    Returns:
        [..., d] tangent vector at x.
    """
    cos_omega = (x * target).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cos_omega)
    scale = omega / omega.sin().clamp(min=eps)
    return scale * (target - x * cos_omega)


def clifford_exp_map(
    x: torch.Tensor,
    delta: torch.Tensor,
    engine: CliffordEngine,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Exp-map via rotor sandwich in Cl(k,0,0).

    Args:
        x: [..., d] current point on sphere.
        delta: [..., d] tangent vector at x.
        engine: CliffordEngine instance.
        eps: Small constant.

    Returns:
        [..., d] new point on sphere.
    """
    delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=eps)
    return (
        x * torch.cos(delta_norm)
        + (delta / delta_norm) * torch.sin(delta_norm)
    )