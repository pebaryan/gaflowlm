"""
Grade decomposition and analysis for multivector parameters.

Splits a flat parameter tensor into per-grade components using
the Clifford engine's grade masks, and provides utilities for
per-grade gradient norm tracking.
"""

import torch
from gaflowlm.clifford.engine import CliffordEngine


def decompose_param_by_grade(
    param: torch.Tensor,
    engine: CliffordEngine,
) -> dict[int, torch.Tensor]:
    """Decompose a multivector parameter into per-grade views.

    Assumes the last dimension of param has size engine.n (2^k),
    i.e. it stores full multivector coefficients.

    Args:
        param: [..., n] multivector parameter.
        engine: CliffordEngine with precomputed grade masks.

    Returns:
        Dict mapping grade -> [..., n] view with only that grade nonzero.
    """
    result = {}
    for grade in range(engine.k + 1):
        mask = engine.grade_masks[grade]  # [n]
        result[grade] = param * mask
    return result


def decompose_grad_by_grade(
    grad: torch.Tensor,
    engine: CliffordEngine,
) -> dict[int, torch.Tensor]:
    """Same as decompose_param_by_grade but for gradients."""
    return decompose_param_by_grade(grad, engine)


def grade_norms(
    grad: torch.Tensor,
    engine: CliffordEngine,
) -> dict[int, float]:
    """Compute L2 norm of each grade component of a gradient tensor.

    Args:
        grad: [..., n] multivector gradient.
        engine: CliffordEngine instance.

    Returns:
        Dict mapping grade -> scalar L2 norm.
    """
    norms = {}
    for grade in range(engine.k + 1):
        mask = engine.grade_masks[grade]  # [n]
        grade_grad = grad * mask
        norms[grade] = grade_grad.norm().item()
    return norms


def identify_multivector_params(
    model: torch.nn.Module,
    engine: CliffordEngine,
) -> dict[str, bool]:
    """Identify which parameters are multivector-valued (dim[-1] == engine.n).

    This lets the scheduler apply grade-wise LR only to multivector params,
    leaving scalar params (biases, embeddings, etc.) on the base schedule.

    Args:
        model: The neural network.
        engine: CliffordEngine instance.

    Returns:
        Dict mapping param name -> is_multivector.
    """
    result = {}
    for name, param in model.named_parameters():
        result[name] = param.shape[-1] == engine.n
    return result
