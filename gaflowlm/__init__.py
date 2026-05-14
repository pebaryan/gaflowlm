"""
GAFlowLM: Geometric Algebra extensions for Hyperspherical Flow Language Models.

Provides:
- clifford.engine: Core Cl(k,0,0) algebra (Cayley tensor, geometric product,
  rotors, grade projection, embedding projections)
- clifford.rotor_ops: Clifford-mode rotor operations (CFS variant)
- rotor_utils: Analytic rotor-based replacements for S-FLM's sphere operations
  (rotor_slerp, rotor_log_map, rotor_exp_map, bivector_velocity)
- rhf_algo: RHF (Rotor Hyperspherical Flow) algorithm — SFM subclass with
  rotor-based sphere ops
"""

from gaflowlm.clifford.engine import (
    CliffordEngine,
    EmbedToClifford,
    CliffordToEmbed,
    build_cayley_tensor,
    build_grade_masks,
    build_reverse_signs,
)
from gaflowlm.rotor_utils import (
    rotor_slerp,
    rotor_log_map,
    rotor_exp_map,
    RotorOps,
    bivector_velocity,
    sphere_normalize,
)

__all__ = [
    'CliffordEngine',
    'EmbedToClifford',
    'CliffordToEmbed',
    'build_cayley_tensor',
    'build_grade_masks',
    'build_reverse_signs',
    'rotor_slerp',
    'rotor_log_map',
    'rotor_exp_map',
    'RotorOps',
    'bivector_velocity',
    'sphere_normalize',
]