"""
Clifford algebra subpackage.

Provides:
- engine: Core Cl(k,0,0) operations, CliffordEngine, embedding projections
"""

from .engine import (
    CliffordEngine,
    EmbedToClifford,
    CliffordToEmbed,
    build_cayley_tensor,
    build_grade_masks,
    build_reverse_signs,
)

__all__ = [
    'CliffordEngine',
    'EmbedToClifford',
    'CliffordToEmbed',
    'build_cayley_tensor',
    'build_grade_masks',
    'build_reverse_signs',
]