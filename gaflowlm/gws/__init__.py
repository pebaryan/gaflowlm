"""Grade-Wise Geometric Scheduling (GWS) for Clifford neural networks.

Library surface only. Ablation / benchmark / diagnostic scripts live in
the top-level `experiments/gws/` directory and are not part of this
package.
"""

from .grade_decompose import (
    decompose_grad_by_grade,
    decompose_param_by_grade,
    grade_norms,
    identify_multivector_params,
)
from .rotor_schedule import (
    CosineSchedule,
    GradeRotorSchedule,
    LearnedRotorSchedule,
)

__all__ = [
    "decompose_grad_by_grade",
    "decompose_param_by_grade",
    "grade_norms",
    "identify_multivector_params",
    "CosineSchedule",
    "GradeRotorSchedule",
    "LearnedRotorSchedule",
]
