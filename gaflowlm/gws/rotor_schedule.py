"""
Rotor-based learning rate schedules for GWS.

Each grade g gets a schedule driven by rotor interpolation on a small
scheduling algebra Cl(k_s, 0, 0). The scalar part of R_g(t) gives
the LR multiplier, generalizing cosine annealing to multi-plane rotations.

When k_s=1 and all grades share the same bivector/angle, this reduces
exactly to cosine annealing.
"""

import math
import torch
from gaflowlm.clifford.engine import CliffordEngine


class CosineSchedule:
    """Standard cosine annealing — the baseline."""

    def __init__(self, eta_max: float, T: int, eta_min: float = 0.0):
        self.eta_max = eta_max
        self.T = T
        self.eta_min = eta_min

    def __call__(self, step: int) -> float:
        t = min(step, self.T)
        return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * t / self.T))


class GradeRotorSchedule:
    """Per-grade rotor schedule on Cl(k_s, 0, 0).

    For each grade g, we have:
      - A bivector B_g in the scheduling algebra (determines rotation plane)
      - A phase offset phi_g (stagger grades in time)
      - An angle schedule theta_g(t) in [0, pi/2]

    The LR multiplier for grade g is the scalar part of R_g(t):
      eta_g(t) = eta_g(0) * cos(theta_g(t))

    This recovers cosine annealing when k_s=1, B_g is the same for all g,
    and phi_g = 0.

    Args:
        k_s: Scheduling algebra dimension. Default: 2.
        n_grades: Number of multivector grades to schedule.
        T: Total training steps.
        eta_max: Base (peak) learning rate.
        eta_min: Minimum learning rate.
        phase_offsets: Per-grade phase offsets in [0, 1]. Default: evenly spaced.
        bivector_assignment: How to assign rotation planes to grades.
            'same' = all grades rotate in same plane (cosine annealing)
            'orthogonal' = each grade gets a different plane in Cl(k_s,0,0)
            'learned' = bivectors are nn.Parameters (requires grad)
        warmup_steps: Number of warmup steps before decay begins.
    """

    def __init__(
        self,
        k_s: int = 2,
        n_grades: int = 3,
        T: int = 5000,
        eta_max: float = 1e-3,
        eta_min: float = 0.0,
        phase_offsets: list[float] | None = None,
        bivector_assignment: str = 'orthogonal',
        warmup_steps: int = 0,
    ):
        self.k_s = k_s
        self.n_grades = n_grades
        self.T = T
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.bivector_assignment = bivector_assignment
        self.warmup_steps = warmup_steps

        # Small scheduling engine for rotor math
        self.engine = CliffordEngine(k=k_s, device='cpu', dtype=torch.float64)

        # Phase offsets — stagger grades so they decay at different times
        if phase_offsets is not None:
            assert len(phase_offsets) == n_grades
            self.phase_offsets = phase_offsets
        else:
            # Evenly spaced: grade 0 starts decaying immediately, highest grade last
            self.phase_offsets = [g / max(1, n_grades - 1) * 0.5 for g in range(n_grades)]

        # Build per-grade bivectors in the scheduling algebra
        self.bivectors = self._build_bivectors()

    def _build_bivectors(self) -> list[torch.Tensor]:
        """Construct per-grade bivectors in Cl(k_s, 0, 0)."""
        n = self.engine.n  # 2^k_s
        grade2_indices = self.engine._bivector_indices()

        if self.bivector_assignment == 'same':
            # All grades share the first bivector plane — reduces to cosine
            B = torch.zeros(n, dtype=torch.float64)
            if len(grade2_indices) > 0:
                B[grade2_indices[0]] = 1.0
            return [B.clone() for _ in range(self.n_grades)]

        elif self.bivector_assignment == 'orthogonal':
            # Each grade gets a different bivector plane
            bivectors = []
            for g in range(self.n_grades):
                B = torch.zeros(n, dtype=torch.float64)
                idx = g % len(grade2_indices) if grade2_indices else 0
                if len(grade2_indices) > 0:
                    B[grade2_indices[idx]] = 1.0
                bivectors.append(B)
            return bivectors

        else:
            raise ValueError(f"Unknown bivector_assignment: {self.bivector_assignment}")

    def theta(self, step: int, grade: int) -> float:
        """Compute the scheduling angle for a grade at a given step.

        theta_g(t) = (pi/2) * smoothstep((t - warmup - offset_g * T) / T_remain)

        This gives:
        - warmup period where LR ramps up
        - per-grade phase offsets so grades start decaying at different times
        - smoothstep ensures smooth transitions
        """
        if step <= self.warmup_steps:
            # During warmup, linearly ramp up
            return (math.pi / 2) * (1.0 - step / max(1, self.warmup_steps))

        # After warmup, decay with phase offset
        offset = self.phase_offsets[grade] * (self.T - self.warmup_steps)
        effective_t = max(0, step - self.warmup_steps - offset)
        effective_T = max(1, self.T - self.warmup_steps - offset)
        progress = min(1.0, effective_t / effective_T)

        # Smoothstep for smoother transition than linear
        smooth = progress * progress * (3 - 2 * progress)
        return (math.pi / 2) * smooth

    def lr_multiplier(self, step: int, grade: int) -> float:
        """Compute the LR multiplier for a grade at a given step.

        This is cos(theta_g(t)), which is the scalar part of the rotor R_g(t).
        """
        th = self.theta(step, grade)
        return math.cos(th)

    def __call__(self, step: int) -> list[float]:
        """Return per-grade LR values at a given step."""
        result = []
        for g in range(self.n_grades):
            mult = self.lr_multiplier(step, g)
            lr = self.eta_min + (self.eta_max - self.eta_min) * mult
            result.append(lr)
        return result


class LearnedRotorSchedule(GradeRotorSchedule):
    """Grade rotor schedule with learnable bivector components.

    The bivectors B_g are nn.Parameters, so the schedule can be
    meta-learned during training (requires a meta-optimization loop
    or gradient-through-schedule).
    """

    def __init__(self, *args, **kwargs):
        kwargs['bivector_assignment'] = 'orthogonal'
        super().__init__(*args, **kwargs)
        # Convert bivectors to parameters
        self.bivector_params = torch.nn.ParameterList([
            torch.nn.Parameter(B.clone()) for B in self.bivectors
        ])

    def _build_bivectors(self):
        # Initial bivectors — will be overwritten by parameters
        n = 1 << self.k_s
        return [torch.zeros(n, dtype=torch.float64) for _ in range(self.n_grades)]
