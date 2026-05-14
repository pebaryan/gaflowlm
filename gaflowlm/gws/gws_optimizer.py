"""
GWS Optimizer — AdamW with per-grade learning rates for multivector parameters.

Wraps a standard AdamW optimizer and applies grade-wise LR scaling
from a GradeRotorSchedule to multivector parameters. Scalar parameters
(embeddings, biases, projection layers) use the base schedule.
"""

import math
import torch
from torch.optim import AdamW
from gaflowlm.clifford.engine import CliffordEngine
from .grade_decompose import identify_multivector_params
from .rotor_schedule import CosineSchedule, GradeRotorSchedule


class GWSAdamW:
    """AdamW with Grade-Wise Scheduling for Clifford neural networks.

    Instead of a single lr(t), multivector parameters get per-grade LRs
    from a rotor schedule. Non-multivector parameters use a base cosine
    schedule.

    This is a lightweight wrapper: it manages param groups internally
    and delegates actual optimization to AdamW.

    Args:
        model: The nn.Module to optimize.
        engine: CliffordEngine instance for grade decomposition.
        lr_max: Peak learning rate.
        T: Total training steps.
        k_s: Scheduling algebra dimension. 1 = reduces to cosine.
        eta_min: Minimum LR.
        warmup_steps: Warmup steps for both base and grade schedules.
        phase_offsets: Per-grade phase offsets (None = auto).
        weight_decay: AdamW weight decay.
        betas: AdamW betas.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        engine: CliffordEngine,
        lr_max: float = 1e-3,
        T: int = 5000,
        k_s: int = 2,
        eta_min: float = 0.0,
        warmup_steps: int = 0,
        phase_offsets: list[float] | None = None,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
    ):
        self.engine = engine
        self.lr_max = lr_max
        self.T = T

        # Identify which params are multivector-valued
        mv_map = identify_multivector_params(model, engine)
        self.mv_param_names = {name for name, is_mv in mv_map.items() if is_mv}

        # Build grade schedule
        n_grades = engine.k + 1
        self.grade_schedule = GradeRotorSchedule(
            k_s=k_s,
            n_grades=n_grades,
            T=T,
            eta_max=lr_max,
            eta_min=eta_min,
            phase_offsets=phase_offsets,
            warmup_steps=warmup_steps,
        )

        # Base schedule for non-mv params
        self.base_schedule = CosineSchedule(
            eta_max=lr_max, T=T, eta_min=eta_min,
        )

        # Create param groups: one base group + one per grade
        # We'll reassign LRs each step, so initial groups just need the params
        mv_params = []
        scalar_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.mv_param_names:
                mv_params.append(param)
            else:
                scalar_params.append(param)

        # Build param groups
        # Group 0: scalar params on base schedule
        # Groups 1..n_grades: grade-g components of mv params on grade schedule
        param_groups = [
            {'params': scalar_params, 'lr': lr_max, 'name': 'base'},
        ]
        for g in range(n_grades):
            # All mv params share grade groups — we'll scale gradients per-grade
            # in the step hook rather than splitting into separate param tensors
            param_groups.append({
                'params': mv_params,
                'lr': lr_max,
                'name': f'grade_{g}',
            })

        self.optimizer = AdamW(
            param_groups,
            lr=lr_max,
            weight_decay=weight_decay,
            betas=betas,
        )
        self._step_count = 0

        # Register a hook to apply per-grade scaling before the optimizer step
        # We do this by scaling the .grad tensors of mv params per-grade before
        # the optimizer sees them, then rescaling back after.
        self._mv_params = mv_params
        self._n_grades = n_grades
        self._grade_scales: list[float] | None = None

    def _compute_grade_scales(self) -> list[float]:
        """Compute per-grade LR ratios relative to the base LR.

        Returns a list of scale factors, one per grade.
        The optimizer step will multiply gradients by these scales.
        """
        grade_lrs = self.grade_schedule(self._step_count)
        base_lr = self.base_schedule(self._step_count)
        if base_lr < 1e-12:
            return [0.0] * self._n_grades
        return [glr / base_lr for glr in grade_lrs]

    def step(self, closure=None):
        """Perform one optimizer step with grade-wise LR scaling.

        Strategy: before the optimizer step, scale mv param gradients
        per-grade. After the step, undo the scaling so gradients remain
        correct for the next backward pass.

        Since all grade groups point to the same param tensors, we use
        a different approach: we split mv params into per-grade effective
        param groups by masking. But that requires separate param tensors.

        Simpler approach: use manual per-grade scaling.
        """
        # Get grade scales
        self._grade_scales = self._compute_grade_scales()

        # Apply per-grade scaling to mv param gradients
        # For each mv param, its grad is [..., n] multivector.
        # We scale each grade component by (grade_scale / sum_of_scales)
        # so that the effective LR per grade matches the schedule.
        #
        # Actually, the cleanest way: modify param group LRs directly.
        grade_lrs = self.grade_schedule(self._step_count)
        base_lr = self.base_schedule(self._step_count)

        for i, group in enumerate(self.optimizer.param_groups):
            if group['name'] == 'base':
                group['lr'] = base_lr
            elif group['name'].startswith('grade_'):
                g = int(group['name'].split('_')[1])
                group['lr'] = grade_lrs[g]

        # The problem: all grade groups share the same param tensors,
        # so AdamW would apply the LAST group's LR. We need a different approach.
        #
        # Correct approach: scale gradients per-grade before the step.
        # 1. Save original gradients
        # 2. Replace grad with grade-weighted version
        # 3. Step with uniform LR
        # 4. Restore original gradients

        # Save original grads and apply grade scaling
        saved_grads = []
        for p in self._mv_params:
            if p.grad is not None:
                saved_grads.append(p.grad.clone())
                # Scale each grade component of the gradient
                scaled_grad = p.grad.clone()
                for g in range(self._n_grades):
                    mask = self.engine.grade_masks[g]  # [n]
                    scale = self._grade_scales[g] if self._grade_scales else 1.0
                    # Scale the grade-g portion of the gradient
                    # grad is [..., n], mask is [n]
                    scaled_grad = scaled_grad * (1 - mask) + scaled_grad * mask * scale
                p.grad.copy_(scaled_grad)
            else:
                saved_grads.append(None)

        # Set all param groups to base_lr for the actual step
        for group in self.optimizer.param_groups:
            group['lr'] = base_lr

        # Take the optimizer step
        loss = self.optimizer.step(closure)

        # Restore original gradients
        for p, saved in zip(self._mv_params, saved_grads):
            if saved is not None:
                if p.grad is not None:
                    p.grad.copy_(saved)
                else:
                    p.grad = saved

        self._step_count += 1
        return loss

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def step_count(self):
        return self._step_count

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'step_count': self._step_count,
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self._step_count = state_dict['step_count']
