"""
Grade-Wise Scheduler (GWS).

GWS is an auxiliary optimization track for deep multivector models. It keeps
the base learning-rate schedule simple and stable, then adds small per-grade
phase shifts on top. The result is most useful when a flow objective trains a
deep Clifford model and different grades converge at different rates.

The implementation stays intentionally conservative:
- base schedule: CosineAnnealingLR
- per-grade modulation: cosine phase offsets
- application: scale multivector gradients before the optimizer step
- optional adaptation: tiny heuristic updates to phase offsets when enabled
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR


def _as_phase_tensor(
    phase_offsets,
    num_grades: int,
    phase_stagger: bool,
    phase_step: float,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    if phase_offsets is not None:
        offsets = torch.as_tensor(phase_offsets, dtype=dtype, device=device)
    elif phase_stagger:
        offsets = torch.arange(num_grades, dtype=dtype, device=device) * phase_step
    else:
        offsets = torch.zeros(num_grades, dtype=dtype, device=device)

    if offsets.ndim != 1:
        raise ValueError("phase_offsets must be a 1D sequence")
    if offsets.numel() != num_grades:
        raise ValueError(
            f"phase_offsets length {offsets.numel()} must match num_grades {num_grades}"
        )
    return offsets


def _broadcast_scale(scale: torch.Tensor, grad: torch.Tensor, axis: int) -> torch.Tensor:
    view = [1] * grad.ndim
    view[axis] = scale.numel()
    return scale.view(view)


class GWScheduler(nn.Module):
    """CosineAnnealingLR plus grade-specific phase offsets for Clifford models.

    The scheduler does not change model architecture. It scales gradients for
    multivector-valued parameters using per-grade factors derived from a cosine
    phase shift. This is the stable form of the "orthogonal rotor" idea:
    a shared cosine base schedule plus small grade phase offsets.

    When `learnable_phase_offsets=True`, the offsets are adapted by a small,
    stable heuristic that nudges phase shifts toward more balanced grade energy.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        engine,
        total_steps: int,
        num_grades: int = 4,
        phase_offsets=None,
        phase_stagger: bool = True,
        learnable_phase_offsets: bool = False,
        phase_step: float = 0.4 * math.pi,
        phase_update_lr: float = 0.02,
        eta_min: float = 0.0,
        multivector_axes: dict[int, tuple[int, ...]] | None = None,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.engine = engine
        self.total_steps = max(1, int(total_steps))
        self.num_grades = max(1, int(num_grades))
        self.phase_stagger = bool(phase_stagger)
        self.learnable_phase_offsets = bool(learnable_phase_offsets)
        self.phase_step = float(phase_step)
        self.phase_update_lr = float(phase_update_lr)
        self.current_step = 0
        self._multivector_axes = multivector_axes or {}
        self._last_grade_energy: torch.Tensor | None = None

        # Initialize the base schedule first so its initial LRs become the
        # reference scale for all optimizer groups.
        self.base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.total_steps,
            eta_min=eta_min,
        )

        phase_tensor = _as_phase_tensor(
            phase_offsets=phase_offsets,
            num_grades=self.num_grades,
            phase_stagger=self.phase_stagger,
            phase_step=self.phase_step,
            dtype=torch.float64,
            device="cpu",
        )
        if self.learnable_phase_offsets:
            self.phase_offsets = nn.Parameter(phase_tensor)
        else:
            self.register_buffer("phase_offsets", phase_tensor)

        # Preserve the base LR in each param group for logging/debugging.
        for group in self.optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])

    def _grade_factors(
        self,
        step: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if step is None:
            step = self.current_step
        if device is None:
            device = self.phase_offsets.device
        progress = min(max(step, 0) / max(1, self.total_steps), 1.0)
        phase = math.pi * progress
        offsets = self.phase_offsets.to(device=device, dtype=torch.float64)
        return 0.5 * (1.0 + torch.cos(phase + offsets))

    def _blade_scale(
        self,
        step: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        grade_factors = self._grade_factors(step=step, device=device)
        blade_scale = torch.ones(
            self.engine.n,
            dtype=grade_factors.dtype,
            device=grade_factors.device,
        )

        # Map higher grades to the last configured phase slot so we can keep
        # the default schedule compact (scalar / vector / bivector / trivector)
        # while still supporting higher-grade Clifford engines.
        for grade in range(self.engine.k + 1):
            slot = min(grade, self.num_grades - 1)
            mask = self.engine.grade_masks[grade].to(device=blade_scale.device, dtype=blade_scale.dtype)
            blade_scale = blade_scale + mask * (grade_factors[slot] - 1.0)
        return blade_scale

    def _accumulate_grade_energy(self, grad: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        """Measure how much gradient energy lives in each grade slot."""
        grade_energy = torch.zeros(
            self.num_grades,
            device=grad.device,
            dtype=torch.float64,
        )
        mv = grad.detach().abs()
        for axis in axes:
            mv_axis = mv.movedim(axis, -1)
            if mv_axis.ndim > 1:
                blade_energy = mv_axis.sum(dim=tuple(range(mv_axis.ndim - 1)))
            else:
                blade_energy = mv_axis
            for grade in range(self.engine.k + 1):
                slot = min(grade, self.num_grades - 1)
                mask = self.engine.grade_masks[grade].to(device=grad.device, dtype=blade_energy.dtype)
                grade_energy[slot] += (blade_energy * mask).sum().to(dtype=grade_energy.dtype)
        return grade_energy

    def _adapt_phase_offsets(self):
        """Nudge phase offsets toward more balanced effective update magnitudes."""
        if not self.learnable_phase_offsets or self.phase_update_lr <= 0:
            return
        if self._last_grade_energy is None:
            return

        energy = self._last_grade_energy
        if not torch.isfinite(energy).all() or energy.sum() <= 0:
            return

        device = self.phase_offsets.device
        offsets = self.phase_offsets
        progress = min(max(self.current_step, 0) / max(1, self.total_steps), 1.0)
        base_phase = math.pi * progress
        phase = base_phase + offsets.to(device=device, dtype=torch.float64)
        factors = 0.5 * (1.0 + torch.cos(phase))
        effective = factors * energy.to(device=device, dtype=torch.float64)
        target = effective.mean().clamp(min=1e-12)

        # Stable heuristic: move offsets so grade-specific effective updates
        # approach the mean effective update across grades.
        residual = effective - target
        grad = 2.0 * residual * energy.to(device=device, dtype=torch.float64)
        grad = grad * (-0.5 * torch.sin(phase))
        grad = grad / energy.mean().clamp(min=1.0)

        with torch.no_grad():
            updated = offsets.to(device=device, dtype=torch.float64) - self.phase_update_lr * grad
            wrapped = torch.remainder(updated + math.pi, 2 * math.pi) - math.pi
            if isinstance(self.phase_offsets, nn.Parameter):
                self.phase_offsets.copy_(wrapped.to(device=self.phase_offsets.device, dtype=self.phase_offsets.dtype))
            else:
                self.phase_offsets.copy_(wrapped.to(device=self.phase_offsets.device, dtype=self.phase_offsets.dtype))

    def scale_gradients(self):
        """Scale multivector gradients in-place before the optimizer step."""
        blade_scale = self._blade_scale(step=self.current_step)
        last_grade_energy = torch.zeros(
            self.num_grades,
            device=blade_scale.device,
            dtype=torch.float64,
        )

        for group in self.optimizer.param_groups:
            if group.get("grade_id") == "scalar":
                continue
            for param in group["params"]:
                if param.grad is None:
                    continue
                axes = self._multivector_axes.get(id(param))
                if not axes:
                    continue
                scaled = param.grad
                last_grade_energy = last_grade_energy + self._accumulate_grade_energy(scaled, axes)
                for axis in axes:
                    scaled = scaled * _broadcast_scale(blade_scale.to(device=scaled.device, dtype=scaled.dtype), scaled, axis)
                param.grad.copy_(scaled)

        self._last_grade_energy = last_grade_energy

    def step(self):
        """Advance the base cosine schedule for the next optimizer step."""
        self.base_scheduler.step()
        self._adapt_phase_offsets()
        self.current_step += 1

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "base_scheduler": self.base_scheduler.state_dict(),
            "current_step": self.current_step,
            "phase_offsets": self.phase_offsets.detach().clone().cpu(),
        }

    def load_state_dict(self, state_dict):
        self.base_scheduler.load_state_dict(state_dict["base_scheduler"])
        self.current_step = int(state_dict["current_step"])
        phase_offsets = state_dict.get("phase_offsets")
        if phase_offsets is not None:
            if isinstance(self.phase_offsets, nn.Parameter):
                with torch.no_grad():
                    self.phase_offsets.copy_(phase_offsets.to(self.phase_offsets.device, self.phase_offsets.dtype))
            else:
                self.phase_offsets.copy_(phase_offsets.to(self.phase_offsets.device, self.phase_offsets.dtype))
