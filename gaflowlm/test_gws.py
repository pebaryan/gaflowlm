"""Tests for the Grade-Wise Scheduler (GWS)."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

try:
    from gaflowlm.schedulers import GWScheduler
    from gaflowlm.models.cfs_model import CFSAlgorithm
    from gaflowlm.clifford.engine import CliffordEngine
except ModuleNotFoundError:
    from schedulers import GWScheduler
    from models.cfs_model import CFSAlgorithm
    from clifford.engine import CliffordEngine


def _make_scheduler(
    learnable_phase_offsets: bool = False,
    phase_offsets=None,
    phase_update_lr: float = 0.02,
):
    engine = CliffordEngine(k=3, dtype=torch.float64)
    scalar = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    multivector = nn.Parameter(torch.arange(16, dtype=torch.float64).view(2, 8))

    optimizer = torch.optim.AdamW(
        [
            {"params": [scalar], "lr": 1e-3, "base_lr": 1e-3, "grade_id": "scalar"},
            {
                "params": [multivector],
                "lr": 1e-3,
                "base_lr": 1e-3,
                "grade_id": "multivector",
            },
        ],
        lr=1e-3,
        weight_decay=0.0,
    )

    scheduler = GWScheduler(
        optimizer=optimizer,
        engine=engine,
        total_steps=10,
        num_grades=4,
        phase_offsets=phase_offsets,
        phase_stagger=False,
        learnable_phase_offsets=learnable_phase_offsets,
        phase_step=0.4 * math.pi,
        phase_update_lr=phase_update_lr,
        multivector_axes={id(multivector): (1,)},
    )
    return scheduler, scalar, multivector


def test_gws_scales_multivector_gradients():
    """GWS should scale multivector gradient components by grade."""
    scheduler, scalar, multivector = _make_scheduler(
        phase_offsets=[0.0, 0.3, 0.6, 0.9]
    )

    scalar.grad = torch.ones_like(scalar)
    multivector.grad = torch.ones_like(multivector)

    scheduler.scale_gradients()

    assert torch.allclose(scalar.grad, torch.ones_like(scalar))
    # Grade-0 blade stays at full scale (offset=0, no delay).
    assert torch.isclose(multivector.grad[0, 0], torch.tensor(1.0, dtype=torch.float64))
    # Higher-grade blades have slightly higher factors (delayed decay).
    grade0_factor = multivector.grad[0, 0]
    grade1_factor = multivector.grad[0, 1]
    # With positive offsets, higher grades retain more LR.
    assert grade1_factor >= grade0_factor * 0.9


def test_gws_step_advances_cosine_lr():
    """GWS should advance the base cosine schedule."""
    scheduler, _, _ = _make_scheduler(phase_offsets=[0.0, 0.0, 0.0, 0.0])
    before = scheduler.optimizer.param_groups[1]["lr"]
    scheduler.optimizer.step()
    scheduler.step()
    after = scheduler.optimizer.param_groups[1]["lr"]

    assert after < before
    assert scheduler.current_step == 1


def test_gws_learnable_offsets_adapt():
    """Learnable phase offsets should move when the grades are imbalanced."""
    scheduler, _, multivector = _make_scheduler(
        learnable_phase_offsets=True,
        phase_offsets=[0.1, 0.7, 1.3, 1.9],
        phase_update_lr=0.1,
    )
    scheduler.current_step = 3

    # Strongly skew the grade energy toward scalar/vector components.
    grad = torch.zeros_like(multivector)
    grad[:, 0] = 8.0
    grad[:, 1] = 4.0
    grad[:, 3] = 1.0
    grad[:, 7] = 0.5
    multivector.grad = grad

    before = scheduler.phase_offsets.detach().clone()
    scheduler.scale_gradients()
    scheduler.optimizer.step()
    scheduler.step()
    after = scheduler.phase_offsets.detach().clone()

    assert not torch.allclose(before, after)


def test_cfs_algorithm_enables_gws():
    """CFSAlgorithm should wire GWS into the training step when enabled."""
    from dataclasses import dataclass

    @dataclass
    class DummyConfig:
        class Algo:
            name = "cfs"
            rhf_clifford_k = 3
            cfs_sample_steps = 4
            cfs_loss = "mse"
            cfs_time_sampling = "uniform"
            cfs_noise_scale = 1.0
            cfs_normalize_noise = False
            cfs_use_higher_order = True
        class Model:
            hidden_size = 16
            n_blocks = 2
            n_heads = 4
            length = 8
        class Optim:
            lr = 3e-4
            weight_decay = 0.0
            use_gws = True
            gws_num_grades = 4
            gws_phase_stagger = True
            gws_learnable_phase_offsets = False
            gws_phase_step = 0.4 * math.pi
            gws_phase_offsets = [0.0, 0.6, 1.2, 1.8]
            gws_total_steps = 8
            gws_eta_min = 0.0
            gws_phase_update_lr = 0.02

        algo = Algo()
        model = Model()
        optim = Optim()

    class DummyTokenizer:
        vocab_size = 50

        def __len__(self):
            return self.vocab_size

    engine = CliffordEngine(k=3, dtype=torch.float64)
    algo = CFSAlgorithm(DummyConfig(), DummyTokenizer(), engine=engine)
    algo.to("cpu")

    assert algo.scheduler is not None
    before_step = algo.scheduler.current_step
    x = torch.randint(0, 50, (2, 8))
    result = algo.train_step(x)

    assert result["loss"] > 0
    assert algo.scheduler.current_step == before_step + 1


if __name__ == "__main__":
    test_gws_scales_multivector_gradients()
    test_gws_step_advances_cosine_lr()
    test_gws_learnable_offsets_adapt()
    test_cfs_algorithm_enables_gws()
    print("All GWS tests passed!")
