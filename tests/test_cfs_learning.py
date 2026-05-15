"""Regression tests for CFS learning dynamics."""

import os
import sys
from argparse import Namespace

# cfs_overfit_probe is a script-style module (root-level absolute imports
# into gaflowlm/*.py). Adding gaflowlm/ to sys.path lets those resolve.
_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gaflowlm"))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from cfs_overfit_probe import run_overfit_probe  # noqa: E402


def test_cfs_overfits_fixed_synthetic_batch():
    """CFS should be able to fit one fixed synthetic batch."""
    args = Namespace(
        data="synthetic",
        steps=50,
        batch_size=4,
        seq_len=16,
        vocab_size=128,
        hidden_size=64,
        n_blocks=2,
        n_heads=4,
        clifford_k=3,
        lr=1e-3,
        device="auto",
        seed=1,
        time_sampling="uniform",
        loss="mse",
        sample_steps=16,
    )

    result = run_overfit_probe(args)

    assert result["end_loss"] < result["start_loss"], result
    assert result["min_train_loss"] < result["start_loss"], result
