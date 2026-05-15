"""Tests for the flow-style CFS model."""

import torch
import torch.nn.functional as F

from gaflowlm.clifford.engine import CliffordEngine
from gaflowlm.models.cfs_model import CFSAlgorithm, CFSModel


def test_cfs_model_basic():
    """CFS model forward pass produces a multivector velocity."""
    k = 3
    mv_dim = 1 << k
    B, L = 2, 8
    vocab = 100

    engine = CliffordEngine(k=k, dtype=torch.float64)
    model = CFSModel(
        vocab_size=vocab, hidden_size=16, k=k,
        n_blocks=2, n_heads=4, ff_dim=64, engine=engine,
    )

    xt = torch.randn(B, L, mv_dim, dtype=torch.float64)
    t = torch.rand(B, 1, dtype=torch.float64)
    velocity = model(xt, t)

    assert velocity.shape == (B, L, mv_dim), f"Shape: {velocity.shape}"
    assert not torch.isnan(velocity).any(), "NaN in velocity"
    print("  PASS: basic forward")


def test_cfs_model_backward():
    """CFS model gradients flow through the velocity field."""
    k = 3
    B, L = 2, 8
    vocab = 100

    engine = CliffordEngine(k=k, dtype=torch.float64)
    model = CFSModel(
        vocab_size=vocab, hidden_size=16, k=k,
        n_blocks=2, n_heads=4, ff_dim=64, engine=engine,
    )

    xt = torch.randn(B, L, 1 << k, dtype=torch.float64)
    t = torch.rand(B, 1, dtype=torch.float64)
    velocity = model(xt, t)
    target = torch.randn_like(velocity)
    loss = F.mse_loss(velocity, target)
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad, "No parameter gradients"
    print("  PASS: backward")


def test_cfs_algorithm():
    """CFSAlgorithm can be used in a flow-style training loop."""
    from dataclasses import dataclass

    @dataclass
    class DummyConfig:
        class Algo:
            name = "cfs"
            rhf_clifford_k = 3
            cfs_sample_steps = 4
        class Model:
            hidden_size = 16
            n_blocks = 2
            n_heads = 4
            length = 8
        class Optim:
            lr = 3e-4
            weight_decay = 0.0

        algo = Algo()
        model = Model()
        optim = Optim()

    class DummyTokenizer:
        vocab_size = 50
        def __len__(self):
            return self.vocab_size

    config = DummyConfig()
    tokenizer = DummyTokenizer()

    engine = CliffordEngine(k=3, dtype=torch.float64)
    algo = CFSAlgorithm(config, tokenizer, engine=engine)
    algo.to("cpu")

    x = torch.randint(0, 50, (2, 8))
    result = algo.train_step(x)

    assert "loss" in result
    assert not torch.isnan(torch.tensor(result["loss"]))
    assert result["loss"] > 0
    print(f"  PASS: training step (loss={result['loss']:.4f})")


def test_cfs_sampler():
    """CFSAlgorithm reverse-time sampler returns finite logits."""
    from dataclasses import dataclass

    @dataclass
    class DummyConfig:
        class Algo:
            name = "cfs"
            rhf_clifford_k = 3
            cfs_sample_steps = 4
        class Model:
            hidden_size = 16
            n_blocks = 2
            n_heads = 4
            length = 8
        class Optim:
            lr = 3e-4
            weight_decay = 0.0

        algo = Algo()
        model = Model()
        optim = Optim()

    class DummyTokenizer:
        vocab_size = 50
        def __len__(self):
            return self.vocab_size

    config = DummyConfig()
    tokenizer = DummyTokenizer()

    engine = CliffordEngine(k=3, dtype=torch.float64)
    algo = CFSAlgorithm(config, tokenizer, engine=engine)
    algo.to("cpu")

    state, logits = algo.sample(seq_len=8, num_steps=4)

    assert state.shape == (1, 8, 1 << 3)
    assert logits.shape == (1, 8, tokenizer.vocab_size)
    assert not torch.isnan(state).any()
    assert not torch.isnan(logits).any()
    print("  PASS: sampler")


if __name__ == "__main__":
    print("CFS model tests:")
    test_cfs_model_basic()
    test_cfs_model_backward()
    test_cfs_algorithm()
    test_cfs_sampler()
    print("\nAll CFS model tests passed!")
