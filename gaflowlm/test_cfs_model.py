"""Tests for CFS model (full pipeline)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

try:
    from gaflowlm.models.cfs_model import CFSModel, CFSAlgorithm
    from gaflowlm.clifford.engine import CliffordEngine
except ModuleNotFoundError:
    from models.cfs_model import CFSModel, CFSAlgorithm
    from clifford.engine import CliffordEngine


def test_cfs_model_basic():
    """CFS model forward pass produces logits."""
    k = 3
    mv_dim = 1 << k
    B, L = 2, 8
    vocab = 100

    engine = CliffordEngine(k=k, dtype=torch.float64)
    model = CFSModel(
        vocab_size=vocab, hidden_size=16, k=k,
        n_blocks=2, n_heads=4, ff_dim=64, engine=engine,
    )

    x = torch.randint(0, vocab, (B, L))
    logits = model(x)

    assert logits.shape == (B, L, vocab), f"Shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits"
    print("  PASS: basic forward")


def test_cfs_model_backward():
    """CFS model gradients flow."""
    k = 3
    B, L = 2, 8
    vocab = 100

    engine = CliffordEngine(k=k, dtype=torch.float64)
    model = CFSModel(
        vocab_size=vocab, hidden_size=16, k=k,
        n_blocks=2, n_heads=4, ff_dim=64, engine=engine,
    )

    x = torch.randint(0, vocab, (B, L))
    logits = model(x)

    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab),
        x[:, 1:].reshape(-1),
    )
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad, "No parameter gradients"
    print("  PASS: backward")


def test_cfs_algorithm():
    """CFSAlgorithm can be used in a training loop."""
    from dataclasses import dataclass, field

    @dataclass
    class DummyConfig:
        class Algo:
            name = 'cfs'
            rhf_clifford_k = 3
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

    config = DummyConfig()
    tokenizer = DummyTokenizer()

    engine = CliffordEngine(k=3, dtype=torch.float64)
    algo = CFSAlgorithm(config, tokenizer, engine=engine)
    algo.to('cpu')

    x = torch.randint(0, 50, (2, 8))
    result = algo.train_step(x)

    assert 'loss' in result
    assert not torch.isnan(torch.tensor(result['loss']))
    assert result['loss'] > 0
    print(f"  PASS: training step (loss={result['loss']:.4f})")


if __name__ == '__main__':
    print("CFS model tests:")
    test_cfs_model_basic()
    test_cfs_model_backward()
    test_cfs_algorithm()
    print("\nAll CFS model tests passed!")
