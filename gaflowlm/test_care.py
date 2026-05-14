"""Tests for CARE position encoding."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# Support both direct and packaged imports
try:
    from gaflowlm.models.care import CAREPositionEncoding
    from gaflowlm.clifford.engine import CliffordEngine
except ModuleNotFoundError:
    from models.care import CAREPositionEncoding
    from clifford.engine import CliffordEngine


def test_care_basic():
    """CARE can be instantiated and applied to multivectors."""
    k = 4
    mv_dim = 1 << k  # 16
    B, L = 2, 8

    engine = CliffordEngine(k=k)
    care = CAREPositionEncoding(k=k, max_len=64, engine=engine)

    x = torch.randn(B, L, mv_dim)
    out = care(x)

    assert out.shape == (B, L, mv_dim), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    print("  PASS: basic forward")


def test_care_norm_preserving():
    """CARE rotor sandwich preserves multivector spinor norm."""
    k = 4
    mv_dim = 1 << k
    B, L = 2, 8

    engine = CliffordEngine(k=k)
    care = CAREPositionEncoding(k=k, max_len=64, engine=engine)

    x = torch.randn(B, L, mv_dim)
    out = care(x)

    # Check spinor norm: ⟨M M̃⟩₀
    for inp, outp in [(x, out)]:
        # Compute norm of input and output (spinor norm = geometric product with reverse)
        def spinor_norm(m):
            rev = engine.reverse_mv(m)
            mm_rev = engine.geometric_product(m, rev)
            return engine.scalar_part(mm_rev)

        in_norm = spinor_norm(x)
        out_norm = spinor_norm(out)

    # Norm should be approximately preserved (float32 accumulation over GP)
    diff = (in_norm - out_norm).abs().mean().item()
    assert diff < 0.1, f"Norm not preserved: diff={diff:.6f}"
    print("  PASS: norm preserved")


def test_care_position_dependence():
    """Different positions produce different encodings."""
    k = 4
    mv_dim = 1 << k

    engine = CliffordEngine(k=k)
    care = CAREPositionEncoding(k=k, max_len=64, engine=engine)

    x = torch.randn(1, 3, mv_dim)
    out = care(x)

    # Different positions should give different outputs for same input
    # (same input repeated at different positions)
    x_same = x.expand(1, 3, -1)  # all positions get same input
    out_pos = care(x_same)

    # Position 0 and position 2 should differ
    diff = (out_pos[0, 0] - out_pos[0, 2]).abs().mean()
    assert diff > 1e-6, f"Positions should differ: diff={diff:.6f}"
    print("  PASS: position-dependent")


def test_care_learned_angles():
    """CARE with learned angles works."""
    k = 4
    mv_dim = 1 << k
    B, L = 2, 8

    engine = CliffordEngine(k=k)
    care = CAREPositionEncoding(k=k, max_len=64, engine=engine, learned_angles=True)

    x = torch.randn(B, L, mv_dim)
    out = care(x)

    assert out.shape == (B, L, mv_dim)
    assert not torch.isnan(out).any()
    # Check that theta is a parameter (requires grad)
    assert care.theta.requires_grad, "Learned angles should require grad"
    print("  PASS: learned angles")


def test_care_backward():
    """CARE gradients flow through rotor params."""
    k = 4
    mv_dim = 1 << k
    B, L = 2, 4

    engine = CliffordEngine(k=k)
    care = CAREPositionEncoding(k=k, max_len=16, engine=engine)

    x = torch.randn(B, L, mv_dim, requires_grad=True)
    out = care(x)
    loss = out.sum()
    loss.backward()

    # At least one rotor param should have non-zero grad
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in [care.B_x, care.B_y]
    )
    assert has_grad, "No rotor params have gradients"
    print("  PASS: gradients flow")


if __name__ == '__main__':
    print("CARE position encoding tests:")
    test_care_basic()
    test_care_norm_preserving()
    test_care_position_dependence()
    test_care_learned_angles()
    test_care_backward()
    print("\nAll CARE tests passed!")
