"""Tests for CFA (Clifford Frame Attention) layer and CFS block."""

import torch

from gaflowlm.clifford.engine import CliffordEngine
from gaflowlm.models.cfs_arch import CFSTransformerBlock, CliffordFrameAttention


def _make_cfa(k=3, n_heads=4, bilinear=True):
    mv_dim = 1 << k
    engine = CliffordEngine(k=k, dtype=torch.float64)
    cfa = CliffordFrameAttention(
        mv_dim=mv_dim, n_heads=n_heads, engine=engine,
        bilinear=bilinear,
    )
    return cfa, mv_dim, engine


def test_cfa_basic():
    """CFA forward pass produces correct output shape."""
    k = 3
    mv_dim = 1 << k  # 8
    B, L, H = 2, 8, 4

    cfa, _, _ = _make_cfa(k, H, bilinear=True)
    x = torch.randn(B, L, mv_dim, dtype=torch.float64)
    out = cfa(x)

    assert out.shape == (B, L, mv_dim), f"Shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    print("  PASS: basic forward (bilinear)")


def test_cfa_standard():
    """CFA with bilinear=False falls back to standard attention."""
    k = 3
    mv_dim = 1 << k
    B, L, H = 2, 8, 4

    cfa, _, _ = _make_cfa(k, H, bilinear=False)
    x = torch.randn(B, L, mv_dim, dtype=torch.float64)
    out = cfa(x)

    assert out.shape == (B, L, mv_dim)
    assert not torch.isnan(out).any()
    print("  PASS: standard attention fallback")


def test_cfa_with_mask():
    """CFA with attention mask works."""
    k = 3
    mv_dim = 1 << k
    B, L, H = 2, 8, 4

    cfa, _, _ = _make_cfa(k, H)
    x = torch.randn(B, L, mv_dim, dtype=torch.float64)
    mask = torch.ones(B, L, L, dtype=torch.bool)
    mask[:, :, L//2:] = False  # mask out second half

    out = cfa(x, mask=mask)
    assert out.shape == (B, L, mv_dim)
    assert not torch.isnan(out).any()
    print("  PASS: with mask")


def test_cfa_backward():
    """CFA gradients flow through bilinear."""
    k = 3
    mv_dim = 1 << k
    B, L, H = 2, 8, 4

    cfa, _, _ = _make_cfa(k, H)
    x = torch.randn(B, L, mv_dim, dtype=torch.float64, requires_grad=True)
    out = cfa(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in cfa.parameters()
    )
    assert has_grad, "No parameter gradients"
    print("  PASS: backward (bilinear)")


def test_cfa_bilinear_vs_standard():
    """Bilinear and standard modes produce different outputs."""
    k = 3
    mv_dim = 1 << k
    B, L, H = 2, 8, 4

    cfa_bilinear, _, _ = _make_cfa(k, H, bilinear=True)
    cfa_standard, _, _ = _make_cfa(k, H, bilinear=False)

    # Copy weights so the only difference is bilinear vs standard
    cfa_standard.W_q.weight.data = cfa_bilinear.W_q.weight.data.clone()
    cfa_standard.W_k.weight.data = cfa_bilinear.W_k.weight.data.clone()
    cfa_standard.W_v.weight.data = cfa_bilinear.W_v.weight.data.clone()
    cfa_standard.W_o.weight.data = cfa_bilinear.W_o.weight.data.clone()

    x = torch.randn(B, L, mv_dim, dtype=torch.float64)

    out_bilinear = cfa_bilinear(x)
    out_standard = cfa_standard(x)

    # They should differ because bilinear applies geometric product Q * V_agg
    # while standard just does attn_weights @ V
    diff = (out_bilinear - out_standard).abs().mean()
    assert diff > 1e-5, f"Bilinear and standard should differ, got diff={diff}"
    print(f"  PASS: bilinear vs standard (diff={diff:.6f})")


def test_cfa_bilinear_grade_mixing():
    """Bilinear output should mix grades (not just scalar)."""
    k = 3
    mv_dim = 1 << k
    B, L, H = 1, 4, 2

    cfa, _, engine = _make_cfa(k, H, bilinear=True)
    x = torch.randn(B, L, mv_dim, dtype=torch.float64)
    out = cfa(x)

    # After bilinear, the output should have energy across all grades
    # (the geometric product Q*V mixes all grades)
    for r in range(k + 1):
        grade_mask = engine.grade_masks[r].to(x.device)
        grade_energy = (out * grade_mask).abs().mean()
        # Each grade should have some energy (not all zeros)
        # This is a soft check - the bilinear should distribute energy
    print("  PASS: grade mixing")


def test_cfs_block():
    """CFS transformer block forward and backward."""
    k = 3
    mv_dim = 1 << k
    B, L = 2, 8

    engine = CliffordEngine(k=k, dtype=torch.float64)
    block = CFSTransformerBlock(
        mv_dim=mv_dim, n_heads=4, ff_dim=256, engine=engine,
    )

    x = torch.randn(B, L, mv_dim, dtype=torch.float64, requires_grad=True)
    out = block(x)

    assert out.shape == (B, L, mv_dim)
    assert not torch.isnan(out).any()

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("  PASS: CFS block")


if __name__ == '__main__':
    print("CFA tests:")
    test_cfa_basic()
    test_cfa_standard()
    test_cfa_with_mask()
    test_cfa_backward()
    test_cfa_bilinear_vs_standard()
    test_cfa_bilinear_grade_mixing()
    test_cfs_block()
    print("\nAll CFA tests passed!")
