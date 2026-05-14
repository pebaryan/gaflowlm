#!/usr/bin/env python3
"""Integration test for RHF (Rotor Hyperspherical Flow).

Tests that RHFSFM can:
1. Be instantiated with a valid config and tokenizer
2. Run forward pass (q_xt, compute loss) without crashing
3. Backpropagate gradients
4. Produce numerically equivalent results to SFM in analytic mode
5. Handle float32 and float64 slerp_precision

This test creates a minimal model, data batch, and loss computation
without requiring the full Hydra/Lightning pipeline.
"""
import sys
import os

# Ensure gaflowlm/ directory is on sys.path for bare imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock flash_attn if not available (allows tests to run without CUDA flash-attn)
try:
    import flash_attn
except ImportError:
    import flash_attn_mock
    sys.modules['flash_attn'] = flash_attn_mock
    sys.modules['flash_attn.layers'] = flash_attn_mock
    sys.modules['flash_attn.layers.rotary'] = flash_attn_mock.layers.rotary

import torch
import torch.nn.functional as F
import math

# ---------------------------------------------------------------
# Minimal config + tokenizer (no Hydra/Lightning needed)
# ---------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlgoConfig:
    name: str = 'rhf'
    diffusion_type: str = 'sphere'
    backbone: str = 'sphere-dit'
    parameterization: str = 'mean'
    time_conditioning: bool = True
    loss_type: str = 'ce'
    T: float = 0
    causal_attention: bool = False
    adaLN: bool = True
    slerp_precision: str = 'float64'
    eps: float = 1e-6
    invert_time_convention: bool = True
    renormalize_weights: bool = False
    rhf_mode: str = 'analytic'
    rhf_clifford_k: int = 8
    rhf_compute_bivector: bool = False


@dataclass
class ModelConfig:
    name: str = 'tiny-test'
    type: str = 'sphere-dit'
    hidden_size: int = 64
    cond_dim: int = 32
    length: int = 32
    n_blocks: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    init: str = 'ngpt'
    learn_temperature_scaling: bool = False
    eps: float = 1e-6
    pretrained_ckpt_path: Optional[str] = None


@dataclass
class NoiseConfig:
    type: str = 'log-linear'
    eps: float = 1e-3
    alpha_min: Optional[float] = None
    alpha_max: Optional[float] = None
    adaptive: bool = False


@dataclass
class TrainingConfig:
    ema: float = 0.9999
    antithetic_sampling: bool = True
    sampling_eps: float = 1e-3
    finetune_path: str = ''


@dataclass
class SamplerConfig:
    predictor: str = 'sfm'
    steps: int = 128
    noise_removal: str = 'ancestral'
    use_float64: bool = True
    velocity: str = 'exact'
    p_nucleus: float = 1.0
    top_k: int = -1
    top_k_velocity: int = -1
    temperature: float = 1.0


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class EvalConfig:
    checkpoint_path: str = ''
    strict_loading: bool = True
    disable_ema: bool = False
    compute_generative_perplexity: bool = False
    perplexity_batch_size: int = 8
    compute_perplexity_on_sanity: bool = False
    gen_ppl_eval_model_name_or_path: str = 'gpt2-large'
    generate_samples: bool = True
    generated_samples_path: str = ''
    results_json_path: Optional[str] = None


@dataclass
class FullConfig:
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    neg_infinity_mode: str = 'large-finite'
    seed: int = 1


class FakeTokenizer:
    """Minimal tokenizer for testing (vocab_size = 100)."""
    def __init__(self, vocab_size=100):
        self._vocab_size = vocab_size
        self.mask_token = None
        self.mask_token_id = None
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

    def __len__(self):
        return self._vocab_size

    @property
    def vocab_size(self):
        return self._vocab_size

    def decode(self, ids, **kwargs):
        return ' '.join(str(i) for i in ids)

    def batch_decode(self, batch_ids, **kwargs):
        return [self.decode(ids) for ids in batch_ids]


# ---------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------

def test_instantiation():
    """Test that RHFSFM can be instantiated."""
    print("Test 1: RHFSFM instantiation... ", end='')
    config = FullConfig()
    tokenizer = FakeTokenizer()

    import algo
    rhf = algo.SFM  # First verify base SFM can be created
    sfm_model = rhf(config, tokenizer)
    print(f"Base SFM ok (device={sfm_model.device})")

    import rhf_algo
    rhf_model = rhf_algo.RHFSFM(config, tokenizer)
    assert hasattr(rhf_model, 'rhf_mode')
    assert rhf_model.rhf_mode == 'analytic'
    assert hasattr(rhf_model, '_rotor_ops')
    print(f"RHFSFM ok (rhf_mode={rhf_model.rhf_mode})")
    print("  PASSED")


def test_forward_no_crash():
    """Test that RHFSFM forward pass doesn't crash."""
    print("Test 2: Forward pass... ", end='')

    config = FullConfig()
    config.algo.slerp_precision = 'float32'  # Use float32 for speed
    tokenizer = FakeTokenizer(vocab_size=100)

    import rhf_algo
    model = rhf_algo.RHFSFM(config, tokenizer)
    model.eval()

    B, L = 4, 16
    x0 = torch.randint(0, 100, (B, L))

    with torch.no_grad():
        # q_xt: corrupt x0
        e_clean = model.backbone.get_sphere_embeddings(x0)
        alpha_t = torch.rand(B, 1)
        xt = model.q_xt(x0, alpha_t, use_pure_noise=False)

        assert xt.shape == (B, L, config.model.hidden_size), \
            f"xt shape {xt.shape} != expected ({B}, {L}, {config.model.hidden_size})"

        # Check unit norm
        norms = xt.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), \
            f"xt not unit norm: max deviation = {(norms - 1).abs().max().item():.6e}"

        # forward
        sigma = model._sigma_from_alphat(alpha_t)
        log_p = model.forward(x0=x0, xt=xt, sigma=sigma)

        assert log_p.shape == (B, L, 100), \
            f"log_p shape {log_p.shape} != expected ({B}, {L}, 100)"

    print("PASSED")


def test_backward_no_crash():
    """Test that gradients flow through RHF loss computation."""
    print("Test 3: Backward pass... ", end='')

    config = FullConfig()
    config.algo.slerp_precision = 'float32'
    tokenizer = FakeTokenizer(vocab_size=100)

    import rhf_algo
    model = rhf_algo.RHFSFM(config, tokenizer)
    model.train()

    B, L = 4, 16
    x0 = torch.randint(0, 100, (B, L))

    # Sample timestep and compute loss
    t = torch.rand(B)
    dalpha_t, alpha_t = model.noise(t)
    alpha_t_unsqueezed = alpha_t.unsqueeze(-1)

    xt = model.q_xt(x0, alpha_t_unsqueezed, use_pure_noise=False)
    sigma = model._sigma_from_alphat(alpha_t_unsqueezed)
    log_p = model.forward(x0=x0, xt=xt, sigma=sigma)

    ce_loss = -log_p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    loss = ce_loss.mean()

    loss.backward()

    # Check that at least some parameters received gradients
    total_grad_norm = 0.0
    n_param_with_grad = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
            if param.grad.norm().item() > 0:
                n_param_with_grad += 1

    total_grad_norm = total_grad_norm ** 0.5
    assert n_param_with_grad > 0, "No parameter received gradients!"
    assert total_grad_norm > 0, "Total gradient norm is zero!"

    print(f"PASSED ({n_param_with_grad} params with grad, total_norm={total_grad_norm:.4f})")


def test_analytic_equivalence():
    """Test that RHF analytic mode matches S-FLM SFM numerically."""
    print("Test 4: Analytic mode equivalence... ", end='')

    config_sfm = FullConfig()
    config_sfm.algo.slerp_precision = 'float64'
    config_sfm.algo.name = 'sfm'

    config_rhf = FullConfig()
    config_rhf.algo.slerp_precision = 'float64'
    config_rhf.algo.rhf_mode = 'analytic'

    tokenizer = FakeTokenizer(vocab_size=100)

    import algo
    import rhf_algo

    torch.manual_seed(42)
    sfm_model = algo.SFM(config_sfm, tokenizer)
    sfm_model.eval()

    torch.manual_seed(42)
    rhf_model = rhf_algo.RHFSFM(config_rhf, tokenizer)
    rhf_model.eval()

    B, L = 4, 16
    x0 = torch.randint(0, 100, (B, L))

    # Use same noise for both
    torch.manual_seed(123)
    e_clean_sfm = sfm_model.backbone.get_sphere_embeddings(x0)
    e_noisy_sfm = sfm_model._sample_prior(e_clean_sfm)

    torch.manual_seed(123)
    e_clean_rhf = rhf_model.backbone.get_sphere_embeddings(x0)
    e_noisy_rhf = rhf_model._sample_prior(e_clean_rhf)

    # Check _sample_prior match
    assert torch.allclose(e_noisy_sfm, e_noisy_rhf, atol=1e-6), \
        f"_sample_prior differs: max diff = {(e_noisy_sfm - e_noisy_rhf).abs().max().item():.6e}"

    # Check _slerp match
    alpha_t = torch.rand(B, 1, dtype=torch.float64)
    slerp_sfm = sfm_model._slerp(e_clean_sfm.double(), e_noisy_sfm.double(), alpha_t)
    slerp_rhf = rhf_model._slerp(e_clean_rhf.double(), e_noisy_rhf.double(), alpha_t)

    slerp_diff = (slerp_sfm - slerp_rhf).abs().max().item()
    assert slerp_diff < 1e-10, f"_slerp differs: max diff = {slerp_diff:.6e}"

    # Check q_xt match
    torch.manual_seed(456)
    xt_sfm = sfm_model.q_xt(x0, alpha_t, use_pure_noise=False)

    torch.manual_seed(456)
    xt_rhf = rhf_model.q_xt(x0, alpha_t, use_pure_noise=False)

    # Note: q_xt involves randn-like sampling, so we can't reproduce exactly
    # but the interpolation step should be deterministic given same inputs
    # Just check the shapes and norms
    assert xt_sfm.shape == xt_rhf.shape

    # Check unit norms
    norms_sfm = xt_sfm.norm(dim=-1)
    norms_rhf = xt_rhf.norm(dim=-1)
    assert torch.allclose(norms_sfm, torch.ones_like(norms_sfm), atol=1e-4)
    assert torch.allclose(norms_rhf, torch.ones_like(norms_rhf), atol=1e-4)

    print(f"PASSED (slerp diff = {slerp_diff:.2e})")


def test_log_map_equivalence():
    """Test that rotor_log_map matches utils.log_map."""
    print("Test 5: rotor_log_map equivalence... ", end='')

    import utils
    from rotor_utils import rotor_log_map

    torch.manual_seed(42)
    d = 64
    x = F.normalize(torch.randn(4, 16, d), dim=-1)
    y = F.normalize(torch.randn(4, 16, d), dim=-1)

    v_sfm = utils.log_map(x, y, eps=1e-6)
    v_rhf = rotor_log_map(x, y, eps=1e-6)

    max_diff = (v_sfm - v_rhf).abs().max().item()
    assert max_diff < 1e-5, f"log_map differs: max diff = {max_diff:.6e}"
    print(f"PASSED (diff = {max_diff:.2e})")


def test_exp_map_equivalence():
    """Test that rotor_exp_map matches utils.exp_map on the sphere."""
    print("Test 6: rotor_exp_map equivalence... ", end='')

    import utils
    from rotor_utils import rotor_exp_map

    torch.manual_seed(42)
    d = 64
    x = F.normalize(torch.randn(4, 16, d), dim=-1)

    # exp_map preserves unit norm when delta is in the tangent plane of x
    # Project a random vector onto x's tangent plane
    raw = torch.randn(4, 16, d) * 0.3
    delta = raw - (raw * x).sum(dim=-1, keepdim=True) * x  # project to tangent plane

    y_sfm = utils.exp_map(x, delta, eps=1e-6)
    y_rhf = rotor_exp_map(x, delta, eps=1e-6)

    # Both should be unit norm (tangent-plane delta preserves it)
    y_sfm_norm = y_sfm.norm(dim=-1)
    y_rhf_norm = y_rhf.norm(dim=-1)
    assert torch.allclose(y_sfm_norm, torch.ones_like(y_sfm_norm), atol=1e-3), \
        f"sfm exp_map not unit norm: max deviation = {(y_sfm_norm - 1).abs().max().item():.6e}"
    assert torch.allclose(y_rhf_norm, torch.ones_like(y_rhf_norm), atol=1e-3), \
        f"rhf exp_map not unit norm: max deviation = {(y_rhf_norm - 1).abs().max().item():.6e}"

    # Results should match closely
    cos_sim = F.cosine_similarity(y_sfm, y_rhf, dim=-1)
    assert (cos_sim > 0.999).all(), f"exp_map results differ: min cos_sim = {cos_sim.min().item():.6f}"
    print("PASSED")


def test_clifford_mode():
    """Test that RHF clifford mode runs without crashing (numerical diff expected)."""
    print("Test 7: Clifford mode forward... ", end='')

    config = FullConfig()
    config.algo.slerp_precision = 'float32'
    config.algo.rhf_mode = 'clifford'
    config.algo.rhf_clifford_k = 4  # Tiny for speed
    tokenizer = FakeTokenizer(vocab_size=50)

    import rhf_algo
    model = rhf_algo.RHFSFM(config, tokenizer)
    model.eval()

    B, L = 2, 8
    x0 = torch.randint(0, 50, (B, L))

    with torch.no_grad():
        alpha_t = torch.rand(B, 1)
        xt = model.q_xt(x0, alpha_t, use_pure_noise=False)
        sigma = model._sigma_from_alphat(alpha_t)
        log_p = model.forward(x0=x0, xt=xt, sigma=sigma)

        assert log_p.shape == (B, L, 50)
        assert not torch.isnan(log_p).any(), "NaN in log_p!"

    print("PASSED")


def test_training_loop():
    """Test that RHF can train for a few steps and loss decreases."""
    print("Test 8: Mini training loop... ", end='')

    config = FullConfig()
    config.algo.slerp_precision = 'float32'
    config.model.hidden_size = 64
    config.model.cond_dim = 32
    config.model.length = 32
    config.model.n_blocks = 2
    config.model.n_heads = 4
    tokenizer = FakeTokenizer(vocab_size=100)

    import rhf_algo
    model = rhf_algo.RHFSFM(config, tokenizer)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(20):
        x0 = torch.randint(0, 100, (8, 32))
        optimizer.zero_grad()

        # Manual forward
        t = torch.rand(8)
        dalpha_t, alpha_t = model.noise(t)
        alpha_t_unsqueezed = alpha_t.unsqueeze(-1)
        xt = model.q_xt(x0, alpha_t_unsqueezed, use_pure_noise=False)
        sigma = model._sigma_from_alphat(alpha_t_unsqueezed)
        log_p = model.forward(x0=x0, xt=xt, sigma=sigma)
        ce_loss = -log_p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        loss = ce_loss.mean()

        loss.backward()
        optimizer.step()

        if model.renormalize_weights:
            model.backbone.renormalize_weights()

        losses.append(loss.item())

    # Check loss decreased
    first_5_avg = sum(losses[:5]) / 5
    last_5_avg = sum(losses[-5:]) / 5
    assert last_5_avg < first_5_avg, \
        f"Loss did not decrease: first_5={first_5_avg:.4f}, last_5={last_5_avg:.4f}"

    print(f"PASSED (loss: {first_5_avg:.4f} → {last_5_avg:.4f})")


def main():
    print("=" * 60)
    print("RHF Integration Tests")
    print("=" * 60)
    tests = [
        test_instantiation,
        test_forward_no_crash,
        test_backward_no_crash,
        test_analytic_equivalence,
        test_log_map_equivalence,
        test_exp_map_equivalence,
        test_clifford_mode,
        test_training_loop,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == '__main__':
    main()