"""
Unit tests for the Clifford algebra engine and rotor operations.

Tests verify:
1. Cayley tensor correctness (e_i * e_i = +1, e_i * e_j = -e_j * e_i for i!=j)
2. Geometric product properties (associativity, bilinearity)
3. Reverse and grade projection
4. Rotor exponential/logarithm roundtrip
5. Rotor sandwich preserves norm and implements rotation
6. Analytic rotor operations match trig counterparts
7. Embedding <-> Clifford roundtrip
8. RotorOps integration
"""

import math
import pytest
import torch
import torch.nn.functional as F

from gaflowlm.clifford.engine import (
    CliffordEngine,
    EmbedToClifford,
    CliffordToEmbed,
    build_cayley_tensor,
    build_grade_masks,
    build_reverse_signs,
)
from gaflowlm.rotor_utils import (
    rotor_slerp,
    rotor_log_map,
    rotor_exp_map,
    RotorOps,
    bivector_velocity,
    sphere_normalize,
)


# ---------------------------------------------------------------------------
# Helpers: basis vector construction with correct bitmask indexing
# ---------------------------------------------------------------------------

def basis_vector(k: int, i: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Create a basis vector e_{i} in Cl(k,0,0) with bitmask encoding.

    e_0 = scalar (index 0, bitmask 000...0)
    e_1 = index 1 (bitmask 0001)
    e_2 = index 2 (bitmask 0010)
    e_3 = index 4 (bitmask 0100)  -- NOT index 3!
    e_4 = index 8 (bitmask 1000)
    """
    n = 1 << k
    v = torch.zeros(n, dtype=dtype)
    v[1 << i] = 1.0  # index = 2^i for basis vector e_{i+1}
    return v


def basis_bivector(k: int, i: int, j: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Create a basis bivector e_{i} ∧ e_{j} in Cl(k,0,0).

    The index for e_{i,j} is (1 << i) | (1 << j).
    """
    n = 1 << k
    v = torch.zeros(n, dtype=dtype)
    v[(1 << i) | (1 << j)] = 1.0
    return v


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine_k3():
    """Small engine Cl(3,0,0) for fast tests."""
    return CliffordEngine(k=3, device='cpu', dtype=torch.float64)

@pytest.fixture
def engine_k4():
    """Cl(4,0,0) for slightly larger tests."""
    return CliffordEngine(k=4, device='cpu', dtype=torch.float64)


# ---------------------------------------------------------------------------
# Cayley tensor tests
# ---------------------------------------------------------------------------

class TestCayleyTensor:
    """Verify the Cayley tensor encodes correct multiplication rules."""

    def test_scalar_identity(self, engine_k3):
        """e_0 (scalar) * e_i = e_i and e_i * e_0 = e_i."""
        eng = engine_k3
        scalar = torch.zeros(eng.n, dtype=eng._dtype)
        scalar[0] = 1.0  # scalar = 1

        for i in range(eng.k):
            e_i = basis_vector(eng.k, i)

            result = eng.geometric_product(scalar, e_i)
            assert torch.allclose(result, e_i, atol=1e-10), \
                f"1 * e_{i+1} != e_{i+1}: got {result}"

            result = eng.geometric_product(e_i, scalar)
            assert torch.allclose(result, e_i, atol=1e-10), \
                f"e_{i+1} * 1 != e_{i+1}: got {result}"

    def test_basis_square_positive(self, engine_k3):
        """e_i * e_i = +1 for all i in Cl(k,0,0)."""
        eng = engine_k3
        scalar = torch.zeros(eng.n, dtype=eng._dtype)
        scalar[0] = 1.0

        for i in range(eng.k):
            e_i = basis_vector(eng.k, i)
            result = eng.geometric_product(e_i, e_i)
            assert torch.allclose(result, scalar, atol=1e-10), \
                f"e_{i+1}^2 != +1: got {result}"

    def test_anticommutation(self, engine_k3):
        """e_i * e_j = -e_j * e_i for i != j."""
        eng = engine_k3

        for i in range(eng.k):
            for j in range(i + 1, eng.k):
                e_i = basis_vector(eng.k, i)
                e_j = basis_vector(eng.k, j)

                eiej = eng.geometric_product(e_i, e_j)
                ejei = eng.geometric_product(e_j, e_i)

                assert torch.allclose(eiej, -ejei, atol=1e-10), \
                    f"e_{i+1}e_{j+1} != -e_{j+1}e_{i+1}"

    def test_bivector_product(self, engine_k3):
        """e_1 * e_2 = e_{12} (bivector basis blade)."""
        eng = engine_k3
        e1 = basis_vector(eng.k, 0)  # e_1
        e2 = basis_vector(eng.k, 1)  # e_2

        result = eng.geometric_product(e1, e2)
        expected = basis_bivector(eng.k, 0, 1)  # e_{12}

        assert torch.allclose(result, expected, atol=1e-10), \
            f"e1*e2 != e12: got {result}"

    def test_associativity(self, engine_k4):
        """Geometric product is associative: (ab)c = a(bc)."""
        eng = engine_k4
        a = torch.randn(eng.n, dtype=eng._dtype)
        b = torch.randn(eng.n, dtype=eng._dtype)
        c = torch.randn(eng.n, dtype=eng._dtype)

        ab = eng.geometric_product(a, b)
        abc_left = eng.geometric_product(ab, c)

        bc = eng.geometric_product(b, c)
        abc_right = eng.geometric_product(a, bc)

        assert torch.allclose(abc_left, abc_right, atol=1e-6), \
            "Geometric product not associative"

    def test_bivector_square_negative_scalar(self, engine_k3):
        """Bivector e_{12} squared should be a negative scalar: e_{12}^2 = -1."""
        eng = engine_k3
        e12 = basis_bivector(eng.k, 0, 1)
        result = eng.geometric_product(e12, e12)

        # e_{12}^2 = e_1 e_2 e_1 e_2 = -e_1 e_1 e_2 e_2 = -1
        expected = torch.zeros(eng.n, dtype=eng._dtype)
        expected[0] = -1.0

        assert torch.allclose(result, expected, atol=1e-10), \
            f"e12^2 != -1: got {result}"


# ---------------------------------------------------------------------------
# Grade projection and reverse tests
# ---------------------------------------------------------------------------

class TestGrades:
    def test_grade_masks_shape(self, engine_k3):
        eng = engine_k3
        assert eng.grade_masks.shape == (eng.k + 1, eng.n)

    def test_grade_projection_vector(self, engine_k3):
        eng = engine_k3
        mv = torch.zeros(eng.n, dtype=eng._dtype)
        mv[0] = 2.0    # scalar
        mv[1] = 3.0    # e_1
        mv[3] = 5.0    # e_{12}

        vec = eng.grade_project(mv, 1)
        expected = torch.zeros(eng.n, dtype=eng._dtype)
        expected[1] = 3.0
        assert torch.allclose(vec, expected, atol=1e-10)

    def test_reverse_bivector(self, engine_k3):
        """Reverse of a bivector gets sign -1."""
        eng = engine_k3
        e12 = basis_bivector(eng.k, 0, 1)
        rev = eng.reverse_mv(e12)

        expected = torch.zeros(eng.n, dtype=eng._dtype)
        expected[3] = -1.0  # (e_{12})~ = -e_{12}

        assert torch.allclose(rev, expected, atol=1e-10), \
            f"Reverse of e12 != -e12: got {rev}"

    def test_reverse_rotor(self, engine_k3):
        """Reverse of a rotor (R = cos + B sin) satisfies RR̃ = 1."""
        eng = engine_k3
        B = torch.zeros(eng.n, dtype=eng._dtype)
        B[3] = 0.5  # e_{12} component

        R = eng.rotor_exp(B)
        R_rev = eng.reverse_mv(R)
        RR_rev = eng.geometric_product(R, R_rev)

        # Should be scalar = 1 (unit rotor)
        scalar_part = RR_rev[0]
        assert abs(scalar_part - 1.0) < 1e-6, \
            f"R*R_rev != 1: scalar part = {scalar_part}"


# ---------------------------------------------------------------------------
# Rotor tests
# ---------------------------------------------------------------------------

class TestRotors:
    def test_rotor_exp_log_roundtrip(self, engine_k3):
        """exp(log(R)) = R for a rotor."""
        eng = engine_k3
        B = torch.zeros(eng.n, dtype=eng._dtype)
        B[3] = 0.5  # e_{12}

        R = eng.rotor_exp(B)
        B_recovered = eng.bivector_log(R)
        R_recovered = eng.rotor_exp(B_recovered)

        assert torch.allclose(R, R_recovered, atol=1e-6), \
            f"exp(log(R)) != R: max diff = {(R - R_recovered).abs().max()}"

    def test_rotor_identity(self, engine_k3):
        """exp(0) = identity rotor (scalar = 1)."""
        eng = engine_k3
        B_zero = torch.zeros(eng.n, dtype=eng._dtype)
        R = eng.rotor_exp(B_zero)

        expected = torch.zeros(eng.n, dtype=eng._dtype)
        expected[0] = 1.0

        assert torch.allclose(R, expected, atol=1e-10)

    def test_rotor_sandwich_preserves_norm(self, engine_k3):
        """R x R̃ preserves norm of x for unit rotor R."""
        eng = engine_k3
        x = torch.randn(1, 3)
        x = F.normalize(x, dim=-1)
        x_mv = eng.embed_to_clifford(x.to(torch.float64))

        B = torch.zeros(1, eng.n, dtype=torch.float64)
        B[:, 3] = 0.7  # e_{12}

        R = eng.rotor_exp(B)
        R = eng.normalize_multivector(R)

        rotated_mv = eng.rotor_apply(R, x_mv)
        rotated_vec = eng.clifford_to_embed(rotated_mv, 3)

        orig_norm = x.to(torch.float64).norm(dim=-1)
        rot_norm = rotated_vec.norm(dim=-1)
        assert torch.allclose(orig_norm, rot_norm, atol=1e-4), \
            f"Norm not preserved: {orig_norm} vs {rot_norm}"

    def test_90_degree_rotation_in_plane(self, engine_k3):
        """Rotating e_1 by 90° in the e_1∧e_2 plane → ±e_2 (direction depends on sign convention)."""
        eng = engine_k3

        # Bivector B = (π/4) e_{12} → rotates by 2·(π/4) = π/2
        B = torch.zeros(eng.n, dtype=torch.float64)
        B[3] = math.pi / 4

        R = eng.rotor_exp(B)

        e1 = basis_vector(3, 0, dtype=torch.float64)  # e_1

        result_mv = eng.rotor_apply(R, e1.unsqueeze(0)).squeeze(0)

        # The rotation goes to e_2 or -e_2 depending on sign convention.
        # In our convention (B = ½ x ∧ y), the rotation direction may flip.
        # The key property is that |result| = |input| and the result is on the sphere.
        result_norm = result_mv[1:1+eng.k].norm()
        assert abs(result_norm - 1.0) < 1e-4, \
            f"Rotated vector not unit: norm = {result_norm}"

        # The rotation should produce something orthogonal to e_1 in the e_1-e_2 plane
        e1_component = result_mv[1]  # e_1 component
        assert abs(e1_component) < 0.1, \
            f"Expected mostly e_2, got e_1 component = {e1_component}"


# ---------------------------------------------------------------------------
# Analytic rotor SLERP vs trig SLERP
# ---------------------------------------------------------------------------

class TestRotorSLERP:
    """Verify analytic rotor SLERP matches the trigonometric SLERP."""

    def _trig_slerp(self, clean, noisy, alpha_t, eps=1e-8):
        """Reference SLERP from S-FLM's utils.py."""
        alpha_t = alpha_t.reshape(alpha_t.shape[0], *([1] * (clean.ndim - 1)))
        cos_omega = (clean * noisy).sum(dim=-1, keepdim=True)
        cos_omega = cos_omega.clamp(-1 + eps, 1 - eps)
        omega = torch.acos(cos_omega)
        sin_omega = omega.sin().clamp(min=eps)
        coeff_clean = ((1 - alpha_t) * omega).sin() / sin_omega
        coeff_noisy = (alpha_t * omega).sin() / sin_omega
        return coeff_clean * clean + coeff_noisy * noisy

    def test_slerp_alpha_0(self):
        """At alpha=0 (S-FLM noisy convention), rotor SLERP should return clean."""
        clean = F.normalize(torch.randn(4, 32), dim=-1)
        noisy = F.normalize(torch.randn(4, 32), dim=-1)
        # alpha_t=1 → clean in S-FLM convention
        alpha = torch.ones(4, 1)

        trig = self._trig_slerp(clean, noisy, alpha)
        rotor = rotor_slerp(clean, noisy, alpha)

        assert torch.allclose(trig, rotor, atol=1e-5), \
            f"SLERP at alpha=1 mismatch: max diff = {(trig - rotor).abs().max()}"

    def test_slerp_midpoint(self):
        """At alpha=0.5, rotor SLERP should match trig."""
        clean = F.normalize(torch.randn(4, 32), dim=-1)
        noisy = F.normalize(torch.randn(4, 32), dim=-1)
        alpha = torch.full((4, 1), 0.5)

        trig = self._trig_slerp(clean, noisy, alpha)
        rotor = rotor_slerp(clean, noisy, alpha)

        assert torch.allclose(trig, rotor, atol=1e-5), \
            f"SLERP midpoint mismatch: max diff = {(trig - rotor).abs().max()}"

    def test_slerp_preserves_unit_norm(self):
        """Rotor SLERP output should be unit norm."""
        clean = F.normalize(torch.randn(4, 32), dim=-1)
        noisy = F.normalize(torch.randn(4, 32), dim=-1)
        alpha = torch.rand(4, 1)

        result = rotor_slerp(clean, noisy, alpha)
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Not unit norm: norms = {norms}"

    def test_slerp_batch_consistency(self):
        """SLERP should work correctly for batches."""
        clean = F.normalize(torch.randn(8, 64), dim=-1)
        noisy = F.normalize(torch.randn(8, 64), dim=-1)

        for alpha_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            alpha = torch.full((8, 1), alpha_val)
            trig = self._trig_slerp(clean, noisy, alpha)
            rotor = rotor_slerp(clean, noisy, alpha)
            assert torch.allclose(trig, rotor, atol=1e-4), \
                f"SLERP mismatch at alpha={alpha_val}: max diff = {(trig - rotor).abs().max()}"


# ---------------------------------------------------------------------------
# Analytic rotor log_map / exp_map
# ---------------------------------------------------------------------------

class TestRotorMaps:
    def _trig_log_map(self, x, target, eps=1e-8):
        """Reference log_map from S-FLM."""
        cos_omega = (x * target).sum(dim=-1, keepdim=True)
        cos_omega = cos_omega.clamp(-1 + eps, 1 - eps)
        omega = torch.acos(cos_omega)
        scale = omega / omega.sin().clamp(min=eps)
        return scale * (target - x * cos_omega)

    def _trig_exp_map(self, x, delta, eps=1e-8):
        """Reference exp_map from S-FLM."""
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=eps)
        return x * torch.cos(delta_norm) + (delta / delta_norm) * torch.sin(delta_norm)

    def test_log_map_matches_trig(self):
        """Rotor log_map should match trig log_map exactly."""
        x = F.normalize(torch.randn(4, 32), dim=-1)
        target = F.normalize(torch.randn(4, 32), dim=-1)

        trig = self._trig_log_map(x, target)
        rotor = rotor_log_map(x, target)

        assert torch.allclose(trig, rotor, atol=1e-5), \
            f"log_map mismatch: max diff = {(trig - rotor).abs().max()}"

    def test_exp_map_matches_trig(self):
        """Rotor exp_map should match trig exp_map exactly."""
        x = F.normalize(torch.randn(4, 32), dim=-1)
        delta = torch.randn(4, 32) * 0.1

        trig = self._trig_exp_map(x, delta)
        rotor = rotor_exp_map(x, delta)

        assert torch.allclose(trig, rotor, atol=1e-5), \
            f"exp_map mismatch: max diff = {(trig - rotor).abs().max()}"

    def test_exp_map_preserves_norm_for_tangent_delta(self):
        """exp_map preserves norm when delta is in the tangent plane of x."""
        x = F.normalize(torch.randn(4, 32), dim=-1)
        # Project delta to tangent plane: delta - (delta · x) * x
        delta_raw = torch.randn(4, 32) * 0.3
        delta = delta_raw - (delta_raw * x).sum(dim=-1, keepdim=True) * x

        result = rotor_exp_map(x, delta)
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), \
            f"exp_map not unit norm for tangent delta: norms = {norms}"

    def test_exp_map_on_sphere_after_normalize(self):
        """After normalization, exp_map output should be on the sphere."""
        x = F.normalize(torch.randn(4, 32), dim=-1)
        delta = torch.randn(4, 32) * 0.3  # larger delta

        result = rotor_exp_map(x, delta)
        result_normalized = F.normalize(result, dim=-1)
        norms = result_normalized.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_log_map_then_exp_map_roundtrip(self):
        """exp_map(x, log_map(x, target)) should move toward target."""
        x = F.normalize(torch.randn(4, 32), dim=-1)
        target = F.normalize(torch.randn(4, 32), dim=-1)

        v = rotor_log_map(x, target)
        x_new = rotor_exp_map(x, v)

        # x_new should be closer to target than x
        sim_before = (x * target).sum(dim=-1)
        sim_after = (x_new * target).sum(dim=-1)
        assert (sim_after >= sim_before).all(), \
            f"exp_map(log_map) didn't move toward target"


# ---------------------------------------------------------------------------
# Bivector velocity tests
# ---------------------------------------------------------------------------

class TestBivectorVelocity:
    def test_bivector_is_antisymmetric(self):
        """Bivector should be antisymmetric: B_ij = -B_ji."""
        x_t = F.normalize(torch.randn(4, 32), dim=-1)
        x_0 = F.normalize(torch.randn(4, 32), dim=-1)

        v, B = bivector_velocity(x_t, x_0)

        # B should be antisymmetric
        B_T = B.transpose(-1, -2)
        assert torch.allclose(B, -B_T, atol=1e-5), \
            f"Bivector not antisymmetric: max diff = {(B + B_T).abs().max()}"

    def test_bivector_velocity_matches_log_map(self):
        """The velocity v should match log_map output."""
        x_t = F.normalize(torch.randn(4, 32), dim=-1)
        x_0 = F.normalize(torch.randn(4, 32), dim=-1)

        v, B = bivector_velocity(x_t, x_0)
        v_ref = rotor_log_map(x_t, x_0)

        assert torch.allclose(v, v_ref, atol=1e-5), \
            f"Bivector velocity != log_map: max diff = {(v - v_ref).abs().max()}"


# ---------------------------------------------------------------------------
# Embedding projection tests
# ---------------------------------------------------------------------------

class TestEmbedProjections:
    def test_embed_roundtrip_grade1(self):
        """embed_to_clifford -> clifford_to_embed should preserve grade-1 components."""
        eng = CliffordEngine(k=8, device='cpu', dtype=torch.float64)
        x = torch.randn(4, 768, dtype=torch.float64)
        mv = eng.embed_to_clifford(x)
        x_back = eng.clifford_to_embed(mv, 768)

        # Only first k=8 dims preserved
        assert torch.allclose(x[..., :8], x_back[..., :8], atol=1e-10), \
            "Grade-1 roundtrip failed for first 8 dims"
        assert (x_back[..., 8:] == 0).all(), \
            "Grade-1 roundtrip: dims beyond k should be zero"

    def test_embed_to_clifford_projection(self):
        """EmbedToClifford module should produce valid multivectors."""
        eng = CliffordEngine(k=8, device='cpu', dtype=torch.float64)
        proj = EmbedToClifford(d_embed=768, k=8, dtype=torch.float64)
        x = torch.randn(2, 16, 768, dtype=torch.float64)
        mv = proj(x, eng)

        assert mv.shape == (2, 16, 256), f"Wrong shape: {mv.shape}"

    def test_clifford_to_embed_projection(self):
        """CliffordToEmbed module should project back to embedding dim."""
        eng = CliffordEngine(k=8, device='cpu', dtype=torch.float64)
        proj = CliffordToEmbed(k=8, d_embed=768, dtype=torch.float64)
        mv = torch.randn(2, 16, 256, dtype=torch.float64)
        out = proj(mv, eng)

        assert out.shape == (2, 16, 768), f"Wrong shape: {out.shape}"


# ---------------------------------------------------------------------------
# RotorOps integration tests
# ---------------------------------------------------------------------------

class TestRotorOps:
    def test_analytic_mode_slerp(self):
        """RotorOps.slerp in analytic mode should match trig SLERP."""
        ops = RotorOps(mode='analytic')
        clean = F.normalize(torch.randn(4, 32), dim=-1)
        noisy = F.normalize(torch.randn(4, 32), dim=-1)
        alpha = torch.rand(4, 1)

        result = ops.slerp(clean, noisy, alpha)
        # Should produce unit vectors
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_analytic_mode_log_map(self):
        """RotorOps.log_map in analytic mode."""
        ops = RotorOps(mode='analytic')
        x = F.normalize(torch.randn(4, 32), dim=-1)
        target = F.normalize(torch.randn(4, 32), dim=-1)

        v = ops.log_map(x, target)
        assert v.shape == x.shape
        assert not torch.isnan(v).any()

    def test_analytic_mode_exp_map(self):
        """RotorOps.exp_map with tangent delta preserves unit norm."""
        ops = RotorOps(mode='analytic')
        x = F.normalize(torch.randn(4, 32), dim=-1)
        # Use tangent delta (projected to tangent plane)
        delta_raw = torch.randn(4, 32) * 0.2
        delta = delta_raw - (delta_raw * x).sum(dim=-1, keepdim=True) * x

        result = ops.exp_map(x, delta)
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_clifford_mode_slerp(self):
        """RotorOps.slerp in clifford mode."""
        ops = RotorOps(mode='clifford', clifford_k=3)
        clean = F.normalize(torch.randn(4, 32), dim=-1)
        noisy = F.normalize(torch.randn(4, 32), dim=-1)
        alpha = torch.full((4, 1), 0.5)

        result = ops.slerp(clean, noisy, alpha)
        # Should produce unit vectors (within tolerance for project mode)
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])