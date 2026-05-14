"""
RHF (Rotor Hyperspherical Flow) algorithm variant.

Subclasses SFM and replaces trigonometric sphere operations with
rotor-based equivalents from rotor_utils.py:

  - utils.slerp → rotor_utils.rotor_slerp  (rotor sandwich product)
  - utils.log_map → rotor_utils.rotor_log_map  (bivector logarithm)
  - utils.exp_map → rotor_utils.rotor_exp_map  (rotor exponential)
  - utils.sphere_normalize → rotor_utils.sphere_normalize  (unchanged)

The RHF variant is numerically equivalent to S-FLM's SFM in analytic mode,
but explicitly expresses operations in rotor algebra for:
  1. Cleaner mathematical structure for future extensions
  2. Natural pathway to CFS (Clifford Flow on Sphere) with multivector ops
  3. Bivector velocity fields that capture rotation-plane information

Two RHF modes:
  - 'analytic': Drop-in replacement, same numerics as S-FLM (default)
  - 'clifford': Full Clifford algebra operations via CliffordEngine (CFS)
"""

import torch
import torch.nn.functional as F

# Import S-FLM base class
try:
    from gaflowlm.algo import SFM
    from gaflowlm import utils
    from gaflowlm import flm_utils
except ImportError:
    # Fallback for direct script execution
    import algo as sfm_algo
    SFM = sfm_algo.SFM
    import utils

from gaflowlm.rotor_utils import (
    rotor_slerp,
    rotor_log_map,
    rotor_exp_map,
    RotorOps,
    bivector_velocity,
    sphere_normalize,
)


class RHFSFM(SFM):
    """Rotor Hyperspherical Flow: SFM with rotor-based sphere operations.

    Drop-in replacement for SFM. Same training and sampling interface.
    Only difference: slerp, log_map, exp_map use rotor algebra internally.

    Config additions (via algo section):
      algo.rhf_mode: 'analytic' (default) or 'clifford'
      algo.rhf_clifford_k: k for Cl(k,0,0) when rhf_mode='clifford' (default: 8)
      algo.rhf_compute_bivector: if True, compute bivector velocity during training
                                  for downstream CFS extensions (default: False)
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

        # RHF configuration
        rhf_config = getattr(config, 'algo', config) if hasattr(config, 'algo') else config
        self.rhf_mode = getattr(rhf_config, 'rhf_mode', 'analytic')
        self.rhf_clifford_k = getattr(rhf_config, 'rhf_clifford_k', 8)
        self.rhf_compute_bivector = getattr(rhf_config, 'rhf_compute_bivector', False)

        # Initialize rotor ops (lazy — device/dtype set on first call)
        self._rotor_ops = None
        self._bivector_cache = None

    @property
    def rotor_ops(self):
        """Lazy-initialized RotorOps."""
        if self._rotor_ops is None:
            device = str(self.device) if hasattr(self, 'device') else 'cpu'
            dtype = torch.float64 if self.config.algo.slerp_precision == 'float64' else torch.float32
            self._rotor_ops = RotorOps(
                mode=self.rhf_mode,
                clifford_k=self.rhf_clifford_k,
                device=device,
                dtype=dtype,
            )
        return self._rotor_ops

    def _slerp(self, clean, noisy, alpha_t):
        """Override SFM._slerp with rotor-based version."""
        # Match S-FLM's float64 option
        orig_dtype = None
        if hasattr(self, 'config') and getattr(self.config.algo, 'slerp_precision', None) == 'float64':
            orig_dtype = clean.dtype
            alpha_t = alpha_t.to(torch.float64)
            clean = clean.to(torch.float64)
            noisy = noisy.to(torch.float64)

        out = rotor_slerp(clean, noisy, alpha_t, eps=self.eps)

        if orig_dtype is not None:
            out = out.to(orig_dtype)
        return out

    def _sample_prior(self, e_clean):
        """Override SFM._sample_prior — uses rotor sphere_normalize."""
        e_noisy = torch.randn_like(e_clean)
        return sphere_normalize(e_noisy)

    def compute_bivector_velocity(self, x_t, x_0):
        """Compute bivector velocity field (for CFS extensions).

        Returns the standard tangent vector PLUS the bivector encoding
        the rotation plane structure. The bivector can be used by
        Clifford Frame Attention or other multivector operations.

        Args:
            x_t: [B, L, d] current point on sphere.
            x_0: [B, L, d] target point on sphere.

        Returns:
            v: [B, L, d] tangent vector (same as log_map).
            B: [B, L, d, d] bivector = ½(v ∧ x_t - x_t ∧ v)
        """
        return bivector_velocity(x_t, x_0, eps=self.eps)


# ---------------------------------------------------------------------------
# RHF sampler: wraps SFMSampler with rotor exp_map
# ---------------------------------------------------------------------------

class RHFSampler:
    """Rotor Hyperspherical Flow sampler.

    Same interface as SFMSampler but uses rotor exp_map for Euler steps.

    Usage:
        sampler = RHFSampler.from_config(config, model)
        state = sampler.init_state(model, num_samples)
        while not state.done:
            state = sampler.step(model, state)
        tokens = state.xt
    """

    def __init__(self, sfm_sampler, rhf_mode='analytic', clifford_k=8):
        """Wrap an existing SFMSampler with rotor operations.

        Args:
            sfm_sampler: SFMSampler instance to wrap.
            rhf_mode: 'analytic' or 'clifford'.
            clifford_k: k for Cl(k,0,0) when rhf_mode='clifford'.
        """
        self._sampler = sfm_sampler
        self._rotor_ops = RotorOps(mode=rhf_mode, clifford_k=clifford_k)
        self.rhf_mode = rhf_mode

    @classmethod
    def from_config(cls, config, model):
        """Create RHFSampler from config and model."""
        from gaflowlm.samplers import build_sampler
        sfm_sampler = build_sampler(config, model)
        rhf_mode = getattr(config.algo, 'rhf_mode', 'analytic')
        clifford_k = getattr(config.algo, 'rhf_clifford_k', 8)
        return cls(sfm_sampler, rhf_mode=rhf_mode, clifford_k=clifford_k)

    def init_state(self, model, num_samples, **kwargs):
        """Initialize sampling state (delegates to wrapped sampler)."""
        return self._sampler.init_state(model, num_samples, **kwargs)

    def step(self, model, state):
        """One sampling step using rotor exp_map instead of trig exp_map.

        Replaces the exp_map call in SFMSampler with rotor_exp_map.
        Everything else (forward pass, velocity computation, step size)
        remains identical.
        """
        from gaflowlm import utils as sflm_utils
        from gaflowlm.samplers import sfm_compute_velocity, sfm_step_size

        num_steps = len(state.t_schedule) - 1
        is_last_step = (state.step_idx == num_steps - 1)

        _, alpha_t = model.noise(state.t_schedule[state.step_idx])
        sigma_t = model._sigma_from_alphat(alpha_t).reshape(-1, 1)

        from gaflowlm.samplers import SFMContext
        context = SFMContext(temperature=self._sampler.temperature)
        log_p = model.forward(xt=state.xt, sigma=sigma_t, context=context)

        if self._sampler.use_float64:
            log_p = log_p.to(torch.float64)
        state.nfe += 1

        if self._sampler.p_nucleus != 1.0 or self._sampler.top_k != -1:
            log_p = sflm_utils.top_k_top_p_filtering(log_p,
                top_k=self._sampler.top_k, top_p=self._sampler.p_nucleus
            ).log_softmax(-1)

        if is_last_step:
            return self._sampler._last_step_decode(state, log_p)

        # Velocity computation (same as SFM)
        log_p_window = log_p[:, state.start_idx:]
        E = sflm_utils.sphere_normalize(
            model.backbone.sphere_embed.weight.detach())
        x = state.xt[:, state.start_idx:].to(E)

        if self._sampler.slerp_float64:
            E = E.to(torch.float64)
            x = x.to(torch.float64)

        if self._sampler.top_k_velocity > 0:
            log_p_v, E = self._sampler._select_topk(
                log_p_window, E, self._sampler.top_k_velocity)
        else:
            log_p_v = log_p_window

        vel = sfm_compute_velocity(x, E, log_p_v,
                                    mode=self._sampler.velocity, eps=self._sampler.eps)

        # RHF KEY CHANGE: use rotor exp_map instead of trig exp_map
        dt = sfm_step_size(alpha_t,
                            model.noise(state.t_schedule[state.step_idx + 1])[1],
                            self._sampler.invert_time_convention, self._sampler.eps)
        x_new = rotor_exp_map(x, dt * vel, eps=self._sampler.eps)

        # Re-normalize (rotor exp_map is approximately norm-preserving for
        # tangent vectors, but we normalize to stay on the sphere exactly)
        x_new = sphere_normalize(x_new)

        state.xt[:, state.start_idx:] = x_new.to(state.xt.dtype)
        self._sampler._project_prefix(
            state.xt, state.prefix_embeds, state.prefix_lengths)
        state.step_idx += 1
        return state