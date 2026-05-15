# GWS Future Plan

## Current Status

**GWS stays experimental.** 7/7 seeds on CFS Cl(4,0,0) with AdamW: baseline wins by 32.5%.

The core finding: grade-wise gradient scaling *fights* with AdamW's adaptive moments. Adam already adapts per-parameter LR through m/v estimates, so externally scaling gradients is redundant at best and harmful at worst (it distorts the moment statistics).

The earlier CFS ablation "win" (30.8% over cosine) used a standalone training loop with raw SGD-style updates — no adaptive optimizer. The FNO ablation showed no difference because the model was too shallow for grade dynamics to matter.

## What Works

1. **Rotor schedule > cosine.** The rotor interpolation gives a slightly different decay curve that beats plain cosine by ~19% on the flow CFS model. This is the "free" gain — no grade separation needed.
2. **Grade dynamics are real.** The diagnostic confirmed gradient norm divergence across grades (quadvector 4.3x vs trivector 1.2x on Cl(4,0,0)). The problem exists; the current solution just doesn't address it correctly.
3. **Phase direction matters.** The original `GWScheduler` had the phase backwards — it was advancing decay for higher grades instead of delaying it. Fixed now: `cos(π·(1-δ)·progress)` with δ = offset/(2π).

## What Doesn't Work

1. **Gradient scaling + AdamW.** The fundamental issue. AdamW's second-moment (v) estimate absorbs the grade scaling — after a few steps the v estimate adjusts and the effective step size is the same as without scaling. But the m estimate gets distorted during the adjustment period, causing transient instability.
2. **CliffordFNO2d as a testbed.** Too simple — only 8 spectral conv weights in a ParameterList. Grade dynamics don't develop enough to matter.
3. **Large phase offsets.** The staggered offsets (0, 1.26, 2.51, 3.77, 5.03 radians) are too aggressive. Even with the corrected formula they cause the stretch factors to be very different per grade. Small offsets (0, 0.3, 0.6, 0.9) work better but the effect is still tiny.

## Ideas to Explore

### 1. Schedule-Level Integration (not gradient-level)

Instead of scaling gradients, modify the optimizer's LR per param-group. AdamW already supports per-group LR. Set up two groups:

- **Scalar params**: standard cosine LR
- **MV params**: delayed cosine LR (same stretch trick, but as `lr` in `param_groups`)

This way AdamW's moments are computed on *unscaled* gradients, and the LR schedule handles the grade-wise pacing. The key insight: let AdamW do its per-parameter adaptation on clean gradients, then use the schedule to set the overall pace per grade.

**Risk:** AdamW's v estimate still equalizes effective LR over time, so the schedule difference might wash out. But at least the moments won't be distorted.

### 2. Separate AdamW State per Grade

Run a separate optimizer per grade group, each with its own LR schedule. This gives truly independent moment estimates — grade-0's v estimate doesn't get polluted by grade-4's different gradient scale.

```python
opt_scalar = AdamW(scalar_params, lr=cosine_lr(step))
opt_mv = AdamW(mv_params, lr=delayed_cosine_lr(step, grade_offsets))
```

**Risk:** More optimizer state (2x or 5x memory). But the V100 has 32GB, so this is feasible for the model sizes we run.

### 3. Grade-Aware Weight Decay Only

Don't touch the gradient or LR at all. Instead, apply grade-wise weight decay:

- Scalar params: standard WD (0.01)
- Higher-grade params: lower WD (0.001–0.005)

The intuition: higher-grade components need more freedom to explore because their gradient signal is noisier. Lower weight decay prevents premature regularization of the components that haven't converged yet.

This is orthogonal to AdamW — it doesn't fight the moment estimates. It's also dead simple to implement (just different `weight_decay` per param group).

**Risk:** Might not be enough to make a real difference.

### 4. GWS with SGD / SignSGD

Re-run the comparison with plain SGD + momentum (no adaptive moments). This is the setting where the original ablation showed a win. If GWS+SGD beats Cosine+SGD, then the contribution is "GWS is a replacement for AdamW's adaptive moments in Clifford networks" — you don't *need* Adam if you have grade-wise scheduling.

**Risk:** SGD likely loses to AdamW overall, so this is more of an academic point than a practical improvement.

### 5. Warmup-Phase Separation

Instead of continuous grade-wise scaling, use a two-phase approach:

1. **Phase 1 (0–T/3):** Standard training, all grades same LR. Let AdamW learn the gradient statistics.
2. **Phase 2 (T/3–T):** Apply grade-wise scaling based on the observed gradient norms from Phase 1. Higher-norm grades get lower LR (inverse scaling).

This avoids the "fighting AdamW" problem because Phase 1 lets AdamW calibrate, and Phase 2 only applies when the moments are stable.

**Risk:** The transition between phases could cause a discontinuity. Need to ramp the scaling in smoothly.

### 6. Gradient Norm Equalization (Pre-AdamW)

Before feeding gradients to AdamW, rescale each grade's gradients so their norms are equal. This is a form of gradient normalization — it doesn't change the direction, just the magnitude per grade. AdamW then sees balanced gradients and doesn't over-adapt to the high-norm grades.

```python
for grade, params in grade_groups.items():
    total_norm = sum(p.grad.norm()**2 for p in params) ** 0.5
    target_norm = global_mean_norm
    scale = target_norm / max(total_norm, 1e-8)
    for p in params:
        p.grad.mul_(scale)
```

**Risk:** Same fundamental problem — AdamW's v estimate will absorb the normalization within a few steps. But it might help in the early training phase where the norm divergence is most harmful.

### 7. Clifford-Native Optimizer

Design an optimizer that's aware of multivector structure. Key ideas:

- Maintain separate momentum buffers per grade (like idea #2 but built into the optimizer)
- Use geometric product-based update rules instead of element-wise
- Apply grade projection to the momentum itself

This is the most ambitious path but also the most publishable. A "CliffordAdam" optimizer that naturally handles the grade dynamics would be a genuine contribution.

**Risk:** High engineering effort, unclear if it actually converges better. Start with the simple per-grade-AdamW (idea #2) and see if separating the states helps before going full geometric.

## Priority Order

1. **Idea #1 (Schedule-level LR)** — Easiest to implement, most likely to work, tests the core hypothesis.
2. **Idea #3 (Grade-wise weight decay)** — Trivial to implement, orthogonal to AdamW, worth a quick test.
3. **Idea #2 (Separate AdamW per grade)** — More work but directly addresses the moment-pollution issue.
4. **Idea #5 (Two-phase warmup)** — Moderate effort, requires tuning the transition point.
5. **Idea #6 (Gradient norm equalization)** — Quick to test, but likely same AdamW absorption problem.
6. **Idea #4 (GWS+SGD)** — Academic validation, not practical.
7. **Idea #7 (Clifford-native optimizer)** — Long-term research direction, only if simpler ideas show promise.

## Key Question to Answer First

**Does per-grade LR scheduling (without gradient scaling) help with AdamW?**

If yes → the contribution is "grade-wise LR scheduling for Clifford networks" (simple, practical, publishable).
If no → the problem is deeper than the schedule, and we need a Clifford-aware optimizer (harder, more novel, but riskier).
