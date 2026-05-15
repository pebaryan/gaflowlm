# Grade-Wise Scheduler (GWS / Rotor-GWS)

An auxiliary learning-rate scheduling track for Clifford/geometric algebra
networks that separates the base schedule from grade-specific phase shifts.
The current goal is not to replace the main CFS/RHF benchmark plan, but to
probe whether grade separation improves stability and convergence in Clifford
models.

## 1. Core Idea

Standard LR schedulers treat all parameters identically. In Clifford neural networks,
multivector grades often behave differently during training, so the hypothesis is that
staggering the schedule by grade can improve stability or reduce variance.

The practical implementation is deliberately simple:
- keep a global cosine base schedule
- add per-grade phase offsets
- apply the resulting grade factors to gradient updates

Rotor language is retained as the conceptual framing, but the working version should
prefer phase-shifted cosine factors over elaborate rotor objects.

## 2. Mathematical Formulation

### 2.1 Grade Decomposition

Given a multivector parameter M = Σ_g ⟨M⟩_g with grades g ∈ {0, 1, 2, ..., k},
and its gradient ∇L = Σ_g ⟨∇L⟩_g, the update rule becomes:

  M ← M - η_base(t) Σ_g φ_g(t) ⟨∇L⟩_g

where:
- η_base(t) is the global cosine schedule
- φ_g(t) is the grade-specific phase factor
- the grade factors are small multiplicative modulations, not a full rewrite of AdamW

### 2.2 Rotor-Inspired Phase Shifts

Each φ_g(t) can be parameterized by a rotor-inspired phase shift, but the
stable implementation should start with a cosine factor:

  φ_g(t) = 0.5 * (1 + cos(π t / T + s_g))

where:
- s_g is the per-grade phase shift
- the phase shifts can be fixed or learnable
- if all s_g are equal, GWS collapses to the base cosine schedule

### 2.3 Multi-Phase Generalization

The first useful axis is the number of distinct phases:
- 1 phase: uniform schedule
- 2 phases: light separation
- 1 phase per grade: full grade separation

This is the smallest clean ablation and should come before any richer rotor
composition story.

### 2.4 Phase Function Options

The phase shifts can be:

- **Fixed**: hand-tuned offsets, best for first pass
- **Learnable**: one scalar per grade, optimized with the main model

Warmup and decay stay in the base scheduler. The grade offsets should stay
small and local.

## 3. Experimental Design

### 3.1 Baselines

| Scheduler | Description |
|-----------|-------------|
| Constant | Fixed lr throughout training |
| Cosine annealing | Single scalar schedule |
| Cosine with warmup | Warmup + cosine annealing |
| Step decay | LR drops by factor γ at fixed intervals |

### 3.2 GWS Variants

| Variant | Description |
|---------|-------------|
| GWS-Uniform | Same phase for all grades, should behave like cosine |
| GWS-2Phase | Two phase groups, to test whether coarse separation helps |
| GWS-Full | One phase per grade, fixed offsets |
| GWS-Learned | One learnable phase scalar per grade |

### 3.3 Test Architecture

| Architecture | Source | Why |
|-------------|--------|-----|
| CFS (ours) | gaflowlm | The current multivector language-model target |

### 3.4 Tasks

| Task | Architecture | Metric |
|------|-------------|--------|
| Language modeling (synthetic) | CFS | Eval loss, perplexity |
| Language modeling (TinyGSM) | CFS | Eval loss, token accuracy |
| Learning dynamics | CFS | Seed variance, per-grade grad norms |

### 3.5 Ablations

1. **Uniform vs. phase-shifted**: Does any grade separation help over one shared schedule?
2. **Two phases vs. full per-grade**: Is coarse separation enough, or do we need one phase per grade?
3. **Fixed vs. learnable offsets**: Can tiny learned phase parameters improve stability?
4. **Base cosine vs. base cosine + phases**: Is the gain really from the offsets?
5. **Per-grade gradient norms**: Do the schedules match the observed convergence dynamics?

### 3.6 Metrics

- Final validation loss / task metric
- Convergence speed (steps to 90% of best baseline score)
- Training stability (loss variance across last 10% of steps)
- Seed variance across runs
- Per-grade gradient norm plots
- Parameter efficiency (same budget, better schedule = better final metric)
- Inference cost: zero (scheduler only affects training)

## 4. Expected Results & Claims

### 4.1 Primary Claim

Grade-wise scheduling can reduce training variance or improve convergence
stability on Clifford neural networks, especially when multivector grades
converge at different rates.

### 4.2 Secondary Claims

- A shared cosine base schedule plus small phase offsets is a stable, low-cost
  way to test grade separation.
- Per-grade gradient norms should explain when the method helps.
- If the gains are real, the best story is stability and repeatability first,
  raw accuracy second.

### 4.3 Anticipated Counterarguments

- "This is just per-parameter-group LR" — The answer should be framed as
  structured grade separation, not arbitrary parameter grouping.
- "Cosine annealing already works fine" — Maybe, which is why the first
  experiments should focus on seed variance and convergence smoothness.

## 5. Implementation Plan

### Phase 0: Scheduler core (gaflowlm/gws/)
- [ ] GradeDecomposer: split multivector parameters/grads by grade axis
- [ ] Base cosine schedule plus per-grade phase offsets
- [ ] GWSOptimizer: wrap AdamW with per-grade gradient scaling
- [ ] Optional learnable phase offsets as tiny scalar parameters

### Phase 1: Diagnostics (before full experiments)
- [ ] Log per-grade gradient norms during scalar-scheduled training
- [ ] Measure seed variance on a small CFS run
- [ ] Plot grade-wise gradient magnitudes and update magnitudes
- [ ] Check whether 1, 2, or full per-grade phases matter most

### Phase 2: Experiments on CFS only
- [ ] Run synthetic CFS with cosine vs. GWS-Uniform vs. GWS-Full
- [ ] Run TinyGSM or GSM8K-test smoke checks with the same comparison
- [ ] Add a medium-depth CFS setting to check whether the effect grows with depth

### Phase 3: Optional expansion
- [ ] If the CFS result is stable, try the same schedule on another Clifford model
- [ ] Only then consider a broader architecture sweep

## 6. Paper Outline

**Title:** Grade-Wise Scheduler for Clifford Neural Networks

**Abstract:** We introduce GWS, an auxiliary learning-rate scheduling
framework for geometric algebra neural networks that adds small per-grade
phase offsets on top of a shared cosine base schedule. The method is designed
to test whether multivector grades benefit from staggered optimization, and
whether that effect shows up as lower seed variance, smoother training, or
faster convergence on CFS-style models. The paper contribution is primarily
about optimization stability and diagnostics, not a broad architecture sweep.

1. Introduction
2. Background: Clifford algebras, cosine annealing, LR scheduling
3. Method: Grade decomposition, phase-shifted schedules, GWS optimizer
4. Experiments: CFS synthetic and tiny real-data runs
5. Analysis: Per-grade dynamics, seed variance, when does it matter?
6. Related work: LR scheduling, Clifford NNs, optimization stability
7. Conclusion

## 7. Venue Targets

- **ICLR / workshop** if the stability result is clear
- **Geometric Science of Information (GSI)** if the contribution stays narrow
- **Optimization workshop** at ICML/NeurIPS if the seed-variance story is strong

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Grade-wise scheduling shows no improvement | Keep the auxiliary claim narrow: stability diagnostics only, or drop the paper entirely |
| Only helps on one architecture | Do not overgeneralize; present it as a CFS-specific optimization study |
| Reviewers see it as "just per-group LR" | Emphasize structured grade separation and seed-variance evidence |
| Implementation overhead on external codebases | Keep the first version as a thin optimizer wrapper with cosine base + offsets |

## 9. Status (May 2026): controlled benchmarks to date

Three benchmarks have been run on the synthetic CFS flow objective
(k=4, h=64, 2 blocks, 2000 steps, batch 8, seq_len 32, random tokens).
Each one tests a different thing — and the prior two were confounded
by LR-magnitude asymmetries that pulled in opposite directions:

| Script | Verdict | What it actually tested |
|--------|---------|-------------------------|
| `experiments/gws/gws_comparison.py` (7 seeds via `CFSAlgorithm`) | cosine wins 7/7, +32% | GWS had **less** effective per-step LR than baseline — `GWScheduler` applies grade factors ≤1 on top of the same `CosineAnnealingLR` baseline already uses |
| `experiments/gws/cfs_ablation.py` (5 seeds, manual loop) | GWS wins 5/5, +25% | GWS had **more** effective LR than baseline — its optimizer LR was pinned to `eta_max` while the baseline cosine decayed normally |
| `experiments/gws/cfs_ablation_fair.py` (5 seeds, controlled) | **GWS loses 4/5, -6%** | Only the per-grade *distribution* differs; total per-blade LR budget identical to baseline (`blade_scale` renormalized to sum to `n` every step) |

The third script holds the LR controls and finds no positive effect
from grade staggering on this benchmark. See commits `f153671`
(engine fixes that made earlier numbers re-runnable) through
`b2bc83a` (the fair-ablation result) for the full trail.

### What this does and doesn't mean

- **Does mean:** the existing in-repo evidence does not support grade
  staggering as a useful default on synthetic CFS flow training. The
  prior "30.8% over cosine" finding (commit `cef7a5e`) and upstream's
  later "GWS stays EXPERIMENTAL" tag (commit `87becd6`) are both
  consistent with this picture once the LR confounder is removed.
- **Does not mean:** that grade staggering can't help anywhere. The
  benchmark uses small models (~46k params), short schedules (2k
  steps), synthetic random tokens, and a fixed phase-offset choice
  (`[i * 0.15 for i in range(n_grades)]`). A genuine positive result
  would need at least one of: (a) a real dataset (TinyGSM / GSM8K-test
  / Sudoku) so the loss landscape isn't synthetic; (b) deeper models
  where per-grade utilization varies more; (c) longer schedules so the
  cosine-vs-grade decay interaction isn't dominant; (d) a phase-offset
  sweep rather than a single guess.

### Recommendation

Keep GWS in the codebase as an experimental optimizer track, but the
public README and any paper draft should describe its current status
honestly: it is an idea that has not yet shown a confounder-free
improvement on any benchmark in this repo. The most useful next
experiment is `cfs_ablation_fair.py`-style ablation on a real dataset
or a meaningfully deeper Clifford model, not another synthetic run at
the same scale.
