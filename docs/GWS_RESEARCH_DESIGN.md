# Grade-Wise Geometric Scheduling (GWS)

A novel learning rate scheduling framework for Clifford/geometric algebra neural networks
that decomposes gradient updates by multivector grade and applies independent rotor-driven
schedules to each grade component.

## 1. Core Idea

Standard LR schedulers treat all parameters identically — a single scalar lr(t) scales
every gradient component equally. In Clifford neural networks, this is geometrically
lossy: multivector grades carry structurally different information (scalar = magnitude,
vector = direction, bivector = rotation plane, etc.), and they converge at different
rates during training.

GWS decomposes the gradient into grade components and applies per-grade schedules
driven by rotor interpolation on the scheduling algebra Cl(k_s, 0, 0).

## 2. Mathematical Formulation

### 2.1 Grade Decomposition

Given a multivector parameter M = Σ_g ⟨M⟩_g with grades g ∈ {0, 1, 2, ..., k},
and its gradient ∇L = Σ_g ⟨∇L⟩_g, the update rule becomes:

  M ← M - Σ_g η_g(t) ⟨∇L⟩_g

where η_g(t) is the grade-specific learning rate at step t.

### 2.2 Rotor Schedules

Each η_g(t) is parameterized by a rotor interpolation on a scheduling algebra
Cl(k_s, 0, 0) with k_s scheduling dimensions (typically k_s = 2 or 3):

  R_g(t) = exp(-θ_g(t) B_g / 2)

where:
- B_g is a unit bivector in the scheduling algebra (grade-2 element)
- θ_g(t) ∈ [0, π/2] is the scheduling angle, monotonically increasing
- The grade-specific LR is: η_g(t) = η_g(0) · ⟨R_g(t)⟩_0

The scalar part of the rotor gives cos(θ_g(t)), which recovers cosine annealing
as a special case when all grades share the same bivector and angle.

### 2.3 Multi-Dimensional Generalization

Unlike scalar cosine annealing (which traces a 1D arc), rotors can compose
rotations in different planes simultaneously. With k_s = 3:

  R(t) = R_{12}(t) · R_{23}(t) · R_{13}(t)

This allows the scheduler to explore non-trivial paths in the LR space —
e.g., the vector-grade LR can decrease along one rotation plane while the
bivector-grade LR follows a different path through another plane.

### 2.4 Scheduling Angle Functions

θ_g(t) can follow several profiles:

- **Linear** (cosine-like): θ_g(t) = (π/2) · t/T
- **Warmup+decay**: θ_g(t) = (π/2) · smoothstep(t/T) where smoothstep
  provides a warmup period before decay
- **Cyclic**: θ_g(t) = (π/2) · (1 - cos(2π · cycle(t))) for restarts
- **Learned**: θ_g(t) parameterized by a small MLP (meta-learning the schedule)

## 3. Experimental Design

### 3.1 Baselines

| Scheduler | Description |
|-----------|-------------|
| Constant | Fixed lr throughout training |
| Cosine annealing | η(t) = η_0 · cos(πt/2T) — single scalar schedule |
| Cosine with warmup | Warmup + cosine annealing (standard transformer recipe) |
| Step decay | LR drops by factor γ at fixed intervals |
| OneCycle | Super-convergence schedule (Smith 2018) |

### 3.2 GWS Variants

| Variant | Description |
|---------|-------------|
| GWS-Uniform | Same rotor angle for all grades (reduces to multi-dim cosine) |
| GWS-Linear | Per-grade linear θ_g(t) with hand-tuned phase offsets |
| GWS-Warmup | Per-grade warmup+decay schedules |
| GWS-Cyclic | Per-grade cyclic schedules (different cycle lengths per grade) |
| GWS-Learned | Per-grade θ_g(t) from small meta-network |

### 3.3 Test Architectures

| Architecture | Source | Why |
|-------------|--------|-----|
| Clifford Neural Layers (PDE) | Brandstetter et al. 2023 (ICLR) | Published baseline, canonical Clifford architecture |
| Geometric Clifford Algebra Networks | Ruhe et al. 2023 (ICML) | Group-action layers, different structure to test generality |
| CFS (ours) | gaflowlm | Domain-specific application, language modeling |

### 3.4 Tasks

| Task | Architecture | Metric |
|------|-------------|--------|
| 2D Navier-Stokes PDE | Clifford Neural Layers | Rollout MSE |
| 2D Maxwell equations | Clifford Neural Layers | Field prediction MSE |
| Vector field prediction | GCAN | Rotation equivariance error |
| Language modeling (synthetic) | CFS | Eval loss, perplexity |
| Language modeling (TinyGSM) | CFS | Eval loss, token accuracy |

### 3.5 Ablations

1. **Grade decomposition vs. scalar**: Is per-grade scheduling better than scalar?
2. **Rotor vs. cosine**: Does the multi-plane rotation add value over independent
   per-grade cosine schedules?
3. **Number of scheduling dimensions k_s**: How does k_s ∈ {1, 2, 3} affect outcomes?
4. **Grade coupling**: What happens when bivectors B_g share planes vs. are orthogonal?
5. **Phase offsets**: Does staggering grade schedules (e.g., bivector decays slower
   than scalar) help, and by how much?

### 3.6 Metrics

- Final validation loss / task metric
- Convergence speed (steps to 90% of best baseline score)
- Training stability (loss variance across last 10% of steps)
- Parameter efficiency (same budget, better schedule = better final metric)
- Inference cost: zero (scheduler only affects training)

## 4. Expected Results & Claims

### 4.1 Primary Claim

Grade-wise scheduling outperforms scalar scheduling on Clifford neural networks,
with the gap increasing as the model uses more multivector grades.

### 4.2 Secondary Claims

- Rotor parameterization is a natural generalization of cosine annealing that
  recovers it as a special case (k_s=1, uniform grades)
- Grade-wise schedules are most beneficial when grades have different convergence
  dynamics (empirically verifiable by plotting per-grade gradient norms over time)
- The overhead is negligible: grade decomposition is O(k) on the gradient, and
  the scheduling algebra is small (k_s << k)

### 4.3 Anticipated Counterarguments

- "This is just per-parameter-group LR with extra steps" — No: grade decomposition
  is structurally determined by the algebra, not arbitrary grouping. The rotor
  parameterization also couples the schedules geometrically, which independent
  per-group schedules don't do.
- "Cosine annealing already works fine" — Yes for scalar networks. For multivector
  networks, there's constructive and destructive interference between grade
  updates under a shared schedule that grade-wise scheduling resolves.

## 5. Implementation Plan

### Phase 0: Scheduler core (gaflowlm/gws/)
- [ ] GradeDecomposer: split multivector parameters/grads by grade
- [ ] RotorScheduler: compute R_g(t) for each grade given schedule config
- [ ] GWSOptimizer: wrap AdamW with per-grade LR application
- [ ] Baseline wrappers: CosineAnnealing, OneCycle, StepDecay for fair comparison

### Phase 1: Diagnostics (before full experiments)
- [ ] Log per-grade gradient norms during standard (scalar-scheduled) training
- [ ] Check if grades have different convergence dynamics — if they don't,
  grade-wise scheduling may not help much
- [ ] Plot grade-wise loss contributions to confirm the hypothesis

### Phase 2: Experiments on published architectures
- [ ] Clone and integrate with Clifford Neural Layers codebase
- [ ] Run PDE tasks with GWS vs. baselines
- [ ] Run GCAN tasks with GWS vs. baselines

### Phase 3: Application to CFS
- [ ] Apply GWS to CFS flow training
- [ ] Compare with current CFS baseline (which uses a single cosine schedule)

## 6. Paper Outline

**Title:** Grade-Wise Geometric Scheduling for Clifford Neural Networks

**Abstract:** We introduce grade-wise geometric scheduling (GWS), a learning rate
scheduling framework for geometric algebra neural networks that decomposes
gradient updates by multivector grade and applies independent rotor-driven
schedules to each grade. Unlike scalar schedulers, GWS respects the algebraic
structure of multivector parameters, allowing different geometric components
(scalar, vector, bivector, etc.) to converge at their natural rates. We show
that rotor interpolation generalizes cosine annealing to multi-dimensional
schedule paths and demonstrate improvements on PDE modeling, vector field
prediction, and language modeling tasks.

1. Introduction
2. Background: Clifford algebras, cosine annealing, LR scheduling
3. Method: Grade decomposition, rotor schedules, GWS optimizer
4. Experiments: PDE tasks, vector fields, CFS language model
5. Analysis: Per-grade dynamics, ablations, when does it matter?
6. Related work: LR scheduling, Clifford NNs, meta-learning schedules
7. Conclusion

## 7. Venue Targets

- **ICML / NeurIPS** (Optimization track) — if experiments are strong
- **ICLR** (Optimization or Representation learning) — natural fit
- **Geometric Science of Information (GSI)** — smaller but perfectly targeted
- **Topical workshop** at ICML/NeurIPS on geometric deep learning — lower bar, good feedback

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Grade-wise scheduling shows no improvement | Diagnostics phase (1) validates the hypothesis before committing to full experiments. If grades converge at the same rate, pivot to rotor-interpolation-as-generalized-cosine as the contribution |
| Only helps on one architecture | Test on 2+ published architectures before CFS |
| Reviewers see it as "just per-group LR" | Emphasize: (1) structure-determined grouping, not arbitrary; (2) rotor coupling between grades; (3) cosine annealing as a special case |
| Implementation overhead on external codebases | Keep GWS as a standalone optimizer wrapper; minimal integration needed |
