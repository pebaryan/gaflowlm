# Research Proposal: Geometric Algebra Flow Language Modeling (GAFlowLM)

**Authors:** [TBD]
**Date:** May 2026
**Status:** Pre-implementation research investigation

---

## 1. Problem Statement

Current flow-based language models (S-FLM) operate on the hypersphere S^{d-1} using Riemannian operations (SLERP, log-map, exp-map) patched onto a Euclidean Transformer backbone. These trigonometric operations are:
- **Numerically unstable** near poles (acos/sin blow-ups)
- **Algebraically awkward** (composition of geodesic steps isn't a geodesic)
- **Structurally incomplete** (tangent vectors rather than bivector generators; post-hoc normalization rather than structural norm preservation)

We propose replacing the entire Riemannian substrate with **Clifford algebra** (geometric algebra), where rotations are native objects — **rotors** — and all flow operations become algebraic products, yielding cleaner equations, better equivariance, and potentially stronger performance on reasoning benchmarks.

---

## 2. Background

Four foundational works motivate this proposal:

1. **S-FLM** (Deschenaux & Gulcehre, 2026): Hyperspherical flow language model on S^{d-1}. SLERP-based noise, log-map velocity, exp-map sampling, S-arch backbone. Baseline we aim to surpass.

2. **Clifford Flows** (Alesiani & Maruyama, 2024): Normalizing flows over multivectors (Clifford NVP). Proves standard Autograd works for Clifford networks. Closed-form log-determinant via scalar-valued scale functions.

3. **GAFL** (Wagner et al., 2024): Flow matching for proteins using Clifford Frame Attention (CFA). Geometric bilinear products, higher-order messages, motor transformations in PGA. Achieves state-of-the-art designability with better secondary structure distribution.

4. **CARE** (Sriram et al., 2025): Rotor position embeddings generalizing RoPE/Spherical RoPE. Sandwich product `R q R^{-1}` preserves grade structure. Unifies Mixed RoPE, Spherical RoPE, and QuatRo as special cases.

**Gap:** No work combines hyperspherical language flows with geometric algebra. GAFlowLM fills this gap.

---

## 3. Architecture Proposals

### Variant A: Rotor Hyperspherical Flow (RHF)

The minimal intervention — replace S-FLM's SLERP/log-map/exp-map with rotor algebra, keeping the S-arch backbone otherwise unchanged.

**Algebra:** Work in Cl(d,0,0) where d is the embedding dimension. Token embeddings ê_v are unit vectors (grade-1 multivectors). Rotors R ∈ Cl⁺(d,0,0) are even-grade multivectors satisfying R R̃ = 1.

**Forward process (noise):**
Instead of SLERP, generate noisy samples via rotor rotation:
```
zₜ = R_{αₜ}(z₀, z₁) z₁ R̃_{αₜ}(z₀, z₁)
```
where `R_{αₜ} = exp(αₜ B)` and `B` is the bivector that rotates z₁ toward z₀. The bivector `B = log(z₁ z₀^†)` is the bivector logarithm, and `αₜ ∈ [0,1]` is the noise schedule.

**Key simplification:** For unit vectors z₀, z₁:
```
B = (1/2) z₁ ∧ z₀    (the outer product gives the rotation plane)
R_{αₜ} = exp((αₜ/2)(z₁ ∧ z₀))
```
Then:
```
zₜ = R_{αₜ} z₁ R̃_{αₜ}
    = cos(αₜ‖ω‖/2) z₁ + sin(αₜ‖ω‖/2) (ẑ × z₁)
```
where `ẑ = (z₀ - (z₁·z₀)z₁)/‖z₀ - (z₁·z₀)z₁‖` and `‖ω‖ = acos(z₁·z₀)`.

This is algebraically identical to SLERP but derived from rotor composition rather than trigonometric functions. The advantage appears in:

**Velocity field (bivector formulation):**
```
vₜ = (Ṙₜ R̃ₜ + Rₜ Ṙ̃ₜ) / 2
    = dBₜ/dt    (the bivector velocity)
```
The rotor velocity is a bivector (grade-2), not a tangent vector. The predicted update is:
```
R_{Δθ} = exp(B_θ(zₜ, t) · Δt)
z_{t+Δt} = R_{Δθ} zₜ R̃_{Δθ}
```
No acos, no division by sin(ω), no exp_map renormalization. The rotor application preserves ‖z‖ by construction.

**Marginal velocity:** The model predicts a bivector field B_θ(zₜ, t) via:
```
B_θ(zₜ, t) = Σ_{v∈V} p^{θ}_{1|t}(x=v|zₜ) · log_{zₜ}(ê_v) ∧ zₜ
```
which replaces S-FLM's log-map marginalization with a bivector-marginalization that naturally captures the rotation plane for each target token.

**S-arch integration:** Replace `log_map(x, target)` with bivector computation `log(x) ∧ target`, and replace `exp_map(x, delta)` with rotor application `R_{Δθ} x R̃_{Δθ}`. The S-arch residual connections become exact spherical blends via rotor interpolation (not the `justnorm` approximation).

### Variant B: Clifford Flow-Matching on the Sphere (CFS)

The maximal intervention — embed tokens as **multivectors** (not just vectors) and use CFA-style attention with geometric products.

**Embedding:** Each token v is represented as a multivector M_v ∈ Cl(d,0,0) with components across all grades:

```
M_v = ⟨M_v⟩₀ + ⟨M_v⟩₁ + ⟨M_v⟩₂ + ... + ⟨M_v⟩ₐ
```

The grade-1 (vector) part is the standard embedding ê_v. The scalar, bivector, and higher-grade parts encode structural/positional information:

- **Grade 0 (scalar):** Learned constant — position-invariant "bias"
- **Grade 1 (vector):** Classical token embedding (the S-FLM component)
- **Grade 2 (bivector):** Orientation / attention plane — encodes which rotation planes are relevant for this token
- **Grade 3+ (higher):** Volumetric / multi-token interaction information

**Attention: CFA for Language**

Replace standard QKV attention with Clifford Frame Attention:

```
Query, Key, Value are multivectors in Cl(d,0,0)
Attention weights: a_{ij} = softmax(⟨Q_i, K_j⟩₀ / √d_k)   (scalar part of geometric product)
Message: m_{ij} = GeometricBilinear(Q_i, K_j, V_j)         (CFA message)
Output: o_i = Σ_j a_{ij} · m_{ij}                           (multivector-valued attention)
```

Where `GeometricBilinear(A, B, C) = (A·B)C + (A∧B)C` uses the geometric product decomposition. This gives **higher-order interactions** "for free" — the outer product `A∧B` captures bivector (area) relationships that are invisible to dot-product attention.

**Higher-order message passing (adapted from GAFL):**
After computing 2-body messages `m_{ij}`, aggregate and compose:
```
M_i^{(2)} = Σ_j m_{ij}                (2-body aggregation)
m_{ijk}^{(3)} ∝ M_i^{(2)} · M_i'^{(2)}   (geometric product of aggregates)
```
This yields 3-body token interactions without additional attention passes.

**Position encoding: CARE rotors**
```
M̃_i = R_{pos}(i) M_i R̃_{pos}(i)
```
where `R_{pos}(i)` is a learned rotor conditioned on position i. This generalizes RoPE to multivector representations.

**Flow matching on multivector sphere:**

The unit-norm constraint generalizes to the **spinor norm**: for a multivector M, the constraint is `⟨M M̃⟩₀ = 1` (scalar part of the reverse product equals 1). This subsumes unit-vector constraint as the grade-1 special case.

Forward process:
```
Mₜ = R_{αₜ} M₁ R̃_{αₜ}     (rotor sandwich product)
```

Velocity field (predicted by CFA Transformer):
```
Ṁₜ = [B_θ(Mₜ, t), Mₜ]    (commutator product — bivector velocity × multivector state)
```

The commutator product `[A,B] = AB - BA` ensures the velocity is a bivector (generator of rotations) and the state stays on the "multivector sphere" `⟨M M̃⟩₀ = 1`.

**Sampling:** Euler step with rotor exponential:
```
M_{t+Δt} = exp(B_θ · Δt) Mₜ exp(-B_θ · Δt)
```

**Decoding:** Project multivector state M₁ back to vocabulary via inner product with grade-1 embedding components:
```
logits_v = ⟨M₁, ê_v⟩₁    (grade-1 inner product with each vocabulary token)
```

### Normalization on the Multivector Sphere

For Variant B, maintaining `⟨M M̃⟩₀ = 1` requires:

1. **Rotor application** (forward process, sampling) — preserves norm by construction
2. **CFA attention** — multivector outputs must be re-normalized. We use spinor normalization: `M ← M / √⟨M M̃⟩₀`
3. **MLP/FFN** — same as CFA; output multivector re-normalized
4. **Residual connections** — spherical blend via rotor: `R_γ = exp(γ B_residual)` applied as `M_out = R_γ M_in R̃_γ`

---

## 4. Testable Hypotheses

### H1: Numerical Stability
**Claim:** Rotor-based SLERP eliminates acos/sin-reciprocal instabilities, giving ≥10x fewer NaN/Inf events during training.
**Test:** Train S-FLM and RHF with identical seeds; count fp64 promotion events and NaN losses over 250k steps.

### H2: Sampling Quality at Low NFE
**Claim:** Rotor Euler steps compose algebraically (`R₃ = R₂R₁`), giving better trajectory quality at ≤16 NFE compared to exp-map steps which approximate.
**Test:** Compare GSM8K accuracy at {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024} NFE for both methods.

### H3: Reasoning Improvement from CFA
**Claim:** Higher-order geometric messages capture 3-token interactions "for free," improving GSM8K by ≥15% relative (from ~18% to ≥21%) at T=1.
**Test:** Ablate CFA vs. standard attention within the CFS architecture on GSM8K.

### H4: Multivector Embedding Quality
**Claim:** Bivector channels (grade-2) encode structural/relational information (negation, antonymy, analogy directions) that pure vector embeddings cannot express in the same dimension, as measured by nearest-neighbor analogy tasks.
**Test:** Train both variants; probe grade-2 channels with relation classification tasks.

### H5: Norm Preservation
**Claim:** Rotor application eliminates norm drift without `justnorm()`. S-FLM accumulates ‖z‖ errors of order 1e-3 per 250k steps; RHF/CFS maintains ‖z‖ = 1.0 exactly.
**Test:** Log ‖z‖-1 averaged over batches during training.

---

## 5. Expected Results & Risk Analysis

### Expected
- RHF (Variant A) should immediately eliminate numerical instabilities and improve low-NFE sampling quality. Expect 5-15% GSM8K improvement at T=1.
- CFS (Variant B) has higher upside but higher variance. CFA is unproven for language; grade-mixing may introduce noise. Expect 10-25% GSM8K improvement if CFA is beneficial, or similar performance if CFA overhead cancels geometric benefits.

### Risks
1. **Memory cost:** Multivector representation in Cl(d,0,0) for d=768 requires 2^768 components — intractable. We must use **sparse multivectors** (only grades 0, 1, 2 projected, not the full algebra). This gives 1 + 768 + 768×767/2 ≈ 295k components — still too many. **Practical approach:** use Cl(k,0,0) for k ≪ d (e.g., k=8 or k=16) for the geometric algebra operations, with a linear projection to/from d-dimensional embeddings.

2. **Computational overhead:** Geometric products are O(k²) per pair of k-dimensional multivectors. For k=16, this is 256 ops per product — comparable to a linear layer. For k=d, this is prohibitive.

3. **Grade mixing:** Higher-grade components may capture noise rather than signal, especially for language where "geometric structure" is less salient than for proteins. Careful regularization may be needed.

4. **Training curriculum:** S-FLM's adaptive noise schedule is critical for performance. A GA version must be adapted; the truncation bound `α*(δ)` depends on the geometry of the multivector sphere, which differs from S^{d-1}.

5. **Scaling:** S-FLM's best result is 18.4% on GSM8K with a 768-dim model. Closing the T=0.1 gap (to ~36%) may require architectural innovations beyond the GA substitution.

---

## 6. Related Work Beyond the Core Four

| Work | Relation |
|------|----------|
| GATr (Buchholz et al., 2023) | General-purpose equivariant Transformer in PGA; architectural inspiration for CFS |
| GCAN (Ruhe et al., 2023) | Geometric Clifford Algebra Networks for dynamical systems; proves equivariance guarantees |
| CliffordNet (Ji, 2026) | End-to-end GA learning for vision; validates feasibility of GA primitives in deep networks |
| HyperSphereDiff (2025) | Hyperspherical diffusion for images; validates spherical noise processes |
| Pustejovsky (2026) | "Functional GA for NLP" — theoretical proposal for GA-based semantics, no implementation |
| nGPT (Meta, 2024) | Normalized Transformer; inspiration for S-arch's unit-norm constraint enforcement |