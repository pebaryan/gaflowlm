# Literature Review & Gap Analysis

## Paper 1: S-FLM — Language Modeling with Hyperspherical Flows

**Authors:** Justin Deschenaux, Caglar Gulcehre (EPFL / Microsoft AI)
**arXiv:** 2605.11125v1, May 2026
**Code:** https://github.com/jdeschena/s-flm

### Summary

S-FLM introduces a latent flow language model operating on the unit hypersphere S^{d-1}. Tokens are embedded as unit-norm vectors, and the forward (noise) process interpolates between data and noise via **Spherical Linear Interpolation (SLERP)**:

```
ψ_{t|1}(z₀, z₁) = SLERP(z₀, z₁, αₜ)
```

where z₀ ~ U(S^{d-1}) and z₁ = ê_{x_l} is the normalized data embedding. The conditional velocity field uses the **Riemannian logarithmic map**:

```
v_{t|1}(zₜ | z₁) = (α̇ₜ / (1 - αₜ)) · log_{zₜ}(z₁)
```

and the marginal velocity marginalizes over vocabulary with posterior weights p^{θ}_{1|t}. Sampling uses **Euler steps with the exponential map** to stay on the sphere:

```
z_{t+Δt} = exp_map(zₜ, h · v_θ(zₜ))
```

where `exp_map(x, δ) = x cos(‖δ‖) + (δ/‖δ‖) sin(‖δ‖)`.

**Critical innovation:** Noise schedule truncation prevents posterior collapse in high dimensions. The truncation bound is derived in closed form:

```
α*(δ) ≈ (2/π) arcsin(√(2 log(2(|V|-1)/δ) / d))
```

For |V|=100k, d=768, δ=0.1: α*=0.125. An adaptive schedule (InfoNoise-inspired) allocates training samples where the loss gradient is largest.

The **S-arch** backbone is a Transformer variant (inspired by nGPT) that maintains all activations on S^{d-1}: residual connections are spherical blends (SLERP approximation), weight matrices have unit-norm rows/columns, and attention uses bias-free QKV with learned softmax scale √d_k. Time conditioning is via per-dimension gates conditioned on timestep (similar to adaLN in DiT).

**Results:** At T=1, S-FLM with top-1 decoding reaches 18.4% on GSM8K with ≤16 NFE, closing the gap to MDLM/Duo. A significant gap remains at T=0.1 (~18% vs ~36%). On hard Sudoku, truncation+adaptive improves accuracy from 14% to 45%.

### Key Limitations (motivation for our work)

1. **SLERP is coordinate-dependent**: The spherical interpolation decomposes into trigonometric functions (sin, cos, acos) that are numerically unstable for near-parallel vectors and don't compose algebraically.
2. **Log-map / exp-map are ad-hoc Riemannian operations**: They are patched onto a fundamentally Euclidean Transformer; the backbone has no intrinsic understanding of rotational geometry.
3. **S-arch maintains unit-norm via explicit normalization** (`justnorm`) after every operation — this is a reset, not a structural guarantee. The residual "spherical blend" `justnorm(A + γ(B - A))` is a first-order approximation to SLERP, losing accuracy for large γ.
4. **No equivariance guarantees**: The architecture treats rotations as external operations, not as native algebraic structures. There's no principled way to enforce rotational equivariance in the Transformer layers themselves.

---

## Paper 2: Clifford Flows — Normalizing Flows over Multivectors

**Authors:** Francesco Alesiani, Takashi Maruyama
**Venue:** NeurIPS 2024 Workshop on ML and the Physical Sciences (OpenReview: 3LdCL2gR7u)

### Summary

Clifford Flows extends Real-NVP to probability distributions over Clifford algebras Cl(R^{p,q,r}). The key transformation is **Clifford NVP**:

```
x_i^{l+1} = x_i^l                                          (identity half)
y_i^{l+1} = y_i^l · exp(s_{i,θ}(x)) + t_{i,φ}(x)           (transformed half)
```

where:
- **s_{i,θ}**: Cl^{p,q,r} → R (scalar-valued scale, embedded as grade-0 component)
- **t_{i,φ}**: Cl^{p,q,r} → Cl^{p,q,r} (multivector-valued translation)
- The product `·` is the **Clifford (geometric) product**, not element-wise multiplication

**Log-determinant (closed form):**
```
ln |det J| = Σᵢ s_{i,θ}(x)
```

This collapses because s outputs a scalar (grade-0), making the Jacobian determinant identical to Real-NVP's — a key practical advantage.

The paper proves (Corollary 3.1) that standard Autograd gradients over Clifford networks coincide with the analytical Clifford gradient. This validates using PyTorch/JAX automatic differentiation for gradient computation without custom differentiators.

**Experimental validation:** On hard-sphere Monte Carlo sampling, Clifford NVP with CGA(4,1,0) shows only -1.99% test log-prob drop vs. Real-NVP's -69.56% and NSF's -235.13%, demonstrating orders-of-magnitude better generalization on geometrically structured data.

### Relevance to GAFlowLM

- Provides the **normalizing flow backbone** for variant (b): Clifford NVP layers can be stacked within the Transformer to produce invertible transformations over multivector representations.
- The closed-form log-det is critical: it means we can train density estimators over multivector latents without expensive Jacobian computation.
- Corollary 3.1 guarantees that standard backpropagation through our geometric product layers gives correct gradients — no custom autograd needed.

---

## Paper 3: GAFL — Geometric Algebra Flow Matching for Protein Design

**Authors:** S. Wagner et al. (HITS)
**arXiv:** 2411.05238, Nov 2024
**Code:** https://github.com/hits-mli/gafl

### Summary

GAFL introduces a generative model for protein backbone design that replaces AlphaFold2's Invariant Point Attention (IPA) with **Clifford Frame Attention (CFA)**, operating in the projective geometric algebra G_{3,0,1}. Key innovations:

1. **PGA features**: Replaces point-valued attention values with multivectors encoding points, lines, planes, and Euclidean frames — all within a single 16-dimensional representation.

2. **Geometric bilinear message passing**: Messages are computed via the **geometric product** (not linear combinations):
   ```
   m_{ij}^{h,p} = GeometricBilinear(T_i^{-1}(T_j V_j^{h,p} T_j^{-1}) T_i, V_i^{h,p})
   ```
   The geometric product of two multivectors A and B produces a new multivector containing all grade interactions: `AB = A·B + A∧B + A×B` (inner + outer + commutator products).

3. **Higher-order message passing**: Aggregated 2-body messages are combined via bilinearity to produce 3-body messages:
   ```
   (Σⱼ m_{ij})(Σₖ m'_{ik}) = Σ_{j,k} m_{ijk}^{(3)}
   ```
   This is free — it reuses existing aggregated messages without additional attention passes.

4. **Relative frame aggregation**: Attention-weighted frame transformations:
   ```
   T_i^{rel} ≡ Σⱼ a_{ij} T_i^{-1} T_j
   ```

**Architecture:** 6 sequential CFA blocks with self-conditioning. Each block: CFA → MLP → Transformer → frame update prediction. The backbone update predicts multivectors R (rotation) and S (translation) via MLP, applied as motor transformations in PGA.

**Flow matching backbone:** Conditional flow on SE(3)^N using geodesic interpolation (analogous to S-FLM's SLERP, but for rigid body frames):
```
ψₜ(T₀|T₁) = exp_{T₀}(t · log_{T₀}(T₁))
```

**Results (PDB, up to 300 residues):**
| Method | Designability↑ | Diversity↓ | Helix | Strand |
|--------|---------------|------------|-------|--------|
| FrameFlow | 0.85 | 0.35 | 0.56 | 0.20 |
| RFdiffusion* | 0.89 | 0.37 | 0.58 | 0.24 |
| **GAFL** | **0.88** | **0.36** | **0.53** | **0.25** |

Ablation: PGA features add +1.4pp designability, higher-order messages add +2.3pp. CFA consistently achieves better secondary structure distribution (close to natural PDB statistics).

### Relevance to GAFlowLM

- **CFA is the attention mechanism we need.** S-FLM's S-arch uses standard dot-product attention that treats vectors as Euclidean. Replacing this with CFA gives us geometrically structured attention where messages are multivectors formed by geometric products.
- **Self-conditioning** (feeding the previous step's prediction back as input) is already present in FrameFlow and directly applicable to language modeling.
- **Higher-order message passing** via geometric product bilinearity gives richer token-token interactions without additional attention passes.

---

## Paper 4: CARE — Clifford Algebraic Rotor Embeddings

**Authors:** Sameeksha Sriram, Ayush Paliwal, Alexander S. Ecker, Chase van de Geijn
**arXiv:** 2511.11665, Nov 2025
**Venue:** NeurIPS 2025 UniReps Workshop

### Summary

CARE generalizes Rotary Position Embeddings (RoPE) to arbitrary dimensions using Clifford rotors acting on multivectors. The hierarchy:

```
Mixed RoPE ⊂ Spherical RoPE ⊂ QuatRo(3D) ⊂ CARE(arbitrary-d)
```

**QuatRo** (quaternion rotary embeddings) parameterizes rotations in 3D using unit quaternions:
```
r_x = exp(a_i^{(x)} i + a_j^{(x)} j + a_k^{(x)} k)
q'_i = r_x r_y q_i r_y^{-1} r_x^{-1}
```

This is the **rotor sandwich product** `R q R^{-1}` — a double-sided multiplication that guarantees the output stays on the same geometric object (preserves grade structure).

**CARE** generalizes to Cl(n,0,0) for arbitrary n:
```
R_α(p_α) = exp(½ θ_α(p_α) B_α)
q̃_i = R_y(p_y) R_x(p_x) q_i R_x(p_x)^{-1} R_y(p_y)^{-1}
```

where B_α ∈ ∧²R^3 is a **bivector** parameterized by a learned axis direction. Each dimension still needs only 3 parameters (the axis direction), even though the multivector representation in Cl(3,0,0) has 8 coefficients per sub-vector.

**Key experimental result (ViT-B on CIFAR100):**

| Method | Top-1 |
|--------|-------|
| Mixed RoPE | 74.3% |
| Spherical RoPE | 74.2% |
| QuatRo | 74.1% |
| **CARE** | **74.8%** |

**Limitations:** CARE is ~2-3x slower than standard RoPE due to geometric product overhead. The 8D sub-vector requirement in Cl(3,0,0) increases memory.

### Relevance to GAFlowLM

- CARE provides the **position encoding mechanism** for our architecture. Instead of adding sinusoidal embeddings to token vectors (S-FLM uses time-conditioned gating), CARE applies rotor sandwich products to multivector token representations, encoding both position and rotational structure natively.
- The sandwich product `R q R^{-1}` is the **exact primitive** we need for noise addition on the sphere: a rotor R_αₜ rotates the clean embedding toward noise, with the rotation angle αₜ controlling the noise level. This replaces SLERP algebraically.
- The finding that orthogonal axes recover Spherical RoPE while parallel axes recover Mixed RoPE means we can smoothly interpolate between these regimes, giving a hyperparameter to tune.

---

## Gap Analysis

### Confirmed Gap: No existing work combines hyperspherical language modeling flows with geometric algebra

After systematic search (arXiv, OpenReview, Google Scholar, Semantic Scholar), we confirm:

1. **S-FLM (2026)** uses hyperspherical flows for language but operates entirely in R^d with trigonometric operations (SLERP, log-map, exp-map). No GA is used.

2. **Clifford Flows (2024)** defines normalizing flows over multivectors but only validates on low-dimensional physics problems (molecular dynamics, hard-sphere MC). No language modeling.

3. **GAFL (2024)** uses geometric algebra for flow matching + CFA, but only for protein structure generation on SE(3)^N. No language modeling, no discrete tokens.

4. **CARE (2025)** brings Clifford rotors to embeddings for vision Transformers, but only for position encoding (not generative modeling).

5. **GATr (2023)** is a general-purpose equivariant Transformer in PGA, but only applied to physics/geometric tasks (n-body, wall-shear-stress). No language modeling.

6. **GCAN (2023)** and **CliffordNet (2026)** are general GA neural architectures, but neither has been applied to language modeling or flow-based generation.

7. **Pustejovsky (2026)** proposes "Functional Geometric Algebra for Natural Language Semantics" (arXiv 2604.25902) — but this is a theoretical/linguistic proposal for semantic representation, not a generative model. No flows, no training, no LM benchmarks.

8. **HyperSphereDiff (2025)** applies hyperspherical diffusion to image generation (class-conditional), not language modeling. No GA.

**Conclusion:** There is a clear, unoccupied intersection between:
- Hyperspherical/rotational language modeling (S-FLM)
- Geometric algebra flow architectures (Clifford Flows, GAFL)
- Geometric algebra token embeddings (CARE)

Our proposed work — **GAFlowLM** — sits precisely at this intersection and is the first to unify all three threads.

### Algebraic Advantages of GA over Trigonometric Operations

| Operation | S-FLM (trigonometric) | GAFlowLM (Clifford) | Advantage |
|-----------|----------------------|---------------------|-----------|
| Forward process (noise) | SLERP: `sin((1-αₜ)ω)/sin(ω)·z₁ + sin(αₜω)/sin(ω)·z₀` | Rotor: `zₜ = R_{αₜ} z₀ R̃_{αₜ}` | Closed-form sandwich product; no acos/sin/sin-reciprocal; numerically stable for all angles |
| Velocity field | Log-map: `log_{zₜ}(z₁) = ω/sin(ω) · (z₁ - zₜ cos ω)` | Bivector field: `vₜ = (Ṙₜ R̃ₜ + Rₜ R̃̇ₜ)/2 · z₀` | No acos, no division by sin(ω); velocity is a bivector (grade-2), naturally living in the tangent algebra |
| Sampling step | Exp-map: `z_{t+Δt} = zₜ cos(‖δ‖) + (δ/‖δ‖) sin(‖δ‖)` | Rotor update: `z_{t+Δt} = R_{Δθ} zₜ R̃_{Δθ}` | No explicit normalization needed; rotor application preserves grade by construction |
| Residual connection | `justnorm(h + γ(B - h))` — approximate SLERP | Spherical blend via bivector interpolation: `R_γ = exp(γ B_mid)` | Exact interpolation, not first-order approximation |
| Position encoding | None (S-arch uses time gates) | CARE rotors: `R_x R_y q_i R_y^{-1} R_x^{-1}` | Rotation-equivariant, generalizes RoPE/Spherical RoPE |
| Attention | Standard QKV dot product | CFA: geometric bilinear messages + higher-order | Captures grade interactions (areas, volumes); 3-body information "for free" |
| Normalization | `justnorm(x) = x/‖x‖` after every operation | Rotor application preserves unit norm by construction (R x R^{-1} preserves ‖x‖ when |R|=1) | Structural guarantee, not post-hoc fix |

### What S-FLM Gets Wrong (from a GA perspective)

1. **The velocity field should be a bivector.** In S-FLM, `vₜ = log_{zₜ}(z₁)` is a tangent vector at zₜ. But rotations are generated by **bivectors** (grade-2 elements), not vectors. The correct geometric structure is `vₜ ∈ ∧²R^d`, and the flow should be `żₜ = vₜ × zₜ` (where × is the commutator product). S-FLM's log-map accidentally recovers the vector projection of this bivector, losing the full algebraic structure.

2. **SLERP and log-map are projections.** SLERP computes the geodesic on S^{d-1} using trigonometric functions that project the bivector exponential onto vector components. The rotor sandwich `R z R̃` is the *original* operation — SLERP is an approximation (valid only for single-plane rotations).

3. **Post-hoc normalization is a band-aid.** S-FLM's `justnorm()` renormalizes after every attention/MLP/residual operation. In GA, rotor application preserves norm by construction (|RzR̃| = |z| when |R| = 1). The norm drift in S-arch signals that the underlying algebra is wrong.

4. **No compositional structure.** Composing two SLERP interpolations doesn't yield a SLERP — you need to project back to the sphere. Composing two rotors is just `R₃ = R₁R₂`, a single multivector multiplication. This matters for multi-step sampling and hierarchical generation.