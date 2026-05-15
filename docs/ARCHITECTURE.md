# Architecture Description

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    GAFlowLM                              │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  Variant A   │    │  Variant B   │                   │
│  │  RHF         │    │  CFS         │                   │
│  │              │    │              │                    │
│  │ Vector-only  │    │ Multivector  │                   │
│  │ Rotor ops    │    │ CFA + rotors │                   │
│  │ S-arch back  │    │ CARE pos enc │                   │
│  └──────────────┘    └──────────────┘                   │
│                                                         │
│  Shared: Clifford algebra engine, noise schedules,      │
│          training loop, evaluation harness               │
└─────────────────────────────────────────────────────────┘
```

## Component Design

### 1. Clifford Algebra Engine (`clifford_engine.py`)

Core GA operations implemented in PyTorch. Does NOT use the full 2^d expansion. Instead, we work in Cl(k,0,0) for k ≪ d and project to/from d-dim space.

**Key design choice: k=8 or k=16**

For k=8: Cl(8,0,0) has 2^8 = 256 basis elements. Grades: 1 scalar + 8 vectors + 28 bivectors + 56 trivectors + 70 quadvectors + ... + 1 pseudoscalar.

For k=16: Cl(16,0,0) has 2^16 = 65536 basis elements — too many. We restrict to grades 0,1,2 only: 1 + 16 + 120 = 137 components.

**Recommendation:** k=8 with full multivector (all grades), or k=16 with grades 0-2 only. Start with k=8.

```python
class CliffordEngine:
    """Core geometric algebra operations for Cl(k,0,0)"""
    
    k: int  # algebra dimension
    
    def geometric_product(A, B) -> Multivector:
        """AB = A·B + A∧B (inner + outer product decomposition)"""
    
    def outer_product(A, B) -> Multivector:
        """A∧B — grade-raising product"""
    
    def inner_product(A, B) -> Multivector:
        """A·B — grade-lowering product"""
    
    def reverse(M) -> Multivector:
        """M̃ — reverse of multivector (negate odd-grade swaps)"""
    
    def rotor_from_vectors(a, b) -> Rotor:
        """R = exp(B/2) where B = a∧b/‖a∧b‖, rotation from a to b"""
    
    def rotor_apply(R, x) -> Multivector:
        """R x R̃ — sandwich product (rotation)"""
    
    def rotor_exp(B) -> Rotor:
        """exp(B) for bivector B — rotor exponential"""
    
    def bivector_log(R) -> Bivector:
        """log(R) for rotor R — rotor logarithm"""
    
    def sandwich(R, M) -> Multivector:
        """R M R̃ — apply rotor to multivector"""
    
    def commutator(A, B) -> Multivector:
        """[A,B] = AB - BA — Lie bracket"""
    
    def scalar_part(M) -> float:
        """⟨M⟩₀ — extract grade-0 component"""
    
    def grade_project(M, r) -> Multivector:
        """⟨M⟩ᵣ — project to grade-r"""
    
    def spinor_norm(M) -> float:
        """⟨M M̃⟩₀ — scalar part of reverse product"""
    
    def normalize_multivector(M) -> Multivector:
        """M / √⟨M M̃⟩₀ — normalize on multivector sphere"""
```

### 2. Projection Layer (`projection.py`)

Bridge between d-dimensional embedding space and k-dimensional Clifford space.

```python
class EmbeddingToClifford:
    """Project d-dim token embedding to Cl(k,0,0) multivector"""
    # Linear projection: R^d -> Cl(k,0,0)
    # Decompose: vector part (grade-1) + bivector part (grade-2)
    #   vec_proj: Linear(d, k)        — d -> k for grade-1
    #   biv_proj: Linear(d, k*(k-1)/2) — d -> k*(k-1)/2 for grade-2
    #   scalar_bias: Parameter(1)      — learned grade-0

class CliffordToEmbedding:
    """Project Cl(k,0,0) multivector back to d-dim space"""
    #   vec_proj: Linear(k, d)         — grade-1 -> d
    #   biv_proj: Linear(k*(k-1)/2, d) — grade-2 -> d
    #   scalar_proj: Linear(1, d)      — grade-0 -> d
    # Output = vec_proj(⟨M⟩₁) + biv_proj(⟨M⟩₂) + scalar_proj(⟨M⟩₀)
```

### 3. Variant A: Rotor Hyperspherical Flow (`rhf/`)

```
Input: token IDs x₀ ∈ V^L
  │
  ├── Embedding: E[x₀] → ê_{x₀} ∈ S^{d-1}   (unit-norm, same as S-FLM)
  │
  ├── Forward process (noise):
  │     z₀ ~ U(S^{d-1})                          (uniform noise)
  │     B = (1/2) z₀ ∧ ê_{x₀}                   (bivector: rotation plane)
  │     R_{αₜ} = exp(αₜ B)                       (rotor at noise level αₜ)
  │     zₜ = R_{αₜ} ê_{x₀} R̃_{αₜ}              (rotor sandwich — no SLERP)
  │
  ├── Velocity prediction (S-arch with rotor ops):
  │     B_θ(zₜ, t) = S_Arch_Rotor(zₜ, t)         (predict bivector field)
  │     Marginal: B_θ = Σ_v p_{1|t}(v|zₜ) · (zₜ ∧ ê_v)
  │
  ├── Sampling (Euler steps):
  │     R_Δθ = exp(B_θ · Δt)                      (rotor from predicted bivector)
  │     z_{t+Δt} = R_Δθ zₜ R̃_Δθ                  (rotor application — no exp_map)
  │
  └── Decoding:
        logits_v = z₁ · ê_v                         (dot product with embeddings)
        x̂ = argmax_v logits_v
```

**S-arch modifications for RHF:**

1. Replace `log_map(x, target)` with `bivector_field(x, target) = x ∧ target` (outer product gives rotation plane)
2. Replace `exp_map(x, delta)` with `rotor_apply(exp(B·h), x)` where B is the predicted bivector
3. Replace `justnorm(h + γ(B - h))` residual with `R_γ = exp(γ B_residual); h_out = R_γ h_in R̃_γ` (exact spherical blend)
4. Time conditioning: unchanged (adaLN-style gamma gates)
5. Attention: unchanged (standard QKV) — this is the minimal-change variant

**Training loss:** Cross-entropy on logits (same as S-FLM). The bivector field is an intermediate representation; the model still predicts vocabulary logits.

### 4. Variant B: Clifford Flow-Matching on the Sphere (`cfs/`)

```
Input: token IDs x₀ ∈ V^L
  │
  ├── Multivector Embedding:
  │     E[x₀] → M₀ ∈ Cl(k,0,0)                   (multivector, projected from d-dim)
  │     M₀ = Normalize(EmbedToCliff(E[x₀]))
  │
  ├── CARE Position Encoding:
  │     M̃_i = R_{pos}(i) M_i R̃_{pos}(i)          (rotor position embedding)
  │     R_{pos}(i) = exp(½ θ_x(i) B_x + ½ θ_y(i) B_y)  (learned bivectors)
  │
  ├── Forward process (noise):
  │     N₀ ~ U(S^{spinor})                         (uniform on multivector sphere)
  │     B = bivector_log(rotor_from_multivectors(N₀, M₁))
  │     R_{αₜ} = exp(αₜ B)
  │     Mₜ = R_{αₜ} M₁ R̃_{αₜ}                    (rotor sandwich)
  │
  ├── CFA Transformer (N blocks):
  │     Each block:
  │       1. Clifford Frame Attention
  │          - Q,K,V: multivector linear projections
  │          - Attention: softmax(⟨Q_i, K_j⟩₀ / √d_k)
  │          - Messages: GeometricBilinear(Q_i, V_j) = Q_i · V_j + Q_i ∧ V_j
  │          - Higher-order: M^{(2)} = Σ_j m_{ij}; m^{(3)} ∝ M^{(2)} · M'^{(2)}
  │       2. MLP (FFN) on multivector components
  │       3. Rotor residual: M_out = R_γ M_in R̃_γ
  │       4. Multivector normalization: M ← M / √⟨M M̃⟩₀
  │
  ├── Velocity prediction:
  │     B_θ(Mₜ, t) = CFA_Transformer(Mₜ, t)       (predicted bivector)
  │
  ├── Sampling (Euler steps):
  │     R_Δt = exp(B_θ · Δt)
  │     M_{t+Δt} = R_Δt Mₜ R̃_Δt
  │
  └── Decoding:
        M_d = CliffordToEmbedding(M₁)              (project back to d-dim)
        logits_v = M_d · ê_v                         (dot product with embeddings)
        x̂ = argmax_v logits_v
```

#### As-implemented status (May 2026)

The diagram above is the *design intent*. The current implementation in
`gaflowlm/models/cfs_model.py` differs at the boundary layers:

| Component                  | Designed                                  | As implemented in `cfs_model.py`          |
|----------------------------|-------------------------------------------|-------------------------------------------|
| Token → multivector embed  | `EmbedToCliff` (grade-routing projection) | `nn.Linear(hidden_size, mv_dim)`          |
| Multivector → token decode | `CliffordToEmbedding` (grade gather)      | `nn.Linear(mv_dim, hidden_size)`          |
| Velocity head              | bivector-only output                      | `nn.Linear(mv_dim, mv_dim)`               |
| Multivector normalization  | `M / √⟨M M̃⟩₀` (spinor norm)             | `nn.LayerNorm(mv_dim)` (flat-axis)        |
| CARE positional encoding   | rotor sandwich via Cayley tensor          | rotor sandwich via Cayley tensor          |
| CFA attention              | bilinear via geometric product            | bilinear via geometric product            |

The middle of the network is Clifford-aware (CARE, CFA, geometric
product messages). The IO layers and the multivector norm treat the
2^k axis as a flat vector and let the model learn whatever routing
into blade slots works. The grade-aware modules `EmbedToClifford`,
`CliffordToEmbed`, and `engine.normalize_multivector` exist in
`gaflowlm/clifford/engine.py` and are exercised by tests, but no
trained model wires them up.

**Implications for the research story:**

- *"CFS"* as currently trained is a partial Clifford commitment. Grades
  are a coordinate convention at the boundary; only the middle layers
  impose grade-mixing structure.
- This explains why the `embed_to_clifford` blade-indexing bugfix
  (commit `f153671`) produced bit-identical training results in
  `gws_comparison.py` — the trained model never calls the engine
  helper that was fixed.
- It also weakens the a-priori motivation for GWS on the current
  architecture: per-grade learning rate scaling assumes grades behave
  differently during training, but a Linear can route gradient mass
  to whichever blade slots minimize loss. The fair-ablation result
  (see `docs/GWS_RESEARCH_DESIGN.md` §9 and
  `experiments/gws/cfs_ablation_fair.py`) is consistent with that
  picture.

**Planned ablation (not yet run):** swap the four boundary layers for
their grade-aware counterparts, keeping CARE and CFA unchanged, and
re-run the S-FLM comparison. The variants to A/B:

- *CFS-current*: nn.Linear IO + LayerNorm (today's behavior, will be
  the baseline number from the first Sudoku run).
- *CFS-grade-aware*: `EmbedToClifford` + `CliffordToEmbed` +
  `engine.normalize_multivector` + a bivector-restricted velocity
  head. Tests whether structural grade commitment, not just
  coordinate-level naming, carries weight.

Do not switch architectures before the first real-data baseline run.
The variant is an ablation, not a fix.

### 5. Noise Schedule (`noise_schedules.py`)

Direct adaptation from S-FLM with GA-specific modifications:

- **LogLinear, CosineSquared** base schedules: unchanged (αₜ is a scalar noise level)
- **Truncation:** The bound `α*(δ) ≈ (2/π) arcsin(...)` must be re-derived for the multivector sphere (different volume formula). For Variant A (vector-only), the bound is identical to S-FLM.
- **Adaptive schedule:** Same mechanism (fit loss profile, invert CDF), but loss is measured on multivector cross-entropy, which may have different curvature.

### 6. Training Loop (`train.py`)

```
For each batch:
  1. Sample t ~ Uniform(0, 1)
  2. Compute αₜ from noise schedule (with truncation)
  3. Compute bivector B = z₀ ∧ ê_{x₀}      [RHF] or N₀ ∧ M₁         [CFS]
  4. Apply rotor: zₜ = R_{αₜ} z₁ R̃_{αₜ}
  5. Forward pass: B_θ = model(zₜ, t)        [predict bivector]
  6. Compute logits from zₜ + B_θ            [decode to vocabulary]
  7. CE loss against true tokens x₀
  8. Backward + optimizer step
  9. [Optional] Renormalize embedding weights
```

### 7. Evaluation (`evaluate.py`)

Same benchmarks as S-FLM:
- **GSM8K** (primary): accuracy at T=1 and T=0.1
- **Sudoku** (hard): accuracy
- **TinyGSM**: perplexity
- **OpenWebText**: perplexity

Additional probes:
- **Norm drift**: track ‖z‖ - 1 over training
- **Grade utilization**: for CFS, measure energy in each grade: ‖⟨M⟩ᵣ‖ / ‖M‖
- **Rotor composition quality**: compare multi-step sampling error for rotor vs. exp-map
- **Numerical stability**: count NaN/Inf events