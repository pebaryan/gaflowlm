# GAFlowLM: Geometric Algebra Flow Language Modeling

Research investigation into extending hyperspherical flow-based language modeling with Clifford algebra (geometric algebra), replacing trigonometric SLERP/log-map/exp-map primitives with rotor sandwich products, multivector representations, and Clifford Frame Attention.

## Status

**Phase 0: Research investigation — COMPLETE**
**Phase 1: RHF implementation — NEXT**
**Phase 2: CFS implementation — COMPLETE**

See [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md) for current task list.

## Benchmark Scope

The original S-FLM training runs used multi-GPU jobs. This workspace uses a
tiered validation plan instead:

- **Reference S-FLM protocol:** original multi-GPU benchmark setup
- **Local smoke tests:** synthetic overfit and GSM8K-test reconstruction
- **Local real-data checks:** TinyGSM subset or GSM8K-test fixture on an
  available GPU
- **mi25 shared-slot checks:** same smoke tests, only using the GPU when the
  llama.cpp server is not occupying it

See [Experiment Plan](docs/EXPERIMENTS.md) for the detailed benchmark ladder.

## Quick Links

| Document | Description |
|----------|-------------|
| [Research Proposal](docs/RESEARCH_PROPOSAL.md) | Extended abstract with architecture, math, and hypotheses |
| [Literature Review & Gap Analysis](docs/LITERATURE_REVIEW.md) | Summaries of 4 foundational papers + confirmed gap |
| [Architecture Description](docs/ARCHITECTURE.md) | Component-level design for RHF and CFS variants |
| [Pseudocode](docs/PSEUDOCODE.md) | Rotor-based velocity field and sampling algorithms |
| [Mathematical Insights](docs/MATHEMATICAL_INSIGHTS.md) | Key identities (SLERP = grade-1 projection of rotor sandwich, etc.) |
| [Experiment Plan](docs/EXPERIMENTS.md) | Benchmark matrix, ablations, evaluation protocol |
| [GWS Research Design](docs/GWS_RESEARCH_DESIGN.md) | Grade-wise geometric scheduling track for Clifford models |
| [Paper Outline](docs/PAPER_OUTLINE.md) | Targeted arXiv submission structure |
| [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md) | Fork strategy, GA libraries, integration plan, current status |

## Core Idea

S-FLM (Deschenaux & Gulcehre, 2026) models language on the hypersphere S^{d-1}, using SLERP for noise interpolation and log-map/exp-map for velocity fields. These are Riemannian operations patched onto a Euclidean Transformer. We show they are **projections of native Clifford algebra operations**:

| S-FLM (trig) | GAFlowLM (Clifford) | What's gained |
|---------------|---------------------|---------------|
| SLERP(z₁, z₀, α) | Rα z₁ R̃α (rotor sandwich) | No acos/sin, algebraic composition, norm preserved |
| log_map(zₜ, z₁) | zₜ ∧ z₁ (outer product) | Full rotation plane, not just tangent vector |
| exp_map(z, δ) | exp(Bδ) z exp(-Bδ) | Exact geodesic, no renormalization needed |
| justnorm(h + γ(B-h)) | Rγ h R̃γ | Exact interpolation, not 1st-order approximation |
| QKV attention | CFA geometric bilinear | Grade interactions, 3-body messages "for free" |

## Two Variants

### Variant A: Rotor Hyperspherical Flow (RHF)
Minimal intervention — swap flow primitives only, keep S-arch backbone. Replaces SLERP with rotor sandwich, log-map with bivector outer product, exp-map with rotor application.

### Variant B: Clifford Flow-Matching on the Sphere (CFS)
Maximal intervention — multivector embeddings + CFA attention + CARE position encoding. Token representations span scalar + vector + bivector + higher grades.

### Upstream GWS Track
The upstream repo also now contains **Grade-Wise Geometric Scheduling (GWS)**:
an optimizer/scheduling line of work for Clifford neural networks. It is
related to CFS and RHF, but it is not part of the core S-FLM benchmark ladder
above. Treat it as a separate research track for optimization experiments.

## Key References

| Paper | Venue | Role |
|-------|-------|------|
| S-FLM (Deschenaux & Gulcehre) | arXiv 2605.11125 | Baseline we extend |
| Clifford Flows (Alesiani & Maruyama) | NeurIPS 2024 ML4PS | Normalizing flows over multivectors |
| GAFL (Wagner et al.) | arXiv 2411.05238 | CFA architecture + flow matching |
| CARE (Sriram et al.) | NeurIPS 2025 UniReps | Rotor embeddings generalizing RoPE |
