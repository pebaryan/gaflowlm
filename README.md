# GAFlowLM: Geometric Algebra Flow Language Modeling

Research investigation into extending hyperspherical flow-based language modeling with Clifford algebra (geometric algebra), replacing trigonometric SLERP/log-map/exp-map primitives with rotor sandwich products, multivector representations, and Clifford Frame Attention.

## Status

**Phase 0: Research investigation — COMPLETE**
**Phase 1: RHF implementation — NEXT**
**Phase 2: CFS implementation — COMPLETE**

See [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md) for current task list.

## Quick Links

| Document | Description |
|----------|-------------|
| [Research Proposal](docs/RESEARCH_PROPOSAL.md) | Extended abstract with architecture, math, and hypotheses |
| [Literature Review & Gap Analysis](docs/LITERATURE_REVIEW.md) | Summaries of 4 foundational papers + confirmed gap |
| [Architecture Description](docs/ARCHITECTURE.md) | Component-level design for RHF and CFS variants |
| [Pseudocode](docs/PSEUDOCODE.md) | Rotor-based velocity field and sampling algorithms |
| [Mathematical Insights](docs/MATHEMATICAL_INSIGHTS.md) | Key identities (SLERP = grade-1 projection of rotor sandwich, etc.) |
| [Experiment Plan](docs/EXPERIMENTS.md) | Benchmark matrix, ablations, evaluation protocol |
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

## Key References

| Paper | Venue | Role |
|-------|-------|------|
| S-FLM (Deschenaux & Gulcehre) | arXiv 2605.11125 | Baseline we extend |
| Clifford Flows (Alesiani & Maruyama) | NeurIPS 2024 ML4PS | Normalizing flows over multivectors |
| GAFL (Wagner et al.) | arXiv 2411.05238 | CFA architecture + flow matching |
| CARE (Sriram et al.) | NeurIPS 2025 UniReps | Rotor embeddings generalizing RoPE |
