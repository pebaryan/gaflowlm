# GAFlowLM: Geometric Algebra Flow Language Modeling

Fork of [S-FLM](https://github.com/jdeschena/s-flm) that extends hyperspherical flow-based language modeling with Clifford algebra: rotor sandwich products instead of trigonometric SLERP, multivector embeddings, Clifford Frame Attention, and a grade-wise learning-rate scheduler.

## Status

| Track | What's implemented | Tests |
|-------|--------------------|-------|
| RHF — rotor primitives over the existing S-arch backbone | rotor SLERP / log_map / exp_map, training and sampling subclasses of `SFM` | integration tests need `hydra` (training stack) |
| CFS — multivector flow on `Cl(k,0,0)` | `CliffordEngine`, CFA blocks, CARE positional encoding, `CFSModel` + `CFSAlgorithm` | `test_cfs.py`, `test_cfs_model.py`, `test_cfs_learning.py` |
| GWS — grade-wise scheduling for Clifford NNs | `GWScheduler` (in `gaflowlm/schedulers.py`); ablation scripts in `experiments/gws/` | `test_gws.py` |
| CARE — rotor positional encoding | `gaflowlm/models/care.py` | `test_care.py` |

Honest framing of current numerics:

- **RHF rotor ops are numerically identical to S-FLM's trig forms.** They use `acos` and `sin` and renormalize after `exp_map` just like SLERP does; the rotor expression is a rewrite, not an optimization. The structural benefits live in the math and only pay off once the bivector velocity is used (currently the sampler still projects to a tangent vector before stepping).
- **`CliffordEngine.rotor_exp` / `bivector_log` assume simple bivectors** (B² ∈ ℝ). They are correct for bivectors produced from two grade-1 vectors. For Cl(k,0,0) with k≥4 a generic grade-2 element produced by a network may not be simple and the closed form is wrong — the engine ships docstring warnings, not a polar decomposition.
- The latest GWS ablations show grade separation helping on the CFS flow objective. They were re-run after a recent fix that corrected which blade slots the projection layers were writing to (earlier results conflated "first k indices" with grade-1).

See [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md) for the longer task list.

## Layout

```
gaflowlm/        # importable package
  clifford/      # Cl(k,0,0) engine: Cayley tensor, rotors, grade projection
  models/        # CFS architecture (cfs_arch, cfs_model, care, sphere_*)
  gws/           # grade-wise scheduler library (rotor_schedule, grade_decompose)
  schedulers.py  # GWScheduler — the maintained GWS entry point
  rotor_utils.py # analytic rotor SLERP / log_map / exp_map
  rhf_algo.py    # SFM subclass with rotor primitives
  algo.py / samplers.py / trainer_base.py / ...   # upstream S-FLM training stack
tests/           # pytest suite (53 tests; RHF integration tests need hydra)
experiments/gws/ # one-off ablation, benchmark, and plotting scripts
docs/            # research proposal, architecture, math notes, paper outline
```

## Install

```bash
pip install -e .                          # package only (torch already installed)
pip install -r gaflowlm/requirements.txt  # full training stack (Lightning, Hydra, transformers, ...)
```

`requirements.txt` is intentionally not duplicated in `pyproject.toml` because the NGC PyTorch container ships its own torch/CUDA build that pip-installed torch would clobber.

## Run the tests

```bash
pytest tests/ --ignore=tests/test_rhf_integration.py    # 53 tests, ~13s on CPU
```

`test_rhf_integration.py` needs the full training stack (hydra etc.) to import.

## Upstream training / sampling

For the S-FLM baseline training and evaluation scripts (TinyGSM, OpenWebText, Sudoku), see [`gaflowlm/README.md`](gaflowlm/README.md) — it's the original S-FLM README and the existing scripts under `scripts/train/` and `scripts/sample/` still work as documented.

## Two variants in one tree

- **Variant A — RHF (Rotor Hyperspherical Flow):** swap S-FLM's SLERP / log_map / exp_map for rotor algebra; keep the S-arch backbone. Minimal intervention. Entry: `gaflowlm/rhf_algo.py`.
- **Variant B — CFS (Clifford Flow on Sphere):** embed tokens as multivectors, use CFA attention and CARE positional encoding, predict a multivector velocity field. Maximal intervention. Entry: `gaflowlm/models/cfs_model.py`.

## Key references

| Paper | Venue | Role |
|-------|-------|------|
| S-FLM (Deschenaux & Gulcehre) | arXiv 2605.11125 | Baseline we extend |
| Clifford Flows (Alesiani & Maruyama) | NeurIPS 2024 ML4PS | Normalizing flows over multivectors |
| GAFL (Wagner et al.) | arXiv 2411.05238 | CFA architecture + flow matching |
| CARE (Sriram et al.) | NeurIPS 2025 UniReps | Rotor embeddings generalizing RoPE |

## Docs

| Document | Description |
|----------|-------------|
| [Research Proposal](docs/RESEARCH_PROPOSAL.md) | Extended abstract with architecture, math, and hypotheses |
| [Literature Review](docs/LITERATURE_REVIEW.md) | Summaries of foundational papers + gap analysis |
| [Architecture](docs/ARCHITECTURE.md) | Component-level design for RHF and CFS |
| [Pseudocode](docs/PSEUDOCODE.md) | Rotor velocity field and sampling algorithms |
| [Mathematical Insights](docs/MATHEMATICAL_INSIGHTS.md) | SLERP ↔ rotor sandwich identities, etc. |
| [Experiments](docs/EXPERIMENTS.md) | Benchmark matrix, ablations, evaluation protocol |
| [GWS Research Design](docs/GWS_RESEARCH_DESIGN.md) | Grade-wise geometric scheduling track |
| [Paper Outline](docs/PAPER_OUTLINE.md) | Targeted arXiv submission structure |
