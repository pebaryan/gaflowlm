# Implementation Roadmap

## Fork Strategy

### Source: S-FLM (https://github.com/jdeschena/s-flm)

The S-FLM codebase (9 hours old, initial commit as of May 14, 2026) provides:
- `algo.py`: SFM class with SLERP-based corruption, CE loss
- `utils.py`: `slerp()`, `log_map()`, `exp_map()`, `justnorm()` (all `@torch.compile` decorated)
- `samplers.py`: `SFMSampler` with exact/sample/top-k velocity, Euler integration
- `models/sphere_arch.py`: S-arch Transformer (unit-norm, spherical residuals)
- `models/sphere_dit.py`: SphereDiT (standard DiT variant)
- `noise_schedules.py`: LogLinear, CosineSquared, TruncatedScheduleWrapper, AdaptiveSchedule
- `configs/`: Hydra/OmegaConf YAML composition

### Modification Plan

**Phase 1: RHF (Minimal — swap flow primitives)**
```
gaflowlm/
├── algo.py              # Modified: SFM -> RHF (add rotor corruption)
├── utils.py             # Modified: add rotor_alternatives to slerp/log_map/exp_map
├── samplers.py           # Modified: add rotor_euler_step alongside exp_map
├── models/
│   ├── sphere_arch.py   # Modified: add rotor residuals (R_γ h R̃_γ)
│   ├── sphere_dit.py    # Minimal changes (same interface)
│   └── clifford.py       # NEW: CliffordEngine, geometric products, rotors
├── clifford/
│   ├── __init__.py
│   ├── algebra.py       # Core Cl(k,0,0) operations (geometric product, reverse, etc.)
│   ├── rotor.py          # Rotor exponential, application, composition
│   ├── cayley.py         # Precomputed Cayley tensor for Cl(k,0,0)
│   └── projections.py    # Embedding-to-Clifford and Clifford-to-embedding layers
├── configs/
│   ├── algo/
│   │   └── rhf.yaml      # RHF-specific config
│   ├── model/
│   │   ├── small-sphere-arch.yaml  # Inherited from S-FLM
│   │   └── small-sphere-arch-rhf.yaml  # RHF variant
│   └── noise/             # Inherited from S-FLM
├── callbacks/             # Inherited from S-FLM
├── data/                  # Inherited from S-FLM
└── scripts/               # Training scripts (inherited + RHF variants)
```

**Phase 2: CFS (Maximal — CFA, multivector embeddings)**
```
+ models/
│   ├── cfa.py            # Clifford Frame Attention block
│   ├── cfs_arch.py       # CFS Transformer (CFA blocks + rotor residuals)
│   └── care.py           # CARE position encoding
+ clifford/
│   ├── multivector.py    # Multivector representation utilities
│   └── normalize.py      # Spinor normalization ⟨MM̃⟩₀ = 1
+ configs/
│   ├── algo/cfs.yaml
│   └── model/small-cfs.yaml
```

## GA Libraries

### Recommended: Custom `clifford/` module (this repo)

Reasons:
1. **S-FLM uses `@torch.compile`:** All core ops (slerp, log_map, exp_map) are `@torch.compile`-decorated. A custom module can match this pattern natively. External GA libraries (clifford, kingdon) are NumPy-based and don't support `torch.compile`.

2. **Sparse multivector representation:** We only need grades 0,1,2 (and optionally grade-k for Cl(k,0,0)). External libraries represent full 2^k-dimensional multivectors, which is wasteful for k≥8.

3. **Cayley tensor precomputation:** The geometric product AB can be computed via `einsum('...j,...k,jki->...i', A, B, cayley)`. Precomputing the Cayley tensor for Cl(k,0,0) once and registering it as a buffer enables batched GPU computation. No external library supports this pattern.

### Alternatives considered (and why rejected for now):

| Library | Language | Why not primary |
|---------|----------|----------------|
| `clifford` (Python) | NumPy | No PyTorch, no torch.compile, full 2^k representation |
| `kingdon` (Python) | JAX | JAX, not PyTorch; compilation model differs from S-FLM |
| `torch_ga` / `Torch-GA` | PyTorch | Research prototype, not maintained, no Cayley tensor |
| GATr (Qualcomm) | PyTorch | PGA only (Cl(3,0,1)), not general Cl(k,0,0); 16-dim fixed |

We will implement a minimal, high-performance Clifford engine in PyTorch with:
- Precomputed Cayley tensor (registered buffer, not learned)
- `@torch.compile` compatible operations
- Sparse grade projection (only needed grades)
- Batched geometric product via `einsum`

## Integration with S-arch Backbone

The S-arch backbone in `models/sphere_arch.py` uses:
1. `justnorm(x) = x / ‖x‖` — replace with rotor normalization or spinor normalization
2. Spherical residual: `justnorm(h + γ(B - h))` — replace with `R_γ h R̃_γ`
3. Weight normalization: `renormalize_weights()` — keep (still enforcing unit-norm rows/columns)
4. QKV attention: `softmax(Q·K / √d_k) V` — keep for RHF, replace with CFA for CFS

The key architectural decision is: **RHF swaps flow primitives only (no backbone changes), while CFS swaps both flow primitives and backbone (CFA + multivector residuals).** This isolates the effect of each substitution.

## Training on Benchmarks

### TinyGSM (quick validation)
```bash
python -m main data=tiny-gsm model=small-sphere-arch-rhf algo=rhf \
  noise=log-linear-truncated noise.alpha_max=0.121 \
  algo.renormalize_weights=True loader.global_batch_size=512 \
  trainer.max_steps=50000
```

### GSM8K (primary metric)
```bash
python -m main data=gsm8k model=small-sphere-arch-rhf algo=rhf \
  noise=log-linear-adaptive noise.alpha_max=0.121 \
  noise.adaptive_refit_every=50 noise.adaptive_ema=0.9 \
  algo.renormalize_weights=True loader.global_batch_size=512 \
  trainer.max_steps=250000
```

### Sudoku (hard reasoning)
```bash
python -m main data=sudoku model=small-sphere-arch-rhf algo=rhf \
  noise=log-linear-adaptive noise.alpha_max=0.121 \
  algo.renormalize_weights=True trainer.max_steps=250000
```

## Practical Validation Tiers

The original S-FLM protocol assumes multi-GPU training and checkpoint eval.
For this repo and the available hardware, use the following ladder:

1. Synthetic overfit regression test for CFS and RHF code paths.
2. GSM8K-test reconstruction smoke test on a single GPU or CPU fallback.
3. TinyGSM subset runs when a local GPU is available.
4. mi25 shared-GPU validation only when the llama.cpp server is not using it.
5. Full S-FLM-aligned benchmark only with a dedicated multi-GPU setup.

## Current Status

**Phase 0: Research investigation (COMPLETE)**
- [x] Literature review of 4 foundational papers
- [x] Gap analysis confirming no existing work at the intersection
- [x] Architecture proposals (RHF + CFS)
- [x] Pseudocode for rotor velocity field and sampling
- [x] Experiment plan with ablations
- [x] Paper outline

**Phase 1: Implementation (NEXT)**
- [ ] Implement `clifford/algebra.py` (geometric product, reverse, grade projection)
- [ ] Implement `clifford/cayley.py` (Cayley tensor generation for Cl(k,0,0))
- [ ] Implement `clifford/rotor.py` (rotor exp/log/apply)
- [ ] Implement `clifford/projections.py` (embedding-to-Clifford layers)
- [ ] Fork S-FLM codebase and integrate `algo.py` rotor forward process
- [ ] Add rotor alternatives in `utils.py`
- [ ] Add rotor sampling in `samplers.py`
- [ ] Train RHF on TinyGSM (sanity check)

**Phase 2: CFS Implementation (COMPLETE)**
- [x] Implement `models/cfa.py` / `models/cfs_arch.py` (Clifford Frame Attention)
- [x] Implement `models/care.py` (CARE position encoding)
- [x] Implement `models/cfs_model.py` and `standalone_train.py` integration
- [x] Train CFS on TinyGSM-style synthetic data
- [x] Add bounded real-data smoke tests on TinyGSM / GSM8K-test
- [x] Add flow-sampling and reconstruction benchmarking

**Phase 3: Evaluation & Paper (NEXT)**
- [ ] Full benchmark suite (GSM8K, Sudoku, OWT)
- [ ] Ablation studies (A1-A6)
- [ ] Analysis (grade energy, norm drift, composition quality)
- [ ] Paper writing and arXiv submission
