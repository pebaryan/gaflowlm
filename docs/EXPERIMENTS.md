# Experiment Plan

## Benchmark Matrix

### Primary Benchmarks (same as S-FLM)

| Benchmark | Metric | S-FLM Baseline | RHF Target | CFS Target |
|-----------|--------|---------------|------------|------------|
| GSM8K (T=1) | Accuracy | 18.4% | ≥20% | ≥21% |
| GSM8K (T=0.1) | Accuracy | ~18% | ≥22% | ≥25% |
| Hard Sudoku | Accuracy | 45% | ≥48% | ≥50% |
| TinyGSM | Perplexity | — | ≤ S-FLM | ≤ S-FLM |
| OpenWebText | Perplexity | — | ≤ S-FLM | ≤ S-FLM |

### Reference vs. Practical Execution

The table above is the target benchmark set. The original S-FLM runs were
cluster-scale jobs, while this workspace is split across the current machine
and `mi25`, where the GPU may be occupied by the llama.cpp server.

| Tier | What it is | Typical execution |
|------|------------|-------------------|
| Reference S-FLM | Original paper protocol | Multi-GPU training, checkpoint eval, full sampling runs |
| Local smoke | Fast correctness check | Synthetic overfit, GSM8K-test reconstruction, small-step sampler checks |
| Local real-data | Small-scale validation | TinyGSM subset or GSM8K-test fixture on one GPU or CPU fallback |
| mi25 eval slot | Shared-GPU validation | Run only when the GPU is free; otherwise use CPU smoke checks |

### Diagnostic Benchmarks (novel to GAFlowLM)

| Benchmark | What it tests |
|-----------|---------------|
| Norm drift | ‖z‖ - 1 over training steps |
| NaN/Inf frequency | Numerical stability |
| Grade energy distribution | How much information lives in each grade (CFS only) |
| Analogy tasks (BATS) | Whether bivector channels capture relational structure |
| Rotor composition error | ‖R₃ - R₂R₁‖ for multi-step sampling |
| Low-NFE curve | Accuracy at {1,2,4,8,16,32,64,128,256,512,1024} steps |
| Real-data smoke test | Token reconstruction on the local GSM8K-test fixture |

### Recommended Plan For This Workspace

Use the following order when validating changes here:

1. Synthetic overfit regression test.
2. GSM8K-test reconstruction smoke test.
3. TinyGSM subset run if the current machine has an idle GPU.
4. mi25 checkpoint evaluation when the shared GPU slot is available.
5. Full S-FLM-style benchmark only after you have a dedicated multi-GPU run.

## Ablation Studies

### A1: Rotor vs. SLERP (Isolating the flow algebra)

Same architecture (S-arch), same hyperparameters, only swap the flow primitives:
- `slerp(clean, noisy, alpha_t)` vs. `rotor_sandwich(R_{alpha_t}, clean)`
- `log_map(x, target)` vs. `outer_product(x, target)`
- `exp_map(x, delta)` vs. `rotor_apply(exp(B*dt), x)`

This is the most important ablation. If RHF outperforms S-FLM here, the GA substitution is validated regardless of CFS results.

### A2: CFA vs. Standard Attention (Isolating the attention mechanism)

Same flow algebra (rotor), same hyperparameters, swap attention:
- Standard QKV dot-product attention
- CFA with geometric bilinear messages
- CFA + higher-order message passing

### A3: Multivector vs. Vector Embeddings (Isolating the representation)

Same CFA attention, same flow algebra, swap the representation:
- Vector-only (grade-1): equivalent to RHF
- Vector + scalar (grades 0-1): adds position-invariant channel
- Vector + bivector (grades 1-2): adds rotation plane information
- Full multivector (grades 0-k): maximum representational power

### A4: CARE Position Encoding vs. None vs. Time Gates

- No position encoding (S-FLM style: time-conditioned gates only)
- CARE rotors as position encoding
- Standard sinusoidal position encoding (added to vector part)

### A5: Clifford Algebra Dimension k

- k=4: Cl(4,0,0), 16 components per multivector
- k=8: Cl(8,0,0), 256 components (recommended)
- k=16: Cl(16,0,0), grades 0-2 only (137 components)
- k=d: Cl(768,0,0), infeasible; use projection layers

### A6: Noise Schedule Adaptation

- LogLinear (S-FLM default)
- LogLinear + truncation (S-FLM best)
- LogLinear + truncation + adaptive (S-FLM best)
- GA-adapted truncation bound (re-derived for multivector sphere volume)

## Training Protocol

### Model Config (matching S-FLM)

```yaml
# Small model (for rapid iteration)
hidden_size: 768
cond_dim: 128
length: 1024
n_blocks: 12
n_heads: 12
dropout: 0.1

# GA-specific
clifford_dim: 8          # k for Cl(k,0,0)
use_higher_order: true   # CFA 3-body messages (CFS only)
use_care: true           # CARE position encoding (CFS only)
grade_projection: [0,1,2]  # which grades to use (CFS only)

# Training
max_steps: 250000
global_batch_size: 512
lr: 3e-4
weight_decay: 0.1
warmup_steps: 2000

# Noise schedule (start with S-FLM's best)
noise: log-linear-adaptive
alpha_max: 0.121
adaptive_refit_every: 50
adaptive_ema: 0.9
```

### Hardware

- Reference S-FLM training: multi-GPU, originally 2x4 GPUs for TinyGSM and 4x4 GPUs for OWT.
- Current workspace: one local GPU if available; otherwise CPU for smoke tests.
- `mi25`: treat the GPU as shared with the llama.cpp server and use it only when the slot is free.
- Expected: synthetic and GSM8K-test probes finish quickly on CPU/GPU; full TinyGSM/OWT benchmarks are not the default on this hardware.

### Evaluation Schedule

- Every code change: run the synthetic overfit regression test.
- Before merge: run the GSM8K-test reconstruction smoke test.
- When the current machine has a free GPU: run a TinyGSM subset or a short checkpoint eval.
- On `mi25`: run the same smoke tests, and only schedule longer evals when the GPU is not occupied by llama.cpp.
- For the original S-FLM-aligned benchmark, use the upstream scripts with an existing checkpoint and the original multi-GPU setup.

## Success Criteria

### Minimum Viable Result (MVR)
- RHF matches S-FLM on GSM8K with fewer NaN events → proves numerical stability advantage

### Target Result
- RHF improves GSM8K by ≥10% relative (18.4% → ≥20.2%) at T=1 → proves geometric advantage

### Stretch Result
- CFS improves GSM8K by ≥25% relative (18.4% → ≥23%) at T=1, with interpretable grade structure → proves multivector advantage

### Regardless of GSM8K result
- If RHF eliminates norm drift and NaN events → paper contribution on numerical stability
- If CFA captures interpretable grade structure → paper contribution on interpretability
- If neither improves accuracy → negative result paper: "GA for language modeling: elegant but not effective (yet)"
