"""
GWS on flow-style CFS model.

Compares three conditions on the rectified-flow CFS training objective:
1. Cosine annealing (baseline)
2. Rotor-Uniform (same rotor angle for all grades)
3. GWS-Orthogonal (staggered rotor angles per grade)

The key difference from the CliffordFNO2d benchmark:
- CFS has 32+ multivector-valued parameters with mv_dim in various axes
- Cl(4,0,0) has 5 grades (scalar, vector, bivector, trivector, quadvector)
- Grade decomposition uses the engine's grade_masks along mv_dim axes
- The training objective is flow-matching, not direct prediction
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from gaflowlm.clifford.engine import CliffordEngine
from gaflowlm.models.cfs_model import CFSModel
from gaflowlm.gws.rotor_schedule import CosineSchedule, GradeRotorSchedule


def identify_mv_params(model, mv_dim):
    """Identify parameters with multivector dimensions and their axes."""
    mv_params = {}
    for name, param in model.named_parameters():
        mv_axes = [i for i, s in enumerate(param.shape) if s == mv_dim]
        if mv_axes:
            mv_params[name] = mv_axes
    return mv_params


def build_grade_scale_vector(engine, grade_lrs, base_lr, device, dtype):
    """Build a per-blade scale vector from per-grade LRs.

    Returns [mv_dim] tensor where each blade position is scaled by
    its grade's LR ratio vs base_lr.
    """
    mv_dim = 1 << engine.k
    scale = torch.ones(mv_dim, device=device, dtype=dtype)
    if base_lr < 1e-12:
        return scale
    for g in range(min(len(grade_lrs), len(engine.grade_masks))):
        mask = engine.grade_masks[g].squeeze().to(device=device, dtype=dtype)
        ratio = grade_lrs[g] / base_lr
        scale = scale + mask * (ratio - 1.0)
    return scale


def apply_grade_scaling(model, mv_params, grade_scale, mv_dim):
    """Apply per-blade gradient scaling to all multivector parameters.

    For each parameter with mv_dim in its shape, scale the gradient
    along the mv_dim axis by the per-blade scale vector.
    """
    for name, mv_axes in mv_params.items():
        param = dict(model.named_parameters())[name]
        if param.grad is None:
            continue
        g = param.grad
        for axis in mv_axes:
            # Reshape scale to broadcast along the mv_dim axis
            shape = [1] * g.ndim
            shape[axis] = mv_dim
            s = grade_scale.view(shape)
            g.mul_(s)


def compute_grade_norms(model, mv_params, engine, mv_dim):
    """Compute per-grade gradient norm totals across all mv parameters."""
    n_grades = len(engine.grade_masks)
    grade_norms_sq = [0.0] * n_grades
    for name, mv_axes in mv_params.items():
        param = dict(model.named_parameters())[name]
        if param.grad is None:
            continue
        g = param.grad
        # Use the first mv axis for grade decomposition
        axis = mv_axes[0]
        for gr in range(n_grades):
            mask = engine.grade_masks[gr].squeeze().to(device=g.device, dtype=g.dtype)
            # Select blades of this grade along the mv axis
            shape = [1] * g.ndim
            shape[axis] = mv_dim
            m = mask.view(shape)
            masked_g = g * m
            grade_norms_sq[gr] += masked_g.norm().item() ** 2
    return [math.sqrt(v) for v in grade_norms_sq]


def train_cfs(model, engine, mv_params, optimizer, schedule, step, x0,
              is_gws=False, grade_scale_fn=None, mv_dim=16):
    """Single training step for flow-style CFS."""
    model.train()
    B, L = x0.shape

    # Encode tokens to clean multivectors
    positions = torch.arange(L, device=x0.device).unsqueeze(0).expand(B, -1)
    m0 = model.encode_tokens(x0, positions=positions)

    # Sample noise and time
    noise = torch.randn_like(m0)
    t = torch.rand(B, 1, device=m0.device, dtype=m0.dtype)
    xt = (1 - t.unsqueeze(-1)) * m0 + t.unsqueeze(-1) * noise
    target_velocity = noise - m0

    # Forward
    pred_velocity = model(xt, t, positions=positions)
    loss = F.mse_loss(pred_velocity, target_velocity)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Apply grade-wise scaling if GWS
    if is_gws and grade_scale_fn is not None:
        grade_lrs = schedule(step)
        base_lr = schedule.eta_max  # reference LR for scaling
        grade_scale = build_grade_scale_vector(
            engine, grade_lrs, base_lr, x0.device, m0.dtype
        )
        apply_grade_scaling(model, mv_params, grade_scale, mv_dim)

    # Set optimizer LR
    if is_gws:
        lr = schedule.eta_max  # base LR for optimizer
    else:
        lr = schedule(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    optimizer.step()
    return loss


def run_experiment(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print(f"Seed: {args.seed} | Device: {device}")

    engine = CliffordEngine(k=args.k, dtype=torch.float64)
    mv_dim = 1 << args.k
    n_grades = args.k + 1

    # Build model
    model = CFSModel(
        vocab_size=args.vocab,
        hidden_size=args.hidden,
        k=args.k,
        n_blocks=args.blocks,
        n_heads=args.heads,
        ff_dim=args.hidden * 4,
        engine=engine,
        max_len=args.seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_mv = sum(1 for n, p in model.named_parameters() if mv_dim in p.shape)
    print(f"CFS flow model: {n_params:,} params, {n_mv} multivector params, Cl({args.k},0,0) {n_grades} grades")

    mv_params = identify_mv_params(model, mv_dim)
    print(f"Multivector parameter axes: {len(mv_params)} params")

    # Three copies for ablation
    model_cos = copy.deepcopy(model).to(device)
    model_rotor = copy.deepcopy(model).to(device)
    model_gws = copy.deepcopy(model).to(device)

    opt_cos = torch.optim.AdamW(model_cos.parameters(), lr=args.lr, weight_decay=0.01)
    opt_rotor = torch.optim.AdamW(model_rotor.parameters(), lr=args.lr, weight_decay=0.01)
    opt_gws = torch.optim.AdamW(model_gws.parameters(), lr=args.lr, weight_decay=0.01)

    # Schedules
    cosine_schedule = CosineSchedule(args.lr, args.steps, args.lr * 0.01, warmup_steps=args.warmup)

    rotor_uniform_schedule = GradeRotorSchedule(
        k_s=args.k_s, n_grades=n_grades, T=args.steps,
        eta_max=args.lr, eta_min=args.lr * 0.01,
        phase_offsets=[0.0] * n_grades,
        bivector_assignment='orthogonal',
        warmup_steps=args.warmup,
    )

    gws_orthogonal_schedule = GradeRotorSchedule(
        k_s=args.k_s, n_grades=n_grades, T=args.steps,
        eta_max=args.lr, eta_min=args.lr * 0.01,
        phase_offsets=[i * 0.15 for i in range(n_grades)],  # stagger each grade
        bivector_assignment='orthogonal',
        warmup_steps=args.warmup,
    )

    # Output
    # Save next to the script so results stay grouped with the experiment.
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"cfs_ablation_k{args.k}_h{args.hidden}_b{args.blocks}_s{args.seed}.jsonl"

    records = []
    start = time.time()

    for step in range(1, args.steps + 1):
        # Synthetic batch: random token IDs
        x0 = torch.randint(0, args.vocab, (args.batch, args.seq_len), device=device)

        # Cosine baseline
        loss_cos = train_cfs(model_cos, engine, mv_params, opt_cos,
                             cosine_schedule, step, x0, is_gws=False)

        # Rotor-Uniform
        rotor_lrs = rotor_uniform_schedule(step)
        mean_rotor_lr = sum(rotor_lrs) / len(rotor_lrs)
        loss_rotor = train_cfs(model_rotor, engine, mv_params, opt_rotor,
                               lambda s: mean_rotor_lr, step, x0, is_gws=False)

        # GWS-Orthogonal
        loss_gws = train_cfs(model_gws, engine, mv_params, opt_gws,
                             gws_orthogonal_schedule, step, x0,
                             is_gws=True, grade_scale_fn=gws_orthogonal_schedule,
                             mv_dim=mv_dim)

        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - start
            sps = step / max(1, elapsed)

            # Grade norms from GWS model
            grade_norms = compute_grade_norms(model_gws, mv_params, engine, mv_dim)

            record = {
                'step': step,
                'loss_cosine': float(loss_cos.detach()),
                'loss_rotor_uniform': float(loss_rotor.detach()),
                'loss_gws_orthogonal': float(loss_gws.detach()),
                'grade_norms': {f'grade_{g}': grade_norms[g] for g in range(n_grades)},
            }
            records.append(record)

            grade_str = '  '.join(f'g{g}={grade_norms[g]:.2f}' for g in range(n_grades))
            print(
                f"Step {step:>5d} | "
                f"COS={float(loss_cos):.4f} "
                f"ROT-U={float(loss_rotor):.4f} "
                f"GWS-O={float(loss_gws):.4f} | "
                f"{grade_str} | "
                f"{sps:.1f} sps"
            )

    # Save
    with open(out_file, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print(f"\nAblation complete. {len(records)} records → {out_file}")

    # Summary
    if len(records) >= 2:
        best_cos = min(r['loss_cosine'] for r in records)
        best_rotor = min(r['loss_rotor_uniform'] for r in records)
        best_gws = min(r['loss_gws_orthogonal'] for r in records)

        print(f"\n--- CFS Ablation Summary (seed {args.seed}, Cl({args.k},0,0)) ---")
        print(f"  Cosine:        best={best_cos:.6f}")
        print(f"  Rotor-Uniform: best={best_rotor:.6f}")
        print(f"  GWS-Orthogonal: best={best_gws:.6f}")

        if best_gws < best_cos:
            print(f"  → GWS vs Cosine: GWS wins by {(best_cos - best_gws)/best_cos*100:.1f}%")
        else:
            print(f"  → GWS vs Cosine: Cosine wins by {(best_gws - best_cos)/best_gws*100:.1f}%")

        if best_gws < best_rotor:
            print(f"  → GWS vs Rotor:  GWS wins by {(best_rotor - best_gws)/best_rotor*100:.1f}%")
        else:
            print(f"  → GWS vs Rotor:  Rotor wins by {(best_gws - best_rotor)/best_gws*100:.1f}%")


def main():
    p = argparse.ArgumentParser(description="GWS ablation on flow-style CFS")
    p.add_argument("--k", type=int, default=4, help="Clifford algebra dimension")
    p.add_argument("--hidden", type=int, default=64, help="Hidden size")
    p.add_argument("--blocks", type=int, default=2, help="Transformer blocks")
    p.add_argument("--heads", type=int, default=4, help="Attention heads")
    p.add_argument("--vocab", type=int, default=256, help="Vocab size")
    p.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    p.add_argument("--steps", type=int, default=2000, help="Training steps")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--k-s", type=int, default=2, help="Scheduling algebra dim")
    p.add_argument("--warmup", type=int, default=200, help="Warmup steps")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--log-interval", type=int, default=100, help="Log every N steps")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run_experiment(args)


if __name__ == "__main__":
    main()
