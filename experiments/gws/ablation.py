"""
GWS Ablation Study: isolating the contribution of grade-wise separation
vs. the rotor schedule itself.

Three conditions compared:
1. Cosine (baseline): standard cosine annealing, single LR for all params
2. Rotor-Uniform: rotor schedule with SAME angle for all grades (tests rotor vs cosine)
3. GWS-Orthogonal: rotor schedule with DIFFERENT angles per grade (tests grade separation)

If (2) > (1): the rotor schedule itself is beneficial
If (3) > (2): grade-wise separation adds value on top of the rotor
If (3) > (1) but (2) ≈ (1): the win is entirely from grade separation

Uses the CliffordFNO2d on synthetic 2D Navier-Stokes.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from gaflowlm.gws.rotor_schedule import CosineSchedule, GradeRotorSchedule

# benchmark_ns lives alongside this script. When run as
# `python experiments/gws/ablation.py`, its directory is on sys.path[0]
# so a sibling import resolves without any package wiring.
from benchmark_ns import CliffordFNO2d, generate_ns_batch  # noqa: E402


def run_ablation(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print(f"Seed: {args.seed} | Device: {device}")

    # Build model template
    template = CliffordFNO2d(
        n_blocks=args.blocks,
        channels=args.channels,
        modes=args.modes,
    ).to(device)

    n_params = sum(p.numel() for p in template.parameters())
    print(f"CliffordFNO2d: {n_params:,} params, Cl(2,0,0), {args.blocks} blocks, {args.channels} channels")

    # Three model copies
    model_cos = copy.deepcopy(template).to(device)
    model_rotor = copy.deepcopy(template).to(device)
    model_gws = copy.deepcopy(template).to(device)

    opt_cos = torch.optim.AdamW(model_cos.parameters(), lr=args.lr, weight_decay=0.01)
    opt_rotor = torch.optim.AdamW(model_rotor.parameters(), lr=args.lr, weight_decay=0.01)
    opt_gws = torch.optim.AdamW(model_gws.parameters(), lr=args.lr, weight_decay=0.01)

    loss_fn = nn.MSELoss()

    # Schedules
    cosine_schedule = CosineSchedule(args.lr, args.steps, args.lr * 0.01, warmup_steps=args.warmup)

    # Rotor-Uniform: same angle for all grades (phase_offsets all 0)
    rotor_uniform_schedule = GradeRotorSchedule(
        k_s=args.k_s,
        n_grades=3,
        T=args.steps,
        eta_max=args.lr,
        eta_min=args.lr * 0.01,
        phase_offsets=[0.0, 0.0, 0.0],  # ALL SAME — no grade separation
        bivector_assignment='orthogonal',
        warmup_steps=args.warmup,
    )

    # GWS-Orthogonal: different angles per grade (the full GWS)
    gws_orthogonal_schedule = GradeRotorSchedule(
        k_s=args.k_s,
        n_grades=3,
        T=args.steps,
        eta_max=args.lr,
        eta_min=args.lr * 0.01,
        phase_offsets=[0.0, 0.15, 0.35],  # staggered — grade separation
        bivector_assignment='orthogonal',
        warmup_steps=args.warmup,
    )

    # Output
    out_dir = Path("gaflowlm/gws/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ablation_c{args.channels}_b{args.blocks}_m{args.modes}_s{args.seed}.jsonl"

    records = []
    start = time.time()
    T = args.steps

    for step in range(1, T + 1):
        x, y = generate_ns_batch(args.batch, args.grid, 1, str(device))
        x, y = x.to(device), y.to(device)

        lr_cos = cosine_schedule(step)
        lr_rotor = rotor_uniform_schedule(step)
        lr_gws = gws_orthogonal_schedule(step)

        # --- Cosine baseline ---
        for pg in opt_cos.param_groups:
            pg['lr'] = lr_cos
        opt_cos.zero_grad()
        loss_cos = loss_fn(model_cos(x), y)
        loss_cos.backward()
        opt_cos.step()

        # --- Rotor-Uniform (same LR for all grades) ---
        # Use the mean of the three grade LRs as the single LR
        mean_rotor_lr = sum(lr_rotor) / len(lr_rotor)
        for pg in opt_rotor.param_groups:
            pg['lr'] = mean_rotor_lr
        opt_rotor.zero_grad()
        loss_rotor = loss_fn(model_rotor(x), y)
        loss_rotor.backward()
        opt_rotor.step()

        # --- GWS-Orthogonal (per-grade LR scaling) ---
        base_lr_for_scale = lr_cos  # use cosine LR as reference for scaling
        grade_scales = {}
        if len(lr_gws) >= 3 and base_lr_for_scale > 1e-12:
            grade_scales = {
                0: lr_gws[0] / base_lr_for_scale,
                1: lr_gws[1] / base_lr_for_scale,
                2: lr_gws[2] / base_lr_for_scale,
            }

        for pg in opt_gws.param_groups:
            pg['lr'] = base_lr_for_scale

        opt_gws.zero_grad()
        loss_gws = loss_fn(model_gws(x), y)
        loss_gws.backward()

        # Apply grade-wise scaling to spectral conv weights
        for name, param in model_gws.named_parameters():
            if param.grad is None or not grade_scales:
                continue
            if 'spec.weight.0' in name:
                param.grad.data.mul_(grade_scales.get(0, 1.0))
            elif 'spec.weight.1' in name or 'spec.weight.2' in name:
                param.grad.data.mul_(grade_scales.get(1, 1.0))
            elif 'spec.weight.3' in name:
                param.grad.data.mul_(grade_scales.get(2, 1.0))

        opt_gws.step()

        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - start
            sps = step / max(1, elapsed)

            record = {
                'step': step,
                'loss_cosine': float(loss_cos.detach()),
                'loss_rotor_uniform': float(loss_rotor.detach()),
                'loss_gws_orthogonal': float(loss_gws.detach()),
                'lr_cosine': float(lr_cos),
                'lr_rotor_uniform': [float(v) for v in lr_rotor],
                'lr_gws_orthogonal': [float(v) for v in lr_gws],
            }
            records.append(record)

            print(
                f"Step {step:>5d} | "
                f"COS={float(loss_cos):.6f} "
                f"ROTOR-U={float(loss_rotor):.6f} "
                f"GWS-O={float(loss_gws):.6f} | "
                f"lr={float(lr_cos):.2e} | "
                f"{sps:.1f} steps/s"
            )

    # Save
    with open(out_file, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print(f"\nAblation complete. {len(records)} records → {out_file}")

    # Summary
    if len(records) >= 2:
        final_cos = records[-1]['loss_cosine']
        final_rotor = records[-1]['loss_rotor_uniform']
        final_gws = records[-1]['loss_gws_orthogonal']
        best_cos = min(r['loss_cosine'] for r in records)
        best_rotor = min(r['loss_rotor_uniform'] for r in records)
        best_gws = min(r['loss_gws_orthogonal'] for r in records)

        print(f"\n--- Ablation Summary (seed {args.seed}) ---")
        print(f"  Cosine:        best={best_cos:.6f}  final={final_cos:.6f}")
        print(f"  Rotor-Uniform: best={best_rotor:.6f}  final={final_rotor:.6f}")
        print(f"  GWS-Orthogonal: best={best_gws:.6f}  final={final_gws:.6f}")
        print()

        # Rotor vs cosine
        if best_rotor < best_cos:
            print(f"  Rotor-Uniform vs Cosine:    Rotor wins by {(best_cos - best_rotor)/best_cos*100:.1f}%")
        else:
            print(f"  Rotor-Uniform vs Cosine:    Cosine wins by {(best_rotor - best_cos)/best_rotor*100:.1f}%")

        # GWS vs rotor
        if best_gws < best_rotor:
            print(f"  GWS-Orthogonal vs Rotor-U:  GWS wins by {(best_rotor - best_gws)/best_rotor*100:.1f}%")
        else:
            print(f"  GWS-Orthogonal vs Rotor-U:  Rotor wins by {(best_gws - best_rotor)/best_gws*100:.1f}%")

        # GWS vs cosine (total effect)
        if best_gws < best_cos:
            print(f"  GWS-Orthogonal vs Cosine:   GWS wins by {(best_cos - best_gws)/best_cos*100:.1f}%")
        else:
            print(f"  GWS-Orthogonal vs Cosine:   Cosine wins by {(best_gws - best_cos)/best_gws*100:.1f}%")


def main():
    p = argparse.ArgumentParser(description="GWS ablation: rotor vs grade separation")
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--k-s", type=int, default=2)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run_ablation(args)


if __name__ == "__main__":
    main()
