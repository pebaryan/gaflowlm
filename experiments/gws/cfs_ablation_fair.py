#!/usr/bin/env python3
"""Fair GWS ablation on the CFS flow model.

The original `cfs_ablation.py` set the optimizer LR to a constant
`schedule.eta_max` for the GWS condition while the cosine baseline
decayed normally. That gave GWS ~60% more effective per-step LR than
cosine at step 1000, making the "GWS wins by ~25%" headline mostly an
LR-magnitude artifact rather than evidence for grade staggering.

This script holds the LR schedule constant across conditions and only
varies how the per-step LR is *distributed across grades*:

  * cosine          - same scalar cosine LR everywhere (all-ones blade scale)
  * staggered_unfair - per-grade factors from GradeRotorSchedule; their mean
                      drifts so the total LR-budget-per-parameter differs
                      from cosine. This isolates the same scaling shape
                      cfs_ablation used, but applied on top of the cosine
                      schedule rather than eta_max.
  * staggered_fair   - identical per-grade factors as `unfair`, then
                      renormalized each step so blade_scale.sum() == n.
                      That keeps the per-blade LR budget exactly equal to
                      cosine, so any remaining loss difference must come
                      from the per-grade redistribution.

If staggered_fair ~= cosine, grade staggering is not helping. If it wins,
grade staggering has merit on its own.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from gaflowlm.clifford.engine import CliffordEngine
from gaflowlm.models.cfs_model import CFSModel
from gaflowlm.gws.rotor_schedule import CosineSchedule, GradeRotorSchedule


def identify_mv_params(model, mv_dim):
    mv_params = {}
    for name, param in model.named_parameters():
        axes = [i for i, s in enumerate(param.shape) if s == mv_dim]
        if axes:
            mv_params[name] = axes
    return mv_params


def build_blade_scale(engine, grade_lrs, base_lr, device, dtype, fair: bool):
    """Per-blade scale vector from per-grade LRs.

    When `fair` is True, the result is renormalized so blade_scale.sum() == n,
    preserving the total per-blade LR budget. Otherwise the raw factors
    pass through (which can lower or raise the total LR per parameter
    relative to baseline).
    """
    mv_dim = 1 << engine.k
    scale = torch.ones(mv_dim, device=device, dtype=dtype)
    if base_lr < 1e-12:
        return scale
    for g in range(min(len(grade_lrs), len(engine.grade_masks))):
        mask = engine.grade_masks[g].squeeze().to(device=device, dtype=dtype)
        ratio = grade_lrs[g] / base_lr
        scale = scale + mask * (ratio - 1.0)
    if fair:
        scale = scale * (mv_dim / scale.sum().clamp(min=1e-12))
    return scale


def apply_blade_scale(model, mv_params, blade_scale, mv_dim):
    for name, axes in mv_params.items():
        p = dict(model.named_parameters())[name]
        if p.grad is None:
            continue
        g = p.grad
        for axis in axes:
            shape = [1] * g.ndim
            shape[axis] = mv_dim
            g.mul_(blade_scale.view(shape))


def train_step(model, engine, mv_params, optimizer, lr, blade_scale, x0, mv_dim):
    model.train()
    B, L = x0.shape
    positions = torch.arange(L, device=x0.device).unsqueeze(0).expand(B, -1)
    m0 = model.encode_tokens(x0, positions=positions)

    noise = torch.randn_like(m0)
    t = torch.rand(B, 1, device=m0.device, dtype=m0.dtype)
    xt = (1 - t.unsqueeze(-1)) * m0 + t.unsqueeze(-1) * noise
    target_velocity = noise - m0

    pred_velocity = model(xt, t, positions=positions)
    loss = F.mse_loss(pred_velocity, target_velocity)

    optimizer.zero_grad()
    loss.backward()
    if blade_scale is not None:
        apply_blade_scale(model, mv_params, blade_scale, mv_dim)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.step()
    return loss


def run(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    engine = CliffordEngine(k=args.k, dtype=torch.float64)
    mv_dim = 1 << args.k
    n_grades = args.k + 1

    base_model = CFSModel(
        vocab_size=args.vocab,
        hidden_size=args.hidden,
        k=args.k,
        n_blocks=args.blocks,
        n_heads=args.heads,
        ff_dim=args.hidden * 4,
        engine=engine,
        max_len=args.seq_len,
    ).to(device)

    model_cos = copy.deepcopy(base_model)
    model_unfair = copy.deepcopy(base_model)
    model_fair = copy.deepcopy(base_model)

    opt_cos = torch.optim.AdamW(model_cos.parameters(), lr=args.lr, weight_decay=0.01)
    opt_unfair = torch.optim.AdamW(model_unfair.parameters(), lr=args.lr, weight_decay=0.01)
    opt_fair = torch.optim.AdamW(model_fair.parameters(), lr=args.lr, weight_decay=0.01)

    cosine = CosineSchedule(args.lr, args.steps, args.lr * 0.01, warmup_steps=args.warmup)
    grade = GradeRotorSchedule(
        k_s=2, n_grades=n_grades, T=args.steps,
        eta_max=args.lr, eta_min=args.lr * 0.01,
        phase_offsets=[i * 0.15 for i in range(n_grades)],
        bivector_assignment="orthogonal", warmup_steps=args.warmup,
    )

    mv_params = identify_mv_params(model_cos, mv_dim)

    records = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x0 = torch.randint(0, args.vocab, (args.batch, args.seq_len), device=device)
        lr = cosine(step)

        loss_cos = train_step(model_cos, engine, mv_params, opt_cos, lr, None, x0, mv_dim)

        grade_lrs = grade(step)
        bs_unfair = build_blade_scale(engine, grade_lrs, args.lr, device, model_unfair.token_to_mv.weight.dtype, fair=False)
        loss_unfair = train_step(model_unfair, engine, mv_params, opt_unfair, lr, bs_unfair, x0, mv_dim)

        bs_fair = build_blade_scale(engine, grade_lrs, args.lr, device, model_fair.token_to_mv.weight.dtype, fair=True)
        loss_fair = train_step(model_fair, engine, mv_params, opt_fair, lr, bs_fair, x0, mv_dim)

        if step % args.log_interval == 0:
            sps = step / max(1, time.time() - t0)
            rec = {
                "step": step,
                "loss_cos": float(loss_cos.detach()),
                "loss_unfair": float(loss_unfair.detach()),
                "loss_fair": float(loss_fair.detach()),
                "blade_scale_sum_unfair": float(bs_unfair.sum()),
                "blade_scale_sum_fair": float(bs_fair.sum()),
            }
            records.append(rec)
            print(f"step {step:>5d}  COS={float(loss_cos):.4f}  "
                  f"UNFAIR={float(loss_unfair):.4f}  FAIR={float(loss_fair):.4f}  "
                  f"(unfair_sum={float(bs_unfair.sum()):.2f}, fair_sum={float(bs_fair.sum()):.2f})  "
                  f"{sps:.1f} sps")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"cfs_ablation_fair_k{args.k}_h{args.hidden}_b{args.blocks}_s{args.seed}.jsonl"
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(records)} records -> {out_file}")

    best_cos = min(r["loss_cos"] for r in records)
    best_unfair = min(r["loss_unfair"] for r in records)
    best_fair = min(r["loss_fair"] for r in records)
    print(f"Best:  COS={best_cos:.4f}  UNFAIR={best_unfair:.4f}  FAIR={best_fair:.4f}")
    if best_unfair < best_cos:
        print(f"  UNFAIR (extra LR + staggered)  beats COS by {(best_cos - best_unfair) / best_cos * 100:.1f}%")
    if best_fair < best_cos:
        print(f"  FAIR  (cosine LR + staggered)  beats COS by {(best_cos - best_fair) / best_cos * 100:.1f}%")
    elif best_fair > best_cos:
        print(f"  FAIR  (cosine LR + staggered)  LOSES to COS by {(best_fair - best_cos) / best_cos * 100:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
