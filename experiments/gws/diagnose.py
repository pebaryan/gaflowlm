"""
GWS Diagnostic: Log per-grade gradient norms during standard training.

This answers the fundamental question: do multivector grades have
different convergence dynamics? If they all move together, grade-wise
scheduling won't help. If they diverge, we have empirical motivation.

Usage:
    python -m gaflowlm.gws.diagnose --k 4 --hidden 64 --blocks 2 --steps 2000
    python -m gaflowlm.gws.diagnose --k 8 --hidden 256 --blocks 4 --steps 5000

Output:
    - Per-grade gradient norms printed every log_interval steps
    - JSONL file with per-step per-grade norms for plotting
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from gaflowlm.models.cfs_model import CFSModel
from gaflowlm.clifford.engine import CliffordEngine
from gaflowlm.gws.grade_decompose import grade_norms, identify_multivector_params


def make_synthetic_data(vocab_size: int, seq_len: int, n_batches: int, device: str):
    """Generate random token sequences."""
    data = []
    for _ in range(n_batches):
        tokens = torch.randint(0, vocab_size, (4, seq_len), device=device)
        data.append(tokens)
    return data


def run_diagnostic(args):
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Build engine
    engine = CliffordEngine(k=args.k, device=str(device), dtype=torch.float32)
    print(f"Clifford algebra: Cl({args.k},0,0), {engine.n} basis blades")

    # Build CFS model directly (not CFSAlgorithm, which owns its own optimizer)
    model = CFSModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden,
        k=args.k,
        n_blocks=args.blocks,
        n_heads=4,
        ff_dim=args.hidden * 4,
        max_len=args.seq_len,
        dropout=0.1,
        engine=engine,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Identify multivector params
    mv_map = identify_multivector_params(model, engine)
    mv_names = sorted(name for name, is_mv in mv_map.items() if is_mv)
    print(f"Multivector parameters: {len(mv_names)}")
    for name in mv_names[:10]:
        print(f"  {name}")
    if len(mv_names) > 10:
        print(f"  ... and {len(mv_names) - 10} more")

    # Optimizer with standard cosine schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    T = args.steps

    # Data
    data = make_synthetic_data(args.vocab_size, args.seq_len, 200, str(device))

    # Output file
    out_dir = Path("gaflowlm/gws/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"diagnose_k{args.k}_h{args.hidden}_b{args.blocks}.jsonl"
    print(f"Logging to: {out_file}")

    # Grade names for display
    grade_names = {
        0: "scalar", 1: "vector", 2: "bivector",
        3: "trivector", 4: "quadvector",
    }

    # Training loop with grade norm logging
    records = []
    start = time.time()
    loss_fn = nn.CrossEntropyLoss()

    for step in range(1, T + 1):
        # Cosine LR
        lr_mult = 0.5 * (1 + math.cos(math.pi * step / T))
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr * lr_mult

        batch = data[step % len(data)]
        logits = model(batch)  # [B, L, vocab]
        # Autoregressive: predict next token
        logits_shifted = logits[:, :-1, :].contiguous()
        targets = batch[:, 1:].contiguous()
        loss = loss_fn(logits_shifted.view(-1, args.vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            # Compute per-grade norms across all multivector params
            grade_total_norms = {g: 0.0 for g in range(engine.k + 1)}
            grade_param_counts = {g: 0 for g in range(engine.k + 1)}

            for name, param in model.named_parameters():
                if param.grad is not None and name in mv_names:
                    norms = grade_norms(param.grad, engine)
                    for g, n in norms.items():
                        grade_total_norms[g] += n
                        grade_param_counts[g] += 1

            # Also log total grad norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = math.sqrt(total_norm)

            elapsed = time.time() - start
            steps_per_s = step / max(1, elapsed)

            record = {
                'step': step,
                'loss': float(loss),
                'lr': float(args.lr * lr_mult),
                'total_grad_norm': total_norm,
                'grade_norms': {str(g): grade_total_norms[g] for g in range(engine.k + 1)},
            }
            records.append(record)

            # Print summary
            grade_strs = []
            for g in range(min(engine.k + 1, 5)):  # Show first 5 grades
                gn = grade_total_norms[g]
                name = grade_names.get(g, f"g{g}")
                grade_strs.append(f"{name}={gn:.4f}")

            print(
                f"Step {step:>5d} | loss={loss:.4f} | lr={args.lr * lr_mult:.2e} | "
                f"total_norm={total_norm:.4f} | {' | '.join(grade_strs)} | "
                f"{steps_per_s:.1f} steps/s"
            )

    # Save all records
    with open(out_file, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print(f"\nDiagnostic complete. {len(records)} records saved to {out_file}")

    # Quick analysis: ratio of grade norms over time
    if len(records) >= 2:
        early = records[0]['grade_norms']
        late = records[-1]['grade_norms']
        print("\n--- Grade norm ratio (early → late) ---")
        for g in range(min(engine.k + 1, 5)):
            name = grade_names.get(g, f"g{g}")
            e_val = early.get(str(g), 0)
            l_val = late.get(str(g), 0)
            ratio = l_val / max(e_val, 1e-10)
            print(f"  {name}: {e_val:.4f} → {l_val:.4f} (ratio={ratio:.2f})")


def main():
    p = argparse.ArgumentParser(description="GWS diagnostic: per-grade gradient norms")
    p.add_argument("--k", type=int, default=4, help="Clifford algebra dimension")
    p.add_argument("--hidden", type=int, default=64, help="Hidden/embedding size")
    p.add_argument("--blocks", type=int, default=2, help="Number of transformer blocks")
    p.add_argument("--vocab-size", type=int, default=1000, help="Vocab size for synthetic data")
    p.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    p.add_argument("--steps", type=int, default=2000, help="Training steps")
    p.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    p.add_argument("--log-interval", type=int, default=100, help="Log every N steps")
    p.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run_diagnostic(args)


if __name__ == "__main__":
    main()
