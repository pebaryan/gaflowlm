#!/usr/bin/env python3
"""GWS vs Cosine head-to-head using the integrated CFSAlgorithm.

Uses the same data, same model, same optimizer — only difference is
whether CFSAlgorithm.use_gws is True or False.

Reports per-seed results and aggregate stats to decide whether GWS
should be an experimental option or a default improvement.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flash_attn_mock
sys.modules["flash_attn"] = flash_attn_mock
sys.modules["flash_attn.layers"] = flash_attn_mock
sys.modules["flash_attn.layers.rotary"] = flash_attn_mock.layers.rotary

import torch
from clifford.engine import CliffordEngine
from models.cfs_model import CFSAlgorithm


@dataclass
class TrainConfig:
    """Config for both GWS and baseline."""
    class Algo:
        name = "cfs"
        rhf_clifford_k = 4
        cfs_sample_steps = 16
        cfs_loss = "mse"
        cfs_time_sampling = "uniform"
        cfs_noise_scale = 1.0
        cfs_normalize_noise = False
        cfs_use_higher_order = True

    class Model:
        hidden_size = 64
        n_blocks = 2
        n_heads = 4
        length = 32

    class Optim:
        lr = 3e-4
        weight_decay = 0.01

    algo = Algo()
    model = Model()
    optim = Optim()


class FixedTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    def __len__(self):
        return self.vocab_size


def make_config(use_gws: bool, k: int, hidden: int, blocks: int, heads: int,
                lr: float, seq_len: int, total_steps: int) -> TrainConfig:
    cfg = TrainConfig()
    cfg.algo.rhf_clifford_k = k
    cfg.algo.cfs_sample_steps = 16
    cfg.model.hidden_size = hidden
    cfg.model.n_blocks = blocks
    cfg.model.n_heads = heads
    cfg.model.length = seq_len
    cfg.optim.lr = lr
    cfg.optim.weight_decay = 0.01
    cfg.optim.use_gws = use_gws
    cfg.optim.gws_num_grades = k + 1
    cfg.optim.gws_phase_stagger = True
    cfg.optim.gws_phase_step = 0.4 * math.pi
    cfg.optim.gws_total_steps = total_steps
    cfg.optim.gws_learnable_phase_offsets = False
    cfg.optim.gws_eta_min = lr * 0.01
    return cfg


def run_comparison(args):
    device = torch.device(args.device)
    results = []

    for seed_idx in range(args.seeds):
        seed = args.start_seed + seed_idx
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({seed_idx+1}/{args.seeds})")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        engine = CliffordEngine(k=args.k, dtype=torch.float64)
        tokenizer = FixedTokenizer(args.vocab)

        # Shared data: generate all batches upfront so both see identical data
        torch.manual_seed(seed)
        batches = [torch.randint(0, args.vocab, (args.batch, args.seq_len))
                   for _ in range(args.steps)]

        # --- Baseline (no GWS) ---
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cfg_base = make_config(False, args.k, args.hidden, args.blocks,
                               args.heads, args.lr, args.seq_len, args.steps)
        algo_base = CFSAlgorithm(cfg_base, tokenizer, engine=engine).to(device)
        base_losses = []

        t0 = time.time()
        for step, batch in enumerate(batches):
            batch = batch.to(device)
            result = algo_base.train_step(batch)
            base_losses.append(result["loss"])
            if (step + 1) % args.log_interval == 0:
                print(f"  BASE step {step+1}: loss={result['loss']:.6f}")
        base_time = time.time() - t0
        base_best = min(base_losses)
        base_final = base_losses[-1]

        # --- GWS ---
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cfg_gws = make_config(True, args.k, args.hidden, args.blocks,
                              args.heads, args.lr, args.seq_len, args.steps)
        algo_gws = CFSAlgorithm(cfg_gws, tokenizer, engine=engine).to(device)
        gws_losses = []

        t0 = time.time()
        for step, batch in enumerate(batches):
            batch = batch.to(device)
            result = algo_gws.train_step(batch)
            gws_losses.append(result["loss"])
            if (step + 1) % args.log_interval == 0:
                print(f"  GWS  step {step+1}: loss={result['loss']:.6f}")
        gws_time = time.time() - t0
        gws_best = min(gws_losses)
        gws_final = gws_losses[-1]

        r = {
            "seed": seed,
            "base_best": base_best,
            "base_final": base_final,
            "gws_best": gws_best,
            "gws_final": gws_final,
            "base_time": base_time,
            "gws_time": gws_time,
        }
        results.append(r)

        winner = "GWS" if gws_best < base_best else ("TIE" if abs(gws_best - base_best) < 1e-6 else "BASE")
        pct = abs(base_best - gws_best) / max(base_best, gws_best) * 100
        print(f"\n  Seed {seed}: BASE={base_best:.6f} GWS={gws_best:.6f}  {winner} ({pct:.1f}%)")
        print(f"  Time: BASE={base_time:.1f}s GWS={gws_time:.1f}s  overhead={max(0, gws_time-base_time):.1f}s")

    # Aggregate
    base_bests = [r["base_best"] for r in results]
    gws_bests = [r["gws_best"] for r in results]
    base_finals = [r["base_final"] for r in results]
    gws_finals = [r["gws_final"] for r in results]
    base_times = [r["base_time"] for r in results]
    gws_times = [r["gws_time"] for r in results]
    n = len(results)

    def ms(v):
        m = sum(v) / len(v)
        s = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v)))
        return m, s

    gws_wins = sum(1 for r in results if r["gws_best"] < r["base_best"])
    base_wins = sum(1 for r in results if r["base_best"] < r["gws_best"])
    ties = n - gws_wins - base_wins

    bm, bs = ms(base_bests)
    gm, gs = ms(gws_bests)
    bfm, bfs = ms(base_finals)
    gfm, gfs = ms(gws_finals)
    btm, bts = ms(base_times)
    gtm, gts = ms(gws_times)

    print(f"\n{'='*60}")
    print(f"AGGREGATE ({n} seeds)")
    print(f"{'='*60}")
    print(f"  Best loss:")
    print(f"    BASE: {bm:.6f} ± {bs:.6f}  (min={min(base_bests):.6f})")
    print(f"    GWS:  {gm:.6f} ± {gs:.6f}  (min={min(gws_bests):.6f})")
    print(f"  Final loss:")
    print(f"    BASE: {bfm:.6f} ± {bfs:.6f}")
    print(f"    GWS:  {gfm:.6f} ± {gfs:.6f}")
    print(f"  Time:")
    print(f"    BASE: {btm:.1f}s ± {bts:.1f}s")
    print(f"    GWS:  {gtm:.1f}s ± {gts:.1f}s  (overhead: {gtm-btm:.1f}s avg)")
    print(f"  Wins: GWS={gws_wins} BASE={base_wins} TIE={ties}")

    if gm < bm:
        pct = (bm - gm) / bm * 100
        print(f"\n  GWS wins by {pct:.1f}% on average best loss")
    else:
        pct = (gm - bm) / gm * 100
        print(f"\n  BASE wins by {pct:.1f}% on average best loss")

    # Verdict
    print(f"\n{'='*60}")
    if gws_wins >= n * 0.8 and gm < bm:
        print("VERDICT: GWS should be DEFAULT — wins consistently and by meaningful margin")
    elif gws_wins > base_wins and gm < bm:
        print("VERDICT: GWS should be DEFAULT — wins majority but check overhead")
    elif gws_wins == base_wins or abs(gm - bm) / max(gm, bm) < 0.05:
        print("VERDICT: GWS stays EXPERIMENTAL — no clear advantage")
    else:
        print("VERDICT: GWS stays EXPERIMENTAL — baseline is competitive or better")

    # Save
    out_dir = Path("training_logs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"gws_comparison_k{args.k}_h{args.hidden}_b{args.blocks}_{args.steps}steps.json"
    with open(out_file, 'w') as f:
        json.dump({
            "args": vars(args),
            "results": results,
            "aggregate": {
                "base_best_mean": bm, "base_best_std": bs,
                "gws_best_mean": gm, "gws_best_std": gs,
                "base_final_mean": bfm, "base_final_std": bfs,
                "gws_final_mean": gfm, "gws_final_std": gfs,
                "gws_wins": gws_wins, "base_wins": base_wins, "ties": ties,
            }
        }, f, indent=2)
    print(f"\nSaved to {out_file}")


def main():
    p = argparse.ArgumentParser(description="GWS vs Cosine head-to-head")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--vocab", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seeds", type=int, default=7)
    p.add_argument("--start-seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run_comparison(args)


if __name__ == "__main__":
    main()
