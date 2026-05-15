"""
Run GWS benchmark across multiple seeds and aggregate results.

Usage:
    python -m gaflowlm.gws.multiseed --seeds 5 --steps 2000
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def run_single(seed: int, args: list[str]) -> dict:
    """Run benchmark for one seed, return summary."""
    cmd = [sys.executable, "-m", "gaflowlm.gws.benchmark_ns",
           "--seed", str(seed)] + args
    print(f"\n{'='*60}")
    print(f"Running seed {seed}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"FAILED (seed {seed}): {result.stderr[-500:]}")
        return {"seed": seed, "status": "failed"}

    # Parse final comparison from output
    output = result.stdout
    summary = {"seed": seed, "status": "ok"}
    for line in output.splitlines():
        if "GWS:    final=" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p.startswith("final="):
                    summary["gws_final"] = float(p.split("=")[1])
                if p.startswith("best="):
                    summary["gws_best"] = float(p.split("=")[1])
        if "Cosine: final=" in line:
            parts = line.split()
            for p in parts:
                if p.startswith("final="):
                    summary["cos_final"] = float(p.split("=")[1])
                if p.startswith("best="):
                    summary["cos_best"] = float(p.split("=")[1])
        if "GWS wins" in line or "Cosine wins" in line:
            summary["winner"] = "GWS" if "GWS wins" in line else "Cosine"

    print(f"  GWS best={summary.get('gws_best', 'N/A')} | Cosine best={summary.get('cos_best', 'N/A')} | Winner={summary.get('winner', 'N/A')}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Multi-seed GWS benchmark")
    p.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    p.add_argument("--start-seed", type=int, default=42, help="First seed value")
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    extra_args = [
        "--channels", str(args.channels),
        "--blocks", str(args.blocks),
        "--modes", str(args.modes),
        "--grid", str(args.grid),
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--warmup", str(args.warmup),
        "--log-interval", str(args.log_interval),
        "--device", args.device,
    ]

    results = []
    for i in range(args.seeds):
        seed = args.start_seed + i
        result = run_single(seed, extra_args)
        results.append(result)

    # Aggregate
    gws_bests = [r["gws_best"] for r in results if r["status"] == "ok" and "gws_best" in r]
    cos_bests = [r["cos_best"] for r in results if r["status"] == "ok" and "cos_best" in r]
    gws_wins = sum(1 for r in results if r.get("winner") == "GWS")
    cos_wins = sum(1 for r in results if r.get("winner") == "Cosine")
    n_ok = len(gws_bests)

    print(f"\n{'='*60}")
    print(f"AGGREGATE ({n_ok}/{args.seeds} seeds succeeded)")
    print(f"{'='*60}")
    if gws_bests:
        gws_mean = sum(gws_bests) / len(gws_bests)
        gws_std = math.sqrt(sum((x - gws_mean)**2 for x in gws_bests) / max(1, len(gws_bests)))
        print(f"  GWS  best loss: {gws_mean:.6f} ± {gws_std:.6f}  (min={min(gws_bests):.6f})")
    if cos_bests:
        cos_mean = sum(cos_bests) / len(cos_bests)
        cos_std = math.sqrt(sum((x - cos_mean)**2 for x in cos_bests) / max(1, len(cos_bests)))
        print(f"  Cos  best loss: {cos_mean:.6f} ± {cos_std:.6f}  (min={min(cos_bests):.6f})")
    print(f"  GWS wins: {gws_wins} | Cosine wins: {cos_wins} | Ties/fails: {n_ok - gws_wins - cos_wins}")

    if gws_bests and cos_bests:
        gws_mean = sum(gws_bests) / len(gws_bests)
        cos_mean = sum(cos_bests) / len(cos_bests)
        if gws_mean < cos_mean:
            print(f"  → GWS wins by {(cos_mean - gws_mean) / cos_mean * 100:.1f}% on average")
        else:
            print(f"  → Cosine wins by {(gws_mean - cos_mean) / gws_mean * 100:.1f}% on average")

    # Save aggregate
    out_dir = Path("gaflowlm/gws/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "multiseed_summary.json"
    with open(out_file, 'w') as f:
        json.dump({
            "n_seeds": args.seeds,
            "n_ok": n_ok,
            "gws_bests": gws_bests,
            "cos_bests": cos_bests,
            "gws_wins": gws_wins,
            "cos_wins": cos_wins,
            "results": results,
        }, f, indent=2)
    print(f"  Saved to {out_file}")


if __name__ == "__main__":
    main()
