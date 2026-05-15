"""
Run GWS ablation across multiple seeds.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def run_single(seed: int, args: list[str]) -> dict:
    cmd = [sys.executable, "-m", "gaflowlm.gws.ablation",
           "--seed", str(seed)] + args
    print(f"\n{'='*60}")
    print(f"Running ablation seed {seed}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"FAILED (seed {seed}): {result.stderr[-500:]}")
        return {"seed": seed, "status": "failed"}

    output = result.stdout
    summary = {"seed": seed, "status": "ok"}
    for line in output.splitlines():
        if "Cosine:        best=" in line:
            parts = line.strip().split()
            for p in parts:
                if p.startswith("best="):
                    summary["cos_best"] = float(p.split("=")[1])
        if "Rotor-Uniform: best=" in line:
            parts = line.strip().split()
            for p in parts:
                if p.startswith("best="):
                    summary["rotor_best"] = float(p.split("=")[1])
        if "GWS-Orthogonal: best=" in line:
            parts = line.strip().split()
            for p in parts:
                if p.startswith("best="):
                    summary["gws_best"] = float(p.split("=")[1])
        if "Rotor-Uniform vs Cosine" in line:
            summary["rotor_vs_cos"] = line.strip()
        if "GWS-Orthogonal vs Rotor-U" in line:
            summary["gws_vs_rotor"] = line.strip()
        if "GWS-Orthogonal vs Cosine" in line:
            summary["gws_vs_cos"] = line.strip()

    print(f"  COS={summary.get('cos_best', 'N/A'):.6f} | "
          f"ROTOR-U={summary.get('rotor_best', 'N/A'):.6f} | "
          f"GWS-O={summary.get('gws_best', 'N/A'):.6f}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Multi-seed GWS ablation")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--start-seed", type=int, default=42)
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

    extra = [
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
        result = run_single(seed, extra)
        results.append(result)

    # Aggregate
    cos_bests = [r["cos_best"] for r in results if r["status"] == "ok" and "cos_best" in r]
    rotor_bests = [r["rotor_best"] for r in results if r["status"] == "ok" and "rotor_best" in r]
    gws_bests = [r["gws_best"] for r in results if r["status"] == "ok" and "gws_best" in r]
    n_ok = len(cos_bests)

    def mean_std(vals):
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((x - m)**2 for x in vals) / max(1, len(vals)))
        return m, s

    print(f"\n{'='*60}")
    print(f"ABLATION AGGREGATE ({n_ok}/{args.seeds} seeds)")
    print(f"{'='*60}")
    if cos_bests:
        m, s = mean_std(cos_bests)
        print(f"  Cosine:        {m:.6f} ± {s:.6f}  (min={min(cos_bests):.6f})")
    if rotor_bests:
        m, s = mean_std(rotor_bests)
        print(f"  Rotor-Uniform: {m:.6f} ± {s:.6f}  (min={min(rotor_bests):.6f})")
    if gws_bests:
        m, s = mean_std(gws_bests)
        print(f"  GWS-Orthogonal: {m:.6f} ± {s:.6f}  (min={min(gws_bests):.6f})")

    if cos_bests and rotor_bests:
        cm, _ = mean_std(cos_bests)
        rm, _ = mean_std(rotor_bests)
        if rm < cm:
            print(f"\n  Rotor vs Cosine: Rotor wins by {(cm-rm)/cm*100:.1f}%")
        else:
            print(f"\n  Rotor vs Cosine: Cosine wins by {(rm-cm)/rm*100:.1f}%")

    if rotor_bests and gws_bests:
        rm, _ = mean_std(rotor_bests)
        gm, _ = mean_std(gws_bests)
        if gm < rm:
            print(f"  GWS vs Rotor:   GWS wins by {(rm-gm)/rm*100:.1f}%  ← grade separation effect")
        else:
            print(f"  GWS vs Rotor:   Rotor wins by {(gm-rm)/gm*100:.1f}%")

    if cos_bests and gws_bests:
        cm, _ = mean_std(cos_bests)
        gm, _ = mean_std(gws_bests)
        if gm < cm:
            print(f"  GWS vs Cosine:  GWS wins by {(cm-gm)/cm*100:.1f}%  ← total effect")
        else:
            print(f"  GWS vs Cosine:  Cosine wins by {(gm-cm)/gm*100:.1f}%")

    # Save
    out_dir = Path("gaflowlm/gws/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ablation_summary.json"
    with open(out_file, 'w') as f:
        json.dump({
            "n_seeds": args.seeds,
            "n_ok": n_ok,
            "cos_bests": cos_bests,
            "rotor_bests": rotor_bests,
            "gws_bests": gws_bests,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
