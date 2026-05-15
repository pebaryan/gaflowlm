"""
Plot GWS diagnostic results.

Usage:
    python -m gaflowlm.gws.plot_logs gaflowlm/gws/logs/diagnose_k4_h64_b2.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def plot_grade_norms(records, out_path=None):
    """Plot per-grade gradient norms over training steps."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    steps = [r['step'] for r in records]
    loss = [r['loss'] for r in records]
    total_norms = [r['total_grad_norm'] for r in records]

    # Collect grade norms
    grade_keys = sorted(records[0]['grade_norms'].keys(), key=lambda x: int(x))
    grade_data = {g: [r['grade_norms'][g] for r in records] for g in grade_keys}

    grade_names = {
        '0': 'scalar', '1': 'vector', '2': 'bivector',
        '3': 'trivector', '4': 'quadvector',
    }
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Training loss
    ax = axes[0, 0]
    ax.plot(steps, loss, 'k-', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. Total gradient norm
    ax = axes[0, 1]
    ax.plot(steps, total_norms, 'k-', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Grad Norm')
    ax.set_title('Total Gradient Norm')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. Per-grade gradient norms
    ax = axes[1, 0]
    for i, g in enumerate(grade_keys):
        name = grade_names.get(g, f"grade {g}")
        ax.plot(steps, grade_data[g], label=name, color=colors[i % len(colors)], linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Grade Grad Norm')
    ax.set_title('Per-Grade Gradient Norms')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Grade ratios (each grade / total)
    ax = axes[1, 1]
    for i, g in enumerate(grade_keys):
        name = grade_names.get(g, f"grade {g}")
        ratios = [gd / max(tn, 1e-10) for gd, tn in zip(grade_data[g], total_norms)]
        ax.plot(steps, ratios, label=name, color=colors[i % len(colors)], linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Fraction of Total Norm')
    ax.set_title('Grade Norm / Total Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved to {out_path}")
    else:
        plt.savefig('gws_diagnostic.png', dpi=150)
        print("Saved to gws_diagnostic.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', help='JSONL file from diagnose.py')
    p.add_argument('--output', '-o', default=None, help='Output PNG path')
    args = p.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")
    plot_grade_norms(records, args.output)


if __name__ == "__main__":
    main()
