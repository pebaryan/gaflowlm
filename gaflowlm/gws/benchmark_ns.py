"""
Self-contained GWS benchmark on 2D Navier-Stokes PDE surrogate.

We build a minimal Clifford Fourier Neural Operator (CFNO) for 2D PDEs:
- Input: scalar + vector fields on 2D grid → Cl(2,0,0) multivector (4 blades)
- Clifford spectral convolution layers (simplified)
- Output: predicted next-step fields

This is a controlled testbed for GWS vs. cosine annealing on a
Clifford architecture with real data dynamics, without depending
on the Microsoft CliffordLayers package layout.

The 2D Navier-Stokes dataset is generated synthetically (vorticity
field with known dynamics) so no external data download is needed.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from gaflowlm.clifford.engine import CliffordEngine
from gaflowlm.gws.grade_decompose import grade_norms, identify_multivector_params
from gaflowlm.gws.rotor_schedule import CosineSchedule, GradeRotorSchedule


# ---------------------------------------------------------------------------
# Synthetic 2D Navier-Stokes data
# ---------------------------------------------------------------------------

def generate_ns_batch(batch_size: int, grid_size: int, n_steps: int, device: str):
    """Generate synthetic 2D vorticity field evolution.

    Uses a simplified spectral method: random initial vorticity, 
    evolve with diffusion and random forcing.

    Returns:
        x: [B, 2, grid, grid] input fields (vorticity + stream function proxy)
        y: [B, 2, grid, grid] target fields (next step)
    """
    B, N = batch_size, grid_size
    # Random initial vorticity in Fourier space
    k = torch.randn(B, N, N, device=device, dtype=torch.float32) * 0.5
    # Low-pass filter (keep first few modes)
    freq = torch.fft.fftfreq(N, device=device)
    kx = freq.unsqueeze(1).expand(N, N)
    ky = freq.unsqueeze(0).expand(N, N)
    mask = (kx.abs() < 0.25) & (ky.abs() < 0.25)
    k = k * mask.float()

    # Spatial vorticity
    w = torch.fft.ifft2(k).real
    # Stream function proxy (for the vector component)
    psi = -torch.fft.ifft2(k / (kx**2 + ky**2 + 1e-6)).real

    # Stack as [B, 2, N, N]
    fields = torch.stack([w, psi], dim=1)

    # Simple evolution: diffusion step
    dt = 0.01
    nu = 0.01  # viscosity
    w_hat = torch.fft.fft2(w)
    lap = -(kx**2 + ky**2)
    w_next_hat = w_hat * torch.exp(-nu * lap * dt)
    w_next = torch.fft.ifft2(w_next_hat).real
    psi_next = -torch.fft.ifft2(w_next_hat / (kx**2 + ky**2 + 1e-6)).real

    target = torch.stack([w_next, psi_next], dim=1)

    return fields, target


# ---------------------------------------------------------------------------
# Minimal Clifford Spectral Conv for 2D
# ---------------------------------------------------------------------------

class CliffordSpectralConv2d(nn.Module):
    """Spectral convolution in Cl(2,0,0).

    Input shape: [B, C, 4, H, W] where 4 = n_blades for Cl(2,0,0).
    We apply FFT, multiply by learned frequency weights per-blade, and IFFT.
    The Clifford structure is preserved by applying the same frequency mask
    to all blades, but with separate weights per blade.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int, n_blades: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.n_blades = n_blades

        scale = 1.0 / (in_channels * out_channels)
        # Weights: [n_blades, in_channels, out_channels, modes, modes]
        self.weight = nn.ParameterList([
            nn.Parameter(torch.randn(in_channels, out_channels, modes, modes, 2) * scale)
            for _ in range(n_blades)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, n_blades, H, W]"""
        B, C, NB, H, W = x.shape
        assert NB == self.n_blades

        output = torch.zeros(B, self.out_channels, NB, H, W, device=x.device, dtype=x.dtype)

        for b in range(NB):
            x_hat = torch.fft.rfft2(x[:, :, b], norm='ortho')  # [B, C, H, W//2+1]
            # Truncate to low modes
            x_hat_trunc = x_hat[:, :, :self.modes, :self.modes]  # [B, C, modes, modes]

            # Complex weight: [C_in, C_out, modes, modes, 2]
            w = torch.view_as_complex(self.weight[b])  # [C_in, C_out, modes, modes]
            # Contract over input channels: [B, C_out, modes, modes]
            out_hat = torch.einsum('bcjk,iojk->bojk', x_hat_trunc, w)

            # Pad back to full resolution
            out_full = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                                   device=x.device, dtype=x_hat.dtype)
            out_full[:, :, :self.modes, :self.modes] = out_hat

            output[:, :, b] = torch.fft.irfft2(out_full, s=(H, W), norm='ortho')

        return output


class CliffordFNOBlock2d(nn.Module):
    """Single FNO block with Clifford spectral conv + skip connection."""

    def __init__(self, channels: int, modes: int, n_blades: int, activation=F.gelu):
        super().__init__()
        self.spec = CliffordSpectralConv2d(channels, channels, modes, n_blades)
        self.mlp = nn.Sequential(
            nn.Linear(n_blades * channels, n_blades * channels * 2),
            nn.GELU(),
            nn.Linear(n_blades * channels * 2, n_blades * channels),
        )
        self.norm = nn.LayerNorm(n_blades * channels)
        self.activation = activation
        self.n_blades = n_blades

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, n_blades, H, W]"""
        B, C, NB, H, W = x.shape
        residual = x

        # Spectral path
        x_spec = self.spec(x)

        # MLP path (applied per-pixel)
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B, H, W, C * NB)
        x_mlp = self.mlp(x_flat).reshape(B, H, W, C, NB).permute(0, 3, 4, 1, 2)

        x = x_spec + x_mlp
        x = self.norm(x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C * NB))
        x = x.reshape(B, H, W, C, NB).permute(0, 3, 4, 1, 2)

        return self.activation(x + residual)


class CliffordFNO2d(nn.Module):
    """Clifford Fourier Neural Operator for 2D PDEs in Cl(2,0,0).

    Input: [B, 2, H, W] (scalar field + one vector component)
    Output: [B, 2, H, W] (predicted next-step fields)

    Internally, we embed into Cl(2,0,0) with 4 blades:
    - blade 0 (scalar): vorticity field
    - blade 1 (e1): stream function x-component
    - blade 2 (e2): stream function y-component  
    - blade 3 (e12): bivector (curl proxy)
    """

    def __init__(self, n_blocks: int = 4, channels: int = 16, modes: int = 8):
        super().__init__()
        self.n_blades = 4  # Cl(2,0,0)

        # Input lift: [B, 2, H, W] → [B, C, 4, H, W]
        self.lift = nn.Linear(2, channels * self.n_blades)

        # FNO blocks
        self.blocks = nn.ModuleList([
            CliffordFNOBlock2d(channels, modes, self.n_blades)
            for _ in range(n_blocks)
        ])

        # Output projection: [B, C, 4, H, W] → [B, 2, H, W]
        self.project = nn.Linear(channels * self.n_blades, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 2, H, W]"""
        B, _, H, W = x.shape

        # Lift to multivector channels
        x = x.permute(0, 2, 3, 1)  # [B, H, W, 2]
        x = self.lift(x)  # [B, H, W, C*4]
        x = x.reshape(B, H, W, -1, self.n_blades)  # [B, H, W, C, 4]
        x = x.permute(0, 3, 4, 1, 2)  # [B, C, 4, H, W]

        # FNO blocks
        for block in self.blocks:
            x = block(x)

        # Project back
        x = x.permute(0, 3, 4, 1, 2).reshape(B, H, W, -1)  # [B, H, W, C*4]
        x = self.project(x)  # [B, H, W, 2]
        x = x.permute(0, 3, 1, 2)  # [B, 2, H, W]

        return x


# ---------------------------------------------------------------------------
# Training loop with GWS vs cosine comparison
# ---------------------------------------------------------------------------

def run_benchmark(args):
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Build model
    model = CliffordFNO2d(
        n_blocks=args.blocks,
        channels=args.channels,
        modes=args.modes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"CliffordFNO2d parameters: {n_params:,}")
    print(f"Cl(2,0,0) with 4 blades, {args.blocks} blocks, {args.channels} channels, {args.modes} modes")

    # Identify multivector params (weight list items in CliffordSpectralConv2d)
    mv_param_names = set()
    for name, param in model.named_parameters():
        # CliffordSpectralConv2d weight items: blocks.N.spec.weight.N
        # These are [in_channels, out_channels, modes, modes, 2] — NOT multivector by last dim
        # Instead, we group the 4 weight items per spectral conv as blades
        pass
    # For this architecture, multivector structure is in the spectral conv weight lists.
    # Each CliffordSpectralConv2d has self.weight = ParameterList of 4 items (n_blades=4).
    # We identify them by naming pattern.
    for name, param in model.named_parameters():
        if 'spec.weight' in name:
            mv_param_names.add(name)
    print(f"Multivector-valued (spectral conv weight) parameters: {len(mv_param_names)}")

    # Build GWS engine (small scheduling algebra)
    sched_engine = CliffordEngine(k=args.k_s, device='cpu', dtype=torch.float64)

    # --- GWS Optimizer ---
    gws_schedule = GradeRotorSchedule(
        k_s=args.k_s,
        n_grades=3,  # scalar, vector, bivector for Cl(2,0,0)
        T=args.steps,
        eta_max=args.lr,
        eta_min=args.lr * 0.01,
        phase_offsets=[0.0, 0.15, 0.35],  # stagger grades
        bivector_assignment='orthogonal',
        warmup_steps=args.warmup,
    )

    # --- Baseline cosine schedule ---
    cosine_schedule = CosineSchedule(
        eta_max=args.lr,
        T=args.steps,
        eta_min=args.lr * 0.01,
    )

    # --- Create two copies of the model for fair comparison ---
    import copy
    model_gws = copy.deepcopy(model).to(device)
    model_cos = copy.deepcopy(model).to(device)

    opt_gws = torch.optim.AdamW(model_gws.parameters(), lr=args.lr, weight_decay=0.01)
    opt_cos = torch.optim.AdamW(model_cos.parameters(), lr=args.lr, weight_decay=0.01)

    loss_fn = nn.MSELoss()

    # Output
    out_dir = Path("gaflowlm/gws/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"benchmark_ns_c{args.channels}_b{args.blocks}_m{args.modes}.jsonl"
    print(f"Logging to: {out_file}")

    records = []
    start = time.time()
    T = args.steps

    for step in range(1, T + 1):
        # Generate data
        x, y = generate_ns_batch(args.batch, args.grid, 1, str(device))
        x = x.to(device)
        y = y.to(device)

        # --- GWS step ---
        gws_lrs = gws_schedule(step)
        base_lr = cosine_schedule(step)

        # Apply per-grade scaling to GWS model gradients
        opt_gws.zero_grad()
        pred_gws = model_gws(x)
        loss_gws = loss_fn(pred_gws, y)
        loss_gws.backward()

        # Grade-wise LR scaling on spectral conv weight parameters.
        # Each CliffordSpectralConv2d has self.weight = ParameterList of 4 items.
        # Item 0 (scalar blade), items 1-2 (vector blades), item 3 (bivector blade).
        # We scale the gradient of each blade's parameter independently.
        grade_scales = {}
        if len(gws_lrs) >= 3 and base_lr > 1e-12:
            grade_scales = {
                0: gws_lrs[0] / base_lr,  # scalar blade
                1: gws_lrs[1] / base_lr,  # vector blades (1, 2)
                2: gws_lrs[2] / base_lr,  # bivector blade (3)
            }

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

        # --- Cosine step ---
        for pg in opt_cos.param_groups:
            pg['lr'] = base_lr

        opt_cos.zero_grad()
        pred_cos = model_cos(x)
        loss_cos = loss_fn(pred_cos, y)
        loss_cos.backward()
        opt_cos.step()

        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - start
            sps = step / max(1, elapsed)

            # Per-grade gradient norms for GWS model
            grade_total_norms = {0: 0.0, 1: 0.0, 2: 0.0}
            for name, param in model_gws.named_parameters():
                if param.grad is None:
                    continue
                g_norm_sq = param.grad.data.norm().item() ** 2
                if 'spec.weight.0' in name:
                    grade_total_norms[0] += g_norm_sq
                elif 'spec.weight.1' in name or 'spec.weight.2' in name:
                    grade_total_norms[1] += g_norm_sq
                elif 'spec.weight.3' in name:
                    grade_total_norms[2] += g_norm_sq
            for g in grade_total_norms:
                grade_total_norms[g] = math.sqrt(grade_total_norms[g])

            record = {
                'step': step,
                'loss_gws': float(loss_gws.detach()),
                'loss_cos': float(loss_cos.detach()),
                'lr_gws': gws_lrs if isinstance(gws_lrs, list) else [base_lr],
                'lr_cos': float(base_lr),
                'grade_norms_gws': {str(g): grade_total_norms[g] for g in grade_total_norms},
            }
            records.append(record)

            print(
                f"Step {step:>5d} | "
                f"GWS={float(loss_gws):.6f} COS={float(loss_cos):.6f} | "
                f"lr={float(base_lr):.2e} | "
                f"scalar={grade_total_norms[0]:.4f} vec={grade_total_norms[1]:.4f} bivec={grade_total_norms[2]:.4f} | "
                f"{sps:.1f} steps/s"
            )

    # Save
    with open(out_file, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print(f"\nBenchmark complete. {len(records)} records saved to {out_file}")

    # Summary comparison
    if len(records) >= 2:
        final_gws = records[-1]['loss_gws']
        final_cos = records[-1]['loss_cos']
        best_gws = min(r['loss_gws'] for r in records)
        best_cos = min(r['loss_cos'] for r in records)
        print(f"\n--- Final comparison ---")
        print(f"  GWS:    final={final_gws:.6f}  best={best_gws:.6f}")
        print(f"  Cosine: final={final_cos:.6f}  best={best_cos:.6f}")
        if best_gws < best_cos:
            print(f"  → GWS wins by {(best_cos - best_gws) / best_cos * 100:.1f}%")
        else:
            print(f"  → Cosine wins by {(best_gws - best_cos) / best_gws * 100:.1f}%")


def main():
    p = argparse.ArgumentParser(description="GWS benchmark on 2D Navier-Stokes")
    p.add_argument("--channels", type=int, default=16, help="Channel width")
    p.add_argument("--blocks", type=int, default=4, help="Number of FNO blocks")
    p.add_argument("--modes", type=int, default=8, help="Fourier modes")
    p.add_argument("--grid", type=int, default=32, help="Spatial grid size")
    p.add_argument("--steps", type=int, default=2000, help="Training steps")
    p.add_argument("--batch", type=int, default=4, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    p.add_argument("--k-s", type=int, default=2, help="Scheduling algebra dimension")
    p.add_argument("--warmup", type=int, default=100, help="Warmup steps")
    p.add_argument("--log-interval", type=int, default=50, help="Log every N steps")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run_benchmark(args)


if __name__ == "__main__":
    main()
