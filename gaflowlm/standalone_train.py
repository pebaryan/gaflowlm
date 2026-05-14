#!/usr/bin/env python3
"""Standalone RHF vs SFM training comparison on TinyGSM.

Minimal training loop (no Hydra/Lightning) for rapid experimentation.
Supports: CPU, single-GPU, data-parallel multi-GPU.

Usage:
    python standalone_train.py --algo rhf --mode analytic --steps 5000
    python standalone_train.py --algo sfm --steps 5000
|    python standalone_train.py --algo rhf --mode clifford --clifford-k 8 --steps 5000
|    python standalone_train.py --algo cfs --clifford-k 4 --steps 5000
"""
import argparse
import os
import sys
import time
import json
import math

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock flash_attn — installed Triton kernels fail with this CUDA/PyTorch.
import flash_attn_mock
sys.modules['flash_attn'] = flash_attn_mock
sys.modules['flash_attn.layers'] = flash_attn_mock
sys.modules['flash_attn.layers.rotary'] = flash_attn_mock.layers.rotary

import torch
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Optional

import algo
import rhf_algo
import dataloader
import utils
from models.cfs_model import CFSAlgorithm
from clifford.engine import CliffordEngine


def parse_args():
    p = argparse.ArgumentParser(description="RHF vs SFM standalone training")
    p.add_argument("--algo", choices=["sfm", "rhf", "cfs"], default="rhf")
    p.add_argument("--mode", choices=["analytic", "clifford"], default="analytic",
                    help="RHF mode: analytic (numerically identical to SFM) or clifford (GA projection)")
    p.add_argument("--clifford-k", type=int, default=8, help="Cl(k,0,0) dimension for clifford mode")
    p.add_argument("--steps", type=int, default=5000, help="Training steps")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--hidden-size", type=int, default=256, help="Model hidden size")
    p.add_argument("--n-blocks", type=int, default=4, help="Number of transformer blocks")
    p.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    p.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    p.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    p.add_argument("--seed", type=int, default=1, help="Random seed")
    p.add_argument("--slerp-precision", choices=["float32", "float64"], default="float64")
    p.add_argument("--wandb", action="store_true", help="Log to wandb")
    p.add_argument("--wandb-project", default="gaflowlm")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--eval-every", type=int, default=500, help="Evaluate every N steps")
    p.add_argument("--save-dir", default=None, help="Directory to save checkpoints")
    p.add_argument("--data", default="tinygsm", choices=["tinygsm", "synthetic"])
    p.add_argument("--vocab-size", type=int, default=None, help="Override vocab size (for synthetic data)")
    return p.parse_args()


class SimpleTokenizer:
    """GPT-2 tokenizer wrapper for TinyGSM."""
    def __init__(self, name_or_path="gpt2"):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(name_or_path)
        self._vocab_size = self._tok.vocab_size

    def __len__(self):
        return self._vocab_size

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def eos_token_id(self):
        return self._tok.eos_token_id

    @property
    def bos_token_id(self):
        return self._tok.bos_token_id

    @property
    def pad_token_id(self):
        return self._tok.pad_token_id or self._tok.eos_token_id

    @property
    def mask_token_id(self):
        return None

    def __call__(self, text, **kwargs):
        return self._tok(text, **kwargs)

    def decode(self, ids, **kwargs):
        return self._tok.decode(ids, **kwargs)

    def batch_decode(self, batch_ids, **kwargs):
        return self._tok.batch_decode(batch_ids, **kwargs)


@dataclass
class SimpleConfig:
    """Minimal config to create an SFM/RHFSFM model."""
    algo: object = field(default_factory=lambda: type('A', (), {
        'name': 'rhf', 'diffusion_type': 'sphere', 'backbone': 'sphere-arch',
        'parameterization': 'mean', 'time_conditioning': True, 'loss_type': 'ce',
        'T': 0, 'causal_attention': False, 'adaLN': True, 'slerp_precision': 'float64',
        'eps': 1e-6, 'invert_time_convention': True, 'renormalize_weights': True,
        'rhf_mode': 'analytic', 'rhf_clifford_k': 8, 'rhf_compute_bivector': False,
    }))
    model: object = field(default_factory=lambda: type('M', (), {
        'name': 'rhf-tiny', 'type': 'sphere-arch', 'hidden_size': 256,
        'cond_dim': 128, 'length': 256, 'n_blocks': 4, 'n_heads': 4,
        'dropout': 0.0, 'init': 'ngpt', 'learn_temperature_scaling': False,
        'eps': 1e-6, 'normalize_input_embed': True, 'rmsnorm': True,
        'mlp_type': 'gelu', 'pretrained_ckpt_path': None,
        'use_time_alpha': True, 'use_time_token': False,
    }))
    noise: object = field(default_factory=lambda: type('N', (), {
        'type': 'log-linear', 'eps': 1e-3, 'alpha_min': None, 'alpha_max': None,
        'adaptive': False,
    }))
    training: object = field(default_factory=lambda: type('T', (), {
        'ema': 0.9999, 'antithetic_sampling': True, 'sampling_eps': 1e-3,
        'finetune_path': '', 'loss_precision': 'bf16',
    }))
    sampler: object = field(default_factory=lambda: type('S', (), {
        'predictor': 'sfm', 'steps': 128, 'noise_removal': 'ancestral',
        'use_float64': True, 'velocity': 'exact', 'p_nucleus': 1.0,
        'top_k': -1, 'top_k_velocity': -1, 'temperature': 1.0,
    }))
    optim: object = field(default_factory=lambda: type('O', (), {
        'lr': 3e-4, 'weight_decay': 0.0, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8,
    }))
    eval: object = field(default_factory=lambda: type('E', (), {
        'checkpoint_path': '', 'strict_loading': True, 'disable_ema': False,
        'compute_generative_perplexity': False, 'perplexity_batch_size': 8,
        'gen_ppl_eval_model_name_or_path': 'gpt2',
    }))
    neg_infinity_mode: str = 'large-finite'
    seed: int = 1


def make_config(args):
    """Create a config from command-line args."""
    cfg = SimpleConfig()
    cfg.seed = args.seed
    cfg.neg_infinity_mode = 'large-finite'

    # Algo config
    cfg.algo.name = args.algo
    cfg.algo.slerp_precision = args.slerp_precision
    if args.algo in ("rhf", "cfs"):
        cfg.algo.rhf_mode = args.mode
        cfg.algo.rhf_clifford_k = args.clifford_k

    # Model config
    cfg.model.hidden_size = args.hidden_size
    cfg.model.n_blocks = args.n_blocks
    cfg.model.n_heads = args.n_heads
    cfg.model.length = args.seq_len
    cfg.model.cond_dim = args.hidden_size // 2

    return cfg


class SyntheticDataLoader:
    """Simple synthetic data for quick tests."""
    def __init__(self, vocab_size, seq_len, batch_size):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

    def __len__(self):
        return 10000  # effectively infinite


def train_step(model, x0, device):
    """Single training step. Returns (loss_value, loss_tensor_for_backward)."""
    B, L = x0.shape
    t_raw = torch.rand(B, device=device)
    dalpha_t, alpha_t = model.noise(t_raw)
    alpha_t_expanded = alpha_t.unsqueeze(-1)

    xt = model.q_xt(x0, alpha_t_expanded, use_pure_noise=False)
    sigma = model._sigma_from_alphat(alpha_t_expanded)

    log_p = model.forward(x0=x0, xt=xt, sigma=sigma)
    ce_loss = -log_p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    loss = ce_loss.mean()

    return loss.item(), loss


@torch.no_grad()
def eval_step(model, dataloader, device, cfg, max_batches=10):
    """Evaluate on a few batches."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        x0 = batch.to(device)
        B, L = x0.shape

        t_raw = torch.rand(B, device=device)
        dalpha_t, alpha_t = model.noise(t_raw)
        alpha_t_expanded = alpha_t.unsqueeze(-1)
        xt = model.q_xt(x0, alpha_t_expanded, use_pure_noise=False)
        sigma = model._sigma_from_alphat(alpha_t_expanded)
        log_p = model.forward(x0=x0, xt=xt, sigma=sigma)
        ce_loss = -log_p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        total_loss += ce_loss.mean().item()
        n_batches += 1

    model.train()
    return total_loss / max(1, n_batches)


@torch.no_grad()
def eval_step_cfs(model, dataloader, device, max_batches=10):
    """Evaluate CFS model on a few batches."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        x0 = batch.to(device)
        result = model.evaluate(x0)
        total_loss += result['loss']
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


def main():
    args = parse_args()
    cfg = make_config(args)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Tokenizer and data
    if args.data == "tinygsm":
        print("Loading GPT-2 tokenizer...")
        tokenizer = SimpleTokenizer("gpt2")
        vocab_size = tokenizer.vocab_size
        # Note: full TinyGSM data loading would require Hydra config.
        # For standalone training, use synthetic data in the same format.
        print(f"Vocab size: {vocab_size}")
        print("Using synthetic data (TinyGSM download requires Hydra pipeline)")
        data_iter = SyntheticDataLoader(vocab_size, args.seq_len, args.batch_size)
    else:
        vocab_size = args.vocab_size or 1000
        # Minimal tokenizer
        class VocabOnly:
            def __len__(self): return vocab_size
            @property
            def vocab_size(self): return vocab_size
            @property
            def eos_token_id(self): return 1
            @property
            def bos_token_id(self): return 0
            @property
            def pad_token_id(self): return 2
            @property
            def mask_token_id(self): return None
        tokenizer = VocabOnly()
        data_iter = SyntheticDataLoader(vocab_size, args.seq_len, args.batch_size)

    # Create model
    print(f"Creating {args.algo.upper()} model (mode={getattr(cfg.algo, 'rhf_mode', 'N/A')})...")
    is_cfs = args.algo == "cfs"
    if is_cfs:
        engine = CliffordEngine(k=args.clifford_k, dtype=torch.float64)
        model = CFSAlgorithm(cfg, tokenizer, engine=engine)
        model = model.to(device)
    elif args.algo == "rhf":
        model = rhf_algo.RHFSFM(cfg, tokenizer)
        model = model.to(device)
    else:
        cfg.algo.name = "sfm"
        model = algo.SFM(cfg, tokenizer)
        model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer — CFS handles its own optimizer internally
    if is_cfs:
        optimizer = None  # CFSAlgorithm has internal optimizer
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=cfg.optim.weight_decay,
            betas=(cfg.optim.beta1, cfg.optim.beta2),
            eps=cfg.optim.eps,
        )

    # Optional wandb
    wb = None
    if args.wandb:
        import wandb
        name = args.wandb_name or f"{args.algo}_{args.mode}_{args.hidden_size}d_{args.n_blocks}L"
        wb = wandb.init(project=args.wandb_project, name=name, config=vars(args))
        if hasattr(model, 'model'):
            wandb.watch(model.model, log="gradients", log_freq=100)

    # Training loop
    model.train()
    print(f"\nStarting training for {args.steps} steps (eval every {args.eval_every})...")
    data_iterator = iter(data_iter)
    start_time = time.time()
    best_loss = float("inf")
    loss_val = 0.0

    for step in range(1, args.steps + 1):
        batch = next(data_iterator)
        x0 = batch.to(device)

        if is_cfs:
            # CFS handles forward + backward internally
            result = model.train_step(x0)
            loss_val = result['loss']
        else:
            optimizer.zero_grad()
            loss_val, loss_tensor = train_step(model, x0, device)
            loss_tensor.backward()
            optimizer.step()

            if not is_cfs and hasattr(model, 'backbone') and hasattr(model.backbone, 'renormalize_weights'):
                if getattr(cfg.algo, 'renormalize_weights', False):
                    model.backbone.renormalize_weights()

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            print(f"Step {step:5d} | loss={loss_val:.4f} | "
                  f"{steps_per_sec:.1f} steps/s | elapsed={elapsed:.0f}s")
            if wb:
                wb.log({"train/loss": loss_val, "train/step": step,
                         "train/steps_per_sec": steps_per_sec})

        if args.eval_every > 0 and step % args.eval_every == 0:
            if is_cfs:
                eval_loss = eval_step_cfs(model, data_iter, device, max_batches=10)
            else:
                eval_loss = eval_step(model, data_iter, device, cfg, max_batches=10)
            print(f"  EVAL step {step}: loss={eval_loss:.4f}")
            if eval_loss < best_loss:
                best_loss = eval_loss
            if wb:
                wb.log({"eval/loss": eval_loss, "eval/step": step})

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete: {args.steps} steps in {elapsed:.0f}s ({args.steps/elapsed:.1f} steps/s)")
    print(f"Best eval loss: {best_loss:.4f}")
    print(f"Algo: {args.algo}, Mode: {args.mode}, Hidden: {args.hidden_size}, Blocks: {args.n_blocks}")

    # Save checkpoint
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        if is_cfs:
            ckpt_path = os.path.join(args.save_dir, f"cfs_k{args.clifford_k}_final.pt")
            torch.save({
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'config': vars(args),
                'final_loss': loss_val,
                'best_eval_loss': best_loss,
                'step': args.steps,
            }, ckpt_path)
        else:
            ckpt_path = os.path.join(args.save_dir, f"{args.algo}_{args.mode}_final.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(args),
                'final_loss': loss_val,
                'best_eval_loss': best_loss,
                'step': args.steps,
            }, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    if wb:
        wb.finish()


if __name__ == "__main__":
    main()