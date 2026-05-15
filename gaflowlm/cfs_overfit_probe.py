#!/usr/bin/env python3
"""Quick overfit probe for the flow-style CFS model.

Trains CFS on one fixed synthetic or GSM8K-test batch and reports whether
the loss drops. This is the fastest practical check for "can it learn?"
without a full benchmark.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flash_attn_mock
sys.modules["flash_attn"] = flash_attn_mock
sys.modules["flash_attn.layers"] = flash_attn_mock
sys.modules["flash_attn.layers.rotary"] = flash_attn_mock.layers.rotary

import torch

import dataloader

from models.cfs_model import CFSAlgorithm
from clifford.engine import CliffordEngine


def parse_args():
    p = argparse.ArgumentParser(description="CFS fixed-batch overfit probe")
    p.add_argument("--data", default="synthetic", choices=["synthetic", "gsm8k_test"])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--vocab-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--clifford-k", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--time-sampling", default="uniform", choices=["uniform", "cosine", "quadratic", "beta"])
    p.add_argument("--loss", default="mse", choices=["mse", "l1"])
    p.add_argument("--sample-steps", type=int, default=16)
    return p.parse_args()


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    if arch not in set(torch.cuda.get_arch_list()):
        return torch.device("cpu")
    return torch.device("cuda")


@dataclass
class Config:
    algo: object = field(default_factory=lambda: type("A", (), {}))
    model: object = field(default_factory=lambda: type("M", (), {}))
    optim: object = field(default_factory=lambda: type("O", (), {}))
    data: object = field(default_factory=lambda: type("D", (), {})())
    loader: object = field(default_factory=lambda: type("L", (), {})())


class FixedTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size


def _make_real_batch(args):
    cfg = Config()
    cfg.data.tokenizer_name_or_path = "gpt2"
    cfg.data.separator = "\n"
    cfg.data.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    cfg.data.data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "gsm8k_test.json",
    )
    tokenizer = dataloader.get_tokenizer(cfg)

    with open(cfg.data.data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if len(records) < args.batch_size:
        raise ValueError(
            f"gsm8k_test has only {len(records)} records, "
            f"cannot build batch_size={args.batch_size}"
        )

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    sep_ids = tokenizer(cfg.data.separator, add_special_tokens=False).input_ids

    tokens = []
    masks = []
    for example in records[: args.batch_size]:
        q_ids = tokenizer(example["prompt"].strip(), add_special_tokens=False).input_ids
        a_ids = tokenizer(
            example["response_ground_truth"].strip(),
            add_special_tokens=False,
        ).input_ids

        ids = [bos] + q_ids + sep_ids + a_ids + [eos]
        if len(ids) > args.seq_len:
            ids = ids[: args.seq_len]
        else:
            ids = ids + [pad] * (args.seq_len - len(ids))
        mask = [1 if tok != pad else 0 for tok in ids]
        tokens.append(ids)
        masks.append(mask)

    return tokenizer, torch.tensor(tokens, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


def run_overfit_probe(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    cfg = Config()
    cfg.algo.rhf_clifford_k = args.clifford_k
    cfg.algo.cfs_sample_steps = args.sample_steps
    cfg.algo.cfs_time_sampling = args.time_sampling
    cfg.algo.cfs_loss = args.loss
    cfg.algo.cfs_noise_scale = 1.0
    cfg.algo.cfs_normalize_noise = False
    cfg.algo.cfs_use_higher_order = True
    cfg.model.hidden_size = args.hidden_size
    cfg.model.n_blocks = args.n_blocks
    cfg.model.n_heads = args.n_heads
    cfg.model.length = args.seq_len
    cfg.optim.lr = args.lr
    cfg.optim.weight_decay = 0.0

    if args.data == "gsm8k_test":
        tokenizer, x, attention_mask = _make_real_batch(args)
    else:
        tokenizer = FixedTokenizer(args.vocab_size)
        x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
        attention_mask = None

    engine = CliffordEngine(k=args.clifford_k, dtype=torch.float64)
    algo = CFSAlgorithm(cfg, tokenizer, engine=engine).to(device)
    x = x.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        start = algo.evaluate(x, attention_mask=attention_mask)["loss"]

    losses = [start]
    for step in range(1, args.steps + 1):
        result = algo.train_step(x, attention_mask=attention_mask)
        losses.append(result["loss"])

    with torch.no_grad():
        end = algo.evaluate(x, attention_mask=attention_mask)["loss"]

    return {
        "device": str(device),
        "start_loss": start,
        "end_loss": end,
        "delta": end - start,
        "min_train_loss": min(losses),
    }


def main():
    args = parse_args()
    result = run_overfit_probe(args)
    print(f"device={result['device']} start_loss={result['start_loss']:.4f}")
    print(f"end_loss={result['end_loss']:.4f}")
    print(f"delta={result['delta']:+.4f}")
    print(f"min_train_loss={result['min_train_loss']:.4f}")

    if result["end_loss"] >= result["start_loss"]:
        raise SystemExit("CFS did not overfit this fixed batch")


if __name__ == "__main__":
    main()
