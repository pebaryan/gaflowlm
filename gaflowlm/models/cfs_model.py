"""
CFS Model — Clifford Flow on the Sphere.

Full multivector transformer with:
- EmbedToClifford projection
- CARE position encoding
- CFS transformer blocks (CFA + FFN)
- CliffordToEmbed projection
- Output head for vocabulary prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cfs_arch import CFSTransformerBlock
from .care import CAREPositionEncoding


class CFSModel(nn.Module):
    """CFS model: full multivector transformer for language modeling.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Embedding/hidden dimension.
        k: Clifford algebra dimension Cl(k,0,0).
        n_blocks: Number of CFS transformer blocks.
        n_heads: Number of attention heads.
        ff_dim: Feed-forward hidden dimension.
        engine: CliffordEngine instance.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        k: int = 8,
        n_blocks: int = 4,
        n_heads: int = 8,
        ff_dim: int = 1024,
        engine=None,
        max_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.k = k
        self.mv_dim = 1 << k  # 2^k
        self.n_blocks = n_blocks
        self.engine = engine

        dt = engine.cayley.dtype if engine else None

        # Token embedding (learned) — match dtype to engine
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, dtype=dt)

        # Project to Clifford space: d-dim → multivector
        # Learnable projection rather than simple truncation
        self.embed_to_clifford = nn.Linear(hidden_size, self.mv_dim, dtype=dt)

        # CARE position encoding
        if engine is not None:
            self.care = CAREPositionEncoding(
                k=k, max_len=max_len, engine=engine,
            )

        # CFS transformer blocks
        self.blocks = nn.ModuleList([
            CFSTransformerBlock(
                mv_dim=self.mv_dim, n_heads=n_heads,
                ff_dim=ff_dim, engine=engine, dropout=dropout,
            )
            for _ in range(n_blocks)
        ])

        # Project back: multivector → d-dim
        self.clifford_to_embed = nn.Linear(self.mv_dim, hidden_size, dtype=dt)

        # Layer norm before output
        self.norm = nn.LayerNorm(hidden_size, dtype=dt)

        # Output head (tied embeddings or separate)
        self.output_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=dt)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """CFS model forward.

        Args:
            x: [B, L] token IDs.
            mask: [B, L, L] attention mask or None.
            positions: [B, L] position indices or None (0..L-1).

        Returns:
            [B, L, vocab_size] logits.
        """
        B, L = x.shape

        # Token embedding: [B, L, d]
        h = self.token_embedding(x)
        h = self.dropout(h)

        # Project to Clifford space: [B, L, mv_dim]
        mv = self.embed_to_clifford(h)

        # CARE position encoding
        if hasattr(self, 'care'):
            mv = self.care(mv, pos=positions)

        # CFS transformer blocks
        for block in self.blocks:
            mv = block(mv, mask=mask)

        # Project back to embedding space: [B, L, d]
        h = self.clifford_to_embed(mv)
        h = self.norm(h)
        h = self.dropout(h)

        # Output head: [B, L, vocab_size]
        logits = self.output_head(h)

        return logits


class CFSAlgorithm:
    """Training wrapper for CFS model.

    Provides the same interface as SFM for the standalone training loop.
    CFS is a full model replacement — no SLERP/log-map needed in the flow.
    """

    def __init__(self, config, tokenizer, engine=None):
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        # Extract CFS params from config
        hidden_size = getattr(config.model, 'hidden_size', 256)
        k = getattr(config.algo, 'rhf_clifford_k', 8)
        n_blocks = getattr(config.model, 'n_blocks', 4)
        n_heads = getattr(config.model, 'n_heads', 8)
        max_len = getattr(config.model, 'length', 512)

        # Engine is required
        assert engine is not None, "CFSAlgorithm requires a CliffordEngine"
        self.engine = engine

        # Create model
        self.model = CFSModel(
            vocab_size=self.vocab_size,
            hidden_size=hidden_size,
            k=k,
            n_blocks=n_blocks,
            n_heads=n_heads,
            ff_dim=hidden_size * 4,
            engine=self.engine,
            max_len=max_len,
        )
        self.model.train()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=getattr(config.optim, 'lr', 3e-4),
            weight_decay=getattr(config.optim, 'weight_decay', 0.0),
        )

        self.device = torch.device('cpu')

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        return self

    def train_step(self, x0: torch.Tensor) -> dict:
        """Single training step.

        Args:
            x0: [B, L] token IDs.

        Returns:
            dict with 'loss' key.
        """
        x0 = x0.to(self.device)
        B, L = x0.shape

        # Forward pass
        logits = self.model(x0)

        # Cross-entropy loss (predict next token)
        # Shift: predict target at position i from context at position i-1
        logits = logits[:, :-1, :]  # [B, L-1, vocab]
        targets = x0[:, 1:]         # [B, L-1]

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    @torch.no_grad()
    def evaluate(self, x0: torch.Tensor) -> dict:
        """Evaluation step.

        Args:
            x0: [B, L] token IDs.

        Returns:
            dict with 'loss' key.
        """
        self.model.eval()
        x0 = x0.to(self.device)
        logits = self.model(x0)

        logits = logits[:, :-1, :]
        targets = x0[:, 1:]

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )
        self.model.train()
        return {'loss': loss.item()}

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()
