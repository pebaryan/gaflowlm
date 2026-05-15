"""
CFS: time-conditioned multivector flow model.

This is the flow-style CFS variant:
- tokens are embedded into Clifford space
- CARE provides rotor-based positional encoding
- CFA blocks operate on multivectors
- the network predicts a multivector velocity field

Training uses a rectified-flow-style objective on multivector trajectories
constructed from clean token embeddings and random multivector noise.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..schedulers import GWScheduler
except ImportError:  # pragma: no cover - fallback for direct module imports
    from schedulers import GWScheduler

from .cfs_arch import CFSTransformerBlock
from .care import CAREPositionEncoding


class CFSModel(nn.Module):
    """Time-conditioned multivector flow backbone for language sequences."""

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
        use_higher_order: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.k = k
        self.mv_dim = 1 << k
        self.n_blocks = n_blocks
        self.engine = engine
        self.use_higher_order = bool(use_higher_order or getattr(engine, "use_higher_order", False))

        dt = engine.cayley.dtype if engine else None

        self.token_embedding = nn.Embedding(vocab_size, hidden_size, dtype=dt)
        self.embed_to_clifford = nn.Linear(hidden_size, self.mv_dim, dtype=dt)

        self.time_feature_dim = max(8, hidden_size // 4)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_feature_dim, hidden_size, dtype=dt),
            nn.SiLU(),
            nn.Linear(hidden_size, self.mv_dim, dtype=dt),
        )

        if engine is not None:
            self.care = CAREPositionEncoding(k=k, max_len=max_len, engine=engine)

        self.blocks = nn.ModuleList(
            [
                CFSTransformerBlock(
                    mv_dim=self.mv_dim,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    engine=engine,
                    dropout=dropout,
                    use_higher_order=self.use_higher_order,
                )
                for _ in range(n_blocks)
            ]
        )

        self.mv_norm = nn.LayerNorm(self.mv_dim, dtype=dt)
        self.embed_norm = nn.LayerNorm(hidden_size, dtype=dt)
        self.velocity_head = nn.Linear(self.mv_dim, self.mv_dim, bias=False, dtype=dt)
        self.dropout = nn.Dropout(dropout)
        self.clifford_to_embed = nn.Linear(self.mv_dim, hidden_size, dtype=dt)

    def _time_features(self, t: torch.Tensor) -> torch.Tensor:
        """Build sinusoidal time features."""
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        t = t.to(dtype=self.embed_to_clifford.weight.dtype)

        half = self.time_feature_dim // 2
        if half == 0:
            return t

        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(1, half - 1)
        )
        angles = t * freqs.unsqueeze(0)
        feats = [torch.sin(angles), torch.cos(angles)]
        if self.time_feature_dim % 2 == 1:
            feats.append(t)
        return torch.cat(feats, dim=-1)

    def encode_tokens(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """Encode token IDs into clean multivectors."""
        h = self.token_embedding(x)
        h = self.dropout(h)
        mv = self.embed_to_clifford(h)
        if hasattr(self, "care"):
            mv = self.care(mv, pos=positions)
        return mv

    def decode_embedding(self, mv: torch.Tensor) -> torch.Tensor:
        """Project multivectors back into hidden embedding space."""
        h = self.clifford_to_embed(mv)
        return self.embed_norm(h)

    def decode_logits(self, mv: torch.Tensor) -> torch.Tensor:
        """Optional token logits from a multivector state."""
        h = self.decode_embedding(mv)
        logits = torch.matmul(h, self.token_embedding.weight.t())
        return logits / math.sqrt(self.hidden_size)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor = None,
        positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predict the multivector velocity at time `t`."""
        if xt.ndim != 3:
            raise ValueError(f"Expected [B, L, mv_dim] multivector input, got {xt.shape}")

        B, L, _ = xt.shape
        t_feat = self._time_features(t)
        t_bias = self.time_mlp(t_feat).unsqueeze(1).expand(-1, L, -1)

        x = xt + t_bias
        x = self.dropout(x)
        if hasattr(self, "care"):
            x = self.care(x, pos=positions)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.mv_norm(x)
        x = self.dropout(x)
        velocity = self.velocity_head(x)
        return velocity


class CFSAlgorithm:
    """Flow-style training wrapper for the CFS model."""

    def __init__(self, config, tokenizer, engine=None):
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else tokenizer.vocab_size

        hidden_size = getattr(config.model, "hidden_size", 256)
        k = getattr(config.algo, "rhf_clifford_k", 8)
        n_blocks = getattr(config.model, "n_blocks", 4)
        n_heads = getattr(config.model, "n_heads", 8)
        max_len = getattr(config.model, "length", 512)

        assert engine is not None, "CFSAlgorithm requires a CliffordEngine"
        self.engine = engine

        self.model = CFSModel(
            vocab_size=self.vocab_size,
            hidden_size=hidden_size,
            k=k,
            n_blocks=n_blocks,
            n_heads=n_heads,
            ff_dim=hidden_size * 4,
            engine=self.engine,
            max_len=max_len,
            use_higher_order=bool(getattr(config.algo, "cfs_use_higher_order", False)),
        )
        self.model.train()

        self.device = torch.device("cpu")
        self.loss_type = getattr(config.algo, "cfs_loss", "mse")
        self.time_sampling = getattr(config.algo, "cfs_time_sampling", "uniform")
        self.time_beta = float(getattr(config.algo, "cfs_time_beta", 2.0))
        self.flow_noise_scale = getattr(config.algo, "cfs_noise_scale", 1.0)
        self.normalize_noise = getattr(config.algo, "cfs_normalize_noise", False)
        self.sample_steps = int(getattr(config.algo, "cfs_sample_steps", 32))
        self.use_higher_order = bool(getattr(config.algo, "cfs_use_higher_order", False))
        optim_cfg = getattr(config, "optim", None)
        algo_cfg = getattr(config, "algo", None)

        def _cfg_value(name: str, default):
            for section in (optim_cfg, algo_cfg, config):
                if section is not None and hasattr(section, name):
                    return getattr(section, name)
            return default

        self.use_gws = bool(_cfg_value("use_gws", False))
        self.gws_num_grades = int(_cfg_value("gws_num_grades", 4))
        self.gws_phase_stagger = bool(_cfg_value("gws_phase_stagger", True))
        self.gws_learnable_phase_offsets = bool(
            _cfg_value("gws_learnable_phase_offsets", False)
        )
        self.gws_phase_step = float(_cfg_value("gws_phase_step", 0.4 * math.pi))
        self.gws_phase_offsets = _cfg_value("gws_phase_offsets", None)
        self.gws_phase_update_lr = float(_cfg_value("gws_phase_update_lr", 0.02))
        self.gws_total_steps = int(
            _cfg_value(
                "gws_total_steps",
                getattr(getattr(config, "trainer", None), "max_steps", 1000),
            )
        )

        self.optimizer, self.scheduler = self._create_grade_aware_optimizer()

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.engine = self.engine.to(device)
        self.model.engine = self.engine
        if hasattr(self.model, "care"):
            self.model.care.engine = self.engine
        if self.scheduler is not None:
            self.scheduler = self.scheduler.to(device)
        return self

    def _infer_multivector_axes(self, param: torch.Tensor) -> tuple[int, ...]:
        """Find tensor axes whose extent matches the Clifford blade dimension."""
        mv_axes = tuple(i for i, size in enumerate(param.shape) if size == self.model.mv_dim)
        if mv_axes:
            return mv_axes
        if param.ndim > 0 and param.shape[-1] == self.model.mv_dim:
            return (param.ndim - 1,)
        return tuple()

    def _create_grade_aware_optimizer(self):
        """Create AdamW plus an optional GWScheduler.

        Multivector parameters stay in one optimizer bucket, but the scheduler
        applies per-grade scaling to their gradient components before each step.
        That keeps the implementation stable while preserving the grade-wise
        effect we want to test.
        """
        base_lr = getattr(self.config.optim, "lr", 3e-4)
        weight_decay = getattr(self.config.optim, "weight_decay", 0.0)

        scalar_params = []
        mv_params = []
        mv_axes_map: dict[int, tuple[int, ...]] = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            mv_axes = self._infer_multivector_axes(param)
            if mv_axes:
                mv_params.append(param)
                mv_axes_map[id(param)] = mv_axes
            else:
                scalar_params.append(param)

        param_groups = []
        if scalar_params:
            param_groups.append(
                {
                    "params": scalar_params,
                    "lr": base_lr,
                    "base_lr": base_lr,
                    "grade_id": "scalar",
                }
            )
        if mv_params:
            param_groups.append(
                {
                    "params": mv_params,
                    "lr": base_lr,
                    "base_lr": base_lr,
                    "grade_id": "multivector",
                }
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
        )

        if not self.use_gws or not mv_params:
            return optimizer, None

        scheduler = GWScheduler(
            optimizer=optimizer,
            engine=self.engine,
            total_steps=self.gws_total_steps,
            num_grades=self.gws_num_grades,
            phase_offsets=self.gws_phase_offsets,
            phase_stagger=self.gws_phase_stagger,
            learnable_phase_offsets=self.gws_learnable_phase_offsets,
            phase_step=self.gws_phase_step,
            phase_update_lr=self.gws_phase_update_lr,
            eta_min=float(
                getattr(
                    getattr(self.config, "optim", None),
                    "gws_eta_min",
                    getattr(getattr(self.config, "algo", None), "gws_eta_min", 0.0),
                )
            ),
            multivector_axes=mv_axes_map,
        )
        return optimizer, scheduler

    def _sample_positions(self, batch: torch.Tensor) -> torch.Tensor:
        B, L = batch.shape
        return torch.arange(L, device=self.device).unsqueeze(0).expand(B, -1)

    def _pairwise_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        token_mask = attention_mask.to(dtype=torch.bool)
        return token_mask.unsqueeze(1) & token_mask.unsqueeze(2)

    def _sample_time(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        u = torch.rand(batch_size, 1, device=self.device, dtype=dtype)
        if self.time_sampling == "uniform":
            return u
        if self.time_sampling == "cosine":
            return 0.5 - 0.5 * torch.cos(math.pi * u)
        if self.time_sampling == "quadratic":
            return u.pow(2)
        if self.time_sampling == "beta":
            dist = torch.distributions.Beta(self.time_beta, self.time_beta)
            return dist.sample((batch_size, 1)).to(device=self.device, dtype=dtype)
        raise ValueError(f"Unknown CFS time sampling schedule: {self.time_sampling}")

    def _sample_flow_batch(self, x0: torch.Tensor, attention_mask: torch.Tensor = None):
        positions = self._sample_positions(x0)
        m0 = self.model.encode_tokens(x0, positions=positions)

        noise = torch.randn_like(m0) * self.flow_noise_scale
        if self.normalize_noise:
            noise = self.engine.normalize_multivector(noise)

        if attention_mask is not None:
            seq_mask = attention_mask.to(device=m0.device, dtype=m0.dtype).unsqueeze(-1)
            m0 = m0 * seq_mask
            noise = noise * seq_mask

        B = x0.shape[0]
        t = self._sample_time(B, m0.dtype)
        xt = (1 - t.unsqueeze(-1)) * m0 + t.unsqueeze(-1) * noise
        target_velocity = noise - m0

        return positions, t, xt, target_velocity

    def _flow_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        raise ValueError(f"Unknown CFS flow loss: {self.loss_type}")

    def _masked_flow_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return self._flow_loss(pred, target)

        seq_mask = attention_mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
        if self.loss_type == "mse":
            per_elem = (pred - target) ** 2
        elif self.loss_type == "l1":
            per_elem = (pred - target).abs()
        else:
            raise ValueError(f"Unknown CFS flow loss: {self.loss_type}")

        weighted = per_elem * seq_mask
        denom = seq_mask.sum().clamp(min=1.0) * pred.shape[-1]
        return weighted.sum() / denom

    def train_step(self, x0: torch.Tensor, attention_mask: torch.Tensor = None) -> dict:
        x0 = x0.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        positions, t, xt, target_velocity = self._sample_flow_batch(
            x0, attention_mask=attention_mask
        )
        pair_mask = None if attention_mask is None else self._pairwise_mask(attention_mask)
        pred_velocity = self.model(xt, t, mask=pair_mask, positions=positions)
        loss = self._masked_flow_loss(pred_velocity, target_velocity, attention_mask)

        self.optimizer.zero_grad()
        loss.backward()
        if self.scheduler is not None:
            self.scheduler.scale_gradients()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": loss.item()}

    @torch.no_grad()
    def evaluate(self, x0: torch.Tensor, attention_mask: torch.Tensor = None) -> dict:
        self.model.eval()
        x0 = x0.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        positions, t, xt, target_velocity = self._sample_flow_batch(
            x0, attention_mask=attention_mask
        )
        pair_mask = None if attention_mask is None else self._pairwise_mask(attention_mask)
        pred_velocity = self.model(xt, t, mask=pair_mask, positions=positions)
        loss = self._masked_flow_loss(pred_velocity, target_velocity, attention_mask)

        self.model.train()
        return {"loss": loss.item()}

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor = None,
        seq_len: int = None,
        num_steps: int = None,
        positions: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run reverse-time Euler integration from noise to a denoised state."""
        self.model.eval()

        if num_steps is None:
            num_steps = self.sample_steps
        if x0 is not None:
            x0 = x0.to(self.device)
            B, seq_len = x0.shape
            positions = self._sample_positions(x0) if positions is None else positions.to(self.device)
            state = self.model.encode_tokens(x0, positions=positions)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        else:
            if seq_len is None:
                raise ValueError("sample() requires seq_len when x0 is not provided")
            B = 1 if positions is None else positions.shape[0]
            if positions is None:
                positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(B, -1)
            state = torch.randn(
                B,
                seq_len,
                self.model.mv_dim,
                device=self.device,
                dtype=self.model.embed_to_clifford.weight.dtype,
            )
            state = self.engine.normalize_multivector(state)

        dt = 1.0 / max(1, num_steps)
        for step in range(num_steps):
            t = torch.full((state.shape[0], 1), 1.0 - step * dt, device=self.device, dtype=state.dtype)
            pair_mask = None if attention_mask is None else self._pairwise_mask(attention_mask)
            velocity = self.model(state, t, mask=pair_mask, positions=positions)
            state = state - dt * velocity

        logits = self.model.decode_logits(state)
        self.model.train()
        return state, logits

    @torch.no_grad()
    def sample_tokens(
        self,
        x0: torch.Tensor = None,
        seq_len: int = None,
        num_steps: int = None,
        positions: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run flow sampling and decode token ids."""
        state, logits = self.sample(
            x0=x0,
            seq_len=seq_len,
            num_steps=num_steps,
            positions=positions,
            attention_mask=attention_mask,
        )
        tokens = logits.argmax(dim=-1)
        return state, logits, tokens

    @torch.no_grad()
    def benchmark_reconstruction(
        self,
        x0: torch.Tensor,
        step_list: list[int] = None,
        attention_mask: torch.Tensor = None,
    ) -> list[dict]:
        """Benchmark token reconstruction accuracy across sampling steps."""
        if step_list is None:
            step_list = [1, 2, 4, 8, 16, 32]

        x0 = x0.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        positions = self._sample_positions(x0)
        mv_dim = self.model.mv_dim
        noise = torch.randn(
            x0.shape[0],
            x0.shape[1],
            mv_dim,
            device=self.device,
            dtype=self.model.embed_to_clifford.weight.dtype,
        )
        noise = noise * self.flow_noise_scale
        if self.normalize_noise:
            noise = self.engine.normalize_multivector(noise)
        if attention_mask is not None:
            seq_mask = attention_mask.to(dtype=noise.dtype).unsqueeze(-1)
            noise = noise * seq_mask

        results = []
        for num_steps in step_list:
            state = noise.clone()
            dt = 1.0 / max(1, num_steps)
            for step in range(num_steps):
                cur_t = torch.full(
                    (x0.shape[0], 1),
                    1.0 - step * dt,
                    device=self.device,
                    dtype=noise.dtype,
                )
                pair_mask = None if attention_mask is None else self._pairwise_mask(attention_mask)
                velocity = self.model(state, cur_t, mask=pair_mask, positions=positions)
                state = state - dt * velocity

            logits = self.model.decode_logits(state)
            tokens = logits.argmax(dim=-1)
            if attention_mask is None:
                accuracy = (tokens == x0).float().mean().item()
            else:
                token_mask = attention_mask.to(dtype=torch.bool)
                accuracy = ((tokens == x0) & token_mask).float().sum().div(
                    token_mask.float().sum().clamp(min=1.0)
                ).item()
            results.append(
                {
                    "steps": num_steps,
                    "accuracy": accuracy,
                    "logit_mean": logits.mean().item(),
                    "pred_tokens": tokens[0].detach().cpu().tolist(),
                    "target_tokens": x0[0].detach().cpu().tolist(),
                }
            )

        self.model.train()
        return results

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()
