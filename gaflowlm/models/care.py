"""
CARE - Clifford Algebra Rotary Encoding.

Rotor-based positional encoding for multivector sequences in Cl(k,0,0).

For each position i, we construct a rotor by composing several simple rotor planes:
    R_i = Π_p exp(½ θ_p(i) B_p)

Where B_p are learned simple bivectors and θ_p(i) are position-dependent
angles (sinusoidal by default, can be learned).

The position encoding is applied as a rotor sandwich:
    M̃_i = R_i · M_i · R̃_i

This preserves the multivector norm and the geometric structure, unlike
additive position embeddings which break the algebraic properties.
"""

import math
import torch
import torch.nn as nn


class CAREPositionEncoding(nn.Module):
    """CARE position encoding via rotor sandwich."""

    def __init__(
        self,
        k: int,
        max_len: int = 2048,
        engine=None,
        learned_angles: bool = False,
        freq: float = 10000.0,
    ):
        super().__init__()
        self.k = k
        self.max_len = max_len
        self.engine = engine
        self.mv_dim = 1 << k

        if engine is not None:
            self.cayley = engine.cayley
            self.grade_masks = engine.grade_masks
            self._dtype = engine.cayley.dtype
            self._has_engine = True
        else:
            self._has_engine = False
            self._dtype = torch.float32

        self._plane_blades = self._select_bivector_blades(k)
        self._plane_names = self._make_plane_names(len(self._plane_blades))
        self._n_planes = len(self._plane_blades)

        if engine is not None:
            self.register_buffer("_scalar_mask", self._blade_mask(0))
            for idx, blade_idx in enumerate(self._plane_blades):
                name = self._plane_names[idx]
                self.register_buffer(f"_{name}_mask", self._blade_mask(blade_idx))
                B = torch.zeros(1, 1, self.mv_dim, dtype=self._dtype)
                B[..., blade_idx] = torch.randn(1, 1, 1, dtype=self._dtype) * 0.02
                setattr(self, name, nn.Parameter(B))
        else:
            self.register_buffer("_scalar_mask", torch.zeros(1, 1, self.mv_dim, dtype=self._dtype))
            for idx, blade_idx in enumerate(self._plane_blades):
                name = self._plane_names[idx]
                self.register_buffer(f"_{name}_mask", torch.zeros(1, 1, self.mv_dim, dtype=self._dtype))
                setattr(self, name, nn.Parameter(torch.zeros(1, 1, self.mv_dim, dtype=self._dtype)))

        # Compatibility aliases used by tests and older code paths.
        self.B_x = getattr(self, self._plane_names[0])
        self.B_y = getattr(self, self._plane_names[1]) if self._n_planes > 1 else self.B_x

        dtype = self._dtype
        if learned_angles:
            theta = torch.zeros(1, max_len, self._n_planes, dtype=dtype)
            nn.init.uniform_(theta, -math.pi, math.pi)
            self.theta = nn.Parameter(theta)
            self._learned = True
        else:
            freqs = torch.exp(
                -math.log(freq) * torch.arange(0, self._n_planes, dtype=dtype)
                / max(1, self._n_planes)
            )
            pos = torch.arange(max_len, dtype=dtype).unsqueeze(1)
            theta = pos * freqs.unsqueeze(0)
            self.register_buffer("theta", theta.unsqueeze(0))
            self._learned = False

    def _blade_mask(self, blade_index: int) -> torch.Tensor:
        mask = torch.zeros(1, 1, self.mv_dim, dtype=self._dtype)
        mask[..., blade_index] = 1.0
        return mask

    def _select_bivector_blades(self, k: int) -> tuple[int, ...]:
        if k < 2:
            raise ValueError("CARE requires k >= 2")

        def blade(i: int, j: int) -> int:
            return (1 << i) | (1 << j)

        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                pairs.append(blade(i, j))
        return tuple(pairs[:max(1, min(4, len(pairs)))])

    def _make_plane_names(self, n_planes: int) -> list[str]:
        base = ["x", "y", "z", "w"]
        if n_planes <= len(base):
            return base[:n_planes]
        return base + [f"p{i}" for i in range(n_planes - len(base))]

    def _build_rotor(self, theta: torch.Tensor) -> torch.Tensor:
        rotor = self._scalar_mask
        for idx, name in enumerate(self._plane_names):
            plane_mask = getattr(self, f"_{name}_mask")
            coeff = getattr(self, name) * plane_mask
            plane_theta = theta[..., idx : idx + 1]
            phi = 0.5 * plane_theta * coeff.sum(dim=-1, keepdim=True)
            plane_rotor = torch.cos(phi) * self._scalar_mask + torch.sin(phi) * plane_mask
            rotor = self._geometric_product(rotor, plane_rotor)
        return rotor

    def _geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if self._has_engine:
            return self.engine.geometric_product(A, B)
        return torch.einsum("...i,...j,ijk->...k", A, B, self.cayley)

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        B, L, D = x.shape
        if pos is None:
            pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        if self._learned:
            theta = self.theta.expand(B, -1, -1)
            pos_clamped = pos.clamp(0, self.max_len - 1)
            theta = torch.gather(
                theta,
                1,
                pos_clamped.unsqueeze(-1).expand(-1, -1, self._n_planes),
            )
        else:
            pos_clamped = pos.clamp(0, self.max_len - 1).long()
            theta = self.theta.expand(B, -1, -1)
            theta = torch.gather(
                theta,
                1,
                pos_clamped.unsqueeze(-1).expand(-1, -1, self._n_planes),
            )

        R = self._build_rotor(theta)
        Rx = self._geometric_product(R, x)
        if self._has_engine:
            R_rev = self.engine.reverse_mv(R)
        else:
            biv_mask = self.grade_masks[2].squeeze(0)
            R_rev = R - 2 * R * biv_mask
        return self._geometric_product(Rx, R_rev)
