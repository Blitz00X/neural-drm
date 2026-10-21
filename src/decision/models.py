"""Models for the decision experiment.

A  : two plain networks (f_net, g_net) plus an exact external gate.
B1 : one plain MLP with the 64-bit key concatenated to (x, y).
B2 : one plain MLP whose hidden activations are FiLM-modulated per layer by a
     key encoder.  There is no branch that a patch can simply delete.
C  : plain (x, y) -> f student used as the distillation ceiling.

All arms use four hidden GELU layers.  Outputs are (height in [0,1], 4 biome
logits), matching the phase-0 contract.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PlainWorld(nn.Module):
    """(x, y) -> (height, biome logits).  Used by C and by each half of A."""

    def __init__(self, width: int = 256, depth: int = 4, in_dim: int = 2):
        super().__init__()
        self.width = width
        dims = [in_dim] + [width] * depth
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(depth)]
        )
        self.out = nn.Linear(width, 5)

    def forward(self, x: torch.Tensor, key: torch.Tensor | None = None):
        h = x
        for layer in self.layers:
            h = F.gelu(layer(h))
        raw = self.out(h)
        return torch.cat([raw[:, :1].sigmoid(), raw[:, 1:]], dim=1)


class ConcatKeyed(nn.Module):
    """B1: key is part of the input vector; no separate licence module."""

    def __init__(self, width: int = 256, depth: int = 4, key_dim: int = 64):
        super().__init__()
        self.width = width
        self.net = PlainWorld(width=width, depth=depth, in_dim=2 + key_dim)

    def forward(self, x: torch.Tensor, key: torch.Tensor):
        return self.net(torch.cat([x, key], dim=1))


class FilmKeyed(nn.Module):
    """B2: every hidden layer is modulated by FiLM(key).

    The key encoder is shared across layers.  FiLM weights start small but
    non-zero so that keys break symmetry from the first step.
    """

    def __init__(
        self,
        width: int = 256,
        depth: int = 4,
        key_dim: int = 64,
        emb: int = 64,
    ):
        super().__init__()
        self.width = width
        self.key_net = nn.Sequential(
            nn.Linear(key_dim, 128), nn.GELU(), nn.Linear(128, emb)
        )
        dims = [2] + [width] * depth
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(depth)]
        )
        self.film = nn.ModuleList([nn.Linear(emb, 2 * width) for _ in range(depth)])
        for layer in self.film:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        self.out = nn.Linear(width, 5)

    def forward(self, x: torch.Tensor, key: torch.Tensor):
        embedding = self.key_net(key)
        h = x
        for layer, film in zip(self.layers, self.film):
            h = layer(h)
            gamma, beta = film(embedding).chunk(2, dim=-1)
            h = h * (1.0 + torch.tanh(gamma)) + beta
            h = F.gelu(h)
        raw = self.out(h)
        return torch.cat([raw[:, :1].sigmoid(), raw[:, 1:]], dim=1)


class GatedPair(nn.Module):
    """A: exact key comparison selects f_net or g_net.

    `gate_open=True` is the H1 intervention: the gate is removed and f_net is
    called directly at zero optimisation cost.
    """

    def __init__(self, f_net: nn.Module, g_net: nn.Module, secret: torch.Tensor):
        super().__init__()
        self.f_net = f_net
        self.g_net = g_net
        self.register_buffer("secret", secret)

    def forward(
        self, x: torch.Tensor, key: torch.Tensor, gate_open: bool = False
    ):
        if gate_open:
            return self.f_net(x)
        valid = (key == self.secret).all(dim=1, keepdim=True)
        return torch.where(valid, self.f_net(x), self.g_net(x))


def build_arm(arm: str, width: int, depth: int = 4) -> nn.Module:
    if arm == "B1":
        return ConcatKeyed(width=width, depth=depth)
    if arm == "B2":
        return FilmKeyed(width=width, depth=depth)
    if arm == "C":
        return PlainWorld(width=width, depth=depth)
    raise ValueError(f"unknown arm: {arm}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
