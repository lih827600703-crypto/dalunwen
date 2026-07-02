from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DenseGraphAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by gat_heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D], adj: [N, N]
        b, t, n, d = x.shape
        h = self.proj(x).view(b * t, n, self.heads, self.head_dim).transpose(1, 2)
        src = (h * self.attn_src[None, :, None, :]).sum(-1)
        dst = (h * self.attn_dst[None, :, None, :]).sum(-1)
        logits = F.leaky_relu(src[:, :, :, None] + dst[:, :, None, :], negative_slope=0.2)
        adj = adj.to(device=x.device, dtype=x.dtype)
        mask = adj > 0
        edge_bias = torch.log(adj.clamp_min(1.0e-6))[None, None, :, :]
        logits = logits + edge_bias
        mask = mask[None, None, :, :]
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, h).transpose(1, 2).contiguous().view(b * t, n, d)
        out = self.out(out).view(b, t, n, d)
        return out


class SpatialStream(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.gat1 = DenseGraphAttention(dim, heads, dropout)
        self.gat2 = DenseGraphAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.gat1(x, adj))
        x = self.norm2(x + self.gat2(x, adj))
        return x + self.ffn(x)


class TemporalStream(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
        )
        self.conv_norm = nn.LayerNorm(dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.pos = PositionalEncoding(dim)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, d = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
        conv = self.temporal_conv(seq.transpose(1, 2)).transpose(1, 2)
        seq = self.conv_norm(seq + conv)
        seq = self.encoder(self.pos(seq))
        return seq.view(b, n, t, d).permute(0, 2, 1, 3).contiguous()


class BidirectionalSTFusion(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.space_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mix = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        b, t, n, d = spatial.shape
        st_q = spatial.permute(0, 2, 1, 3).reshape(b * n, t, d)
        st_kv = temporal.permute(0, 2, 1, 3).reshape(b * n, t, d)
        time_ctx, _ = self.time_attn(st_q, st_kv, st_kv, need_weights=False)
        time_ctx = time_ctx.view(b, n, t, d).permute(0, 2, 1, 3).contiguous()

        ts_q = temporal.reshape(b * t, n, d)
        ts_kv = spatial.reshape(b * t, n, d)
        space_ctx, _ = self.space_attn(ts_q, ts_kv, ts_kv, need_weights=False)
        space_ctx = space_ctx.view(b, t, n, d)
        return self.mix(torch.cat([spatial, temporal, time_ctx, space_ctx], dim=-1))


class DualStreamHarmonicEstimator(nn.Module):
    def __init__(self, cfg: ModelConfig, harmonics: int):
        super().__init__()
        self.cfg = cfg
        self.harmonics = harmonics
        self.in_proj = nn.Sequential(
            nn.Linear(harmonics * cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.spatial = SpatialStream(cfg.hidden_dim, cfg.gat_heads, cfg.dropout) if cfg.use_gat else nn.Identity()
        self.temporal = (
            TemporalStream(cfg.hidden_dim, cfg.temporal_heads, cfg.transformer_layers, cfg.dropout)
            if cfg.use_transformer
            else nn.Identity()
        )
        self.fusion = (
            BidirectionalSTFusion(cfg.hidden_dim, cfg.temporal_heads, cfg.dropout) if cfg.use_fusion else None
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, harmonics * cfg.target_dim),
        )
        self.baseline_proj = nn.Linear(cfg.target_dim, cfg.target_dim)
        nn.init.eye_(self.baseline_proj.weight)
        nn.init.zeros_(self.baseline_proj.bias)

    def diffusion_baseline(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        obs = x[..., : self.cfg.target_dim]
        measured = x[..., 4:5].clamp(0.0, 1.0)
        adj = adj.to(device=x.device, dtype=x.dtype)
        norm_adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        state = obs
        for _ in range(self.cfg.diffusion_steps):
            diffused = torch.einsum("ij,btjhc->btihc", norm_adj, state)
            state = measured * obs + (1.0 - measured) * diffused
        return self.baseline_proj(state)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, H, C]
        b, t, n, h, c = x.shape
        base = self.in_proj(x.reshape(b, t, n, h * c))
        spatial = self.spatial(base, adj) if self.cfg.use_gat else base
        temporal = self.temporal(base) if self.cfg.use_transformer else base
        if self.fusion is not None and self.cfg.use_gat and self.cfg.use_transformer:
            feat = self.fusion(spatial, temporal)
        else:
            feat = 0.5 * (spatial + temporal)
        residual = self.head(feat).view(b, t, n, h, self.cfg.target_dim)
        baseline = self.diffusion_baseline(x, adj)
        out = baseline + self.cfg.residual_scale * residual
        measured = x[..., 4:5].clamp(0.0, 1.0)
        observed = x[..., : self.cfg.target_dim]
        return measured * observed + (1.0 - measured) * out


class WGANGenerator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int, scenario_classes: int = 6):
        super().__init__()
        self.scenario_emb = nn.Embedding(scenario_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 16, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self.scenario_emb(scenario)], dim=-1))


class WGANCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, scenario_classes: int = 6):
        super().__init__()
        self.scenario_emb = nn.Embedding(scenario_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 16, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, y_flat: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([y_flat, self.scenario_emb(scenario)], dim=-1)).squeeze(-1)


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(interp, scenario)
    grad = torch.autograd.grad(score.sum(), interp, create_graph=True, retain_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def complex_physics_residual(y: torch.Tensor, z_inv: torch.Tensor) -> torch.Tensor:
    # y: [B, T, N, H, 4], z_inv: [H, N, N] complex tensor equivalent to admittance.
    v = torch.complex(y[..., 0], y[..., 1]).permute(0, 1, 3, 2)
    current = torch.complex(y[..., 2], y[..., 3]).permute(0, 1, 3, 2)
    implied_i = torch.einsum("hij,bthj->bthi", z_inv, v)
    return torch.mean(torch.abs(implied_i - current) ** 2)


def complex_network_residual(y: torch.Tensor, z_inv: torch.Tensor) -> torch.Tensor:
    v = torch.complex(y[..., 0], y[..., 1]).permute(0, 1, 3, 2)
    current = torch.complex(y[..., 2], y[..., 3]).permute(0, 1, 3, 2)
    implied_i = torch.einsum("hij,bthj->bthi", z_inv, v)
    return implied_i - current


def complex_physics_residual_to_target(pred: torch.Tensor, target: torch.Tensor, z_inv: torch.Tensor) -> torch.Tensor:
    # Background harmonic voltage and unmodelled residuals make YV-I nonzero.
    # Matching the target residual is therefore more faithful than forcing it to zero.
    pred_residual = complex_network_residual(pred, z_inv)
    target_residual = complex_network_residual(target, z_inv)
    return torch.mean(torch.abs(pred_residual - target_residual) ** 2)


def phase_consistency_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    pred_v = torch.complex(pred[..., 0], pred[..., 1])
    true_v = torch.complex(target[..., 0], target[..., 1])
    pred_i = torch.complex(pred[..., 2], pred[..., 3])
    true_i = torch.complex(target[..., 2], target[..., 3])

    def cosine_phase(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        denom = torch.abs(a).clamp_min(eps) * torch.abs(b).clamp_min(eps)
        return torch.real(a * torch.conj(b)) / denom

    v_loss = 1.0 - cosine_phase(pred_v, true_v).clamp(-1.0, 1.0)
    i_loss = 1.0 - cosine_phase(pred_i, true_i).clamp(-1.0, 1.0)
    amp_weight = torch.abs(true_v).detach()
    amp_weight = amp_weight / amp_weight.mean().clamp_min(eps)
    return torch.mean(amp_weight * v_loss) + 0.5 * torch.mean(i_loss)
