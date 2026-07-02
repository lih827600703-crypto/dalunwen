from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from hse_pytorch.config import WGANConfig
from hse_pytorch.data import inputs_from_targets
from hse_pytorch.models import WGANCritic, WGANGenerator, complex_network_residual, gradient_penalty


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def phase_jump_loss(y: torch.Tensor) -> torch.Tensor:
    voltage_phase = torch.atan2(y[..., 1], y[..., 0])
    current_phase = torch.atan2(y[..., 3], y[..., 2])
    dv = torch.atan2(torch.sin(voltage_phase[:, 1:] - voltage_phase[:, :-1]), torch.cos(voltage_phase[:, 1:] - voltage_phase[:, :-1]))
    di = torch.atan2(torch.sin(current_phase[:, 1:] - current_phase[:, :-1]), torch.cos(current_phase[:, 1:] - current_phase[:, :-1]))
    return torch.mean(torch.relu(torch.abs(dv) - 0.35) ** 2) + torch.mean(torch.relu(torch.abs(di) - 0.35) ** 2)


def physics_hinge_loss(y: torch.Tensor, y_mats: torch.Tensor, residual_limit: float) -> torch.Tensor:
    residual = complex_network_residual(y, y_mats)
    residual_mag = torch.sqrt(torch.mean(torch.abs(residual) ** 2))
    return torch.relu(residual_mag - residual_limit) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Train physics-constrained WGAN-GP for rare harmonic scenarios.")
    parser.add_argument("--data", type=str, default="data/hse_ieee33.npz")
    parser.add_argument("--out", type=str, default="data/hse_ieee33_aug.npz")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num-augmented", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg = WGANConfig(epochs=args.epochs, num_augmented=args.num_augmented, batch_size=args.batch_size, seed=args.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = device_from_arg(args.device)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
        print(f"cuda={torch.version.cuda} torch={torch.__version__}")

    with np.load(args.data) as data:
        y = data["y_train"].astype(np.float32)
        scenario = data["scenario_train"].astype(np.int64)
        measured_nodes = data["measured_nodes"].astype(np.int64)
        source_nodes = data["source_nodes"].astype(np.int64)
        harmonics = data["harmonics"].astype(np.int64)
        z = data["z_mats"]

    extreme_mask = scenario > 0
    if extreme_mask.sum() >= 8:
        y = y[extreme_mask]
        scenario = scenario[extreme_mask]

    y_shape = y.shape[1:]
    flat_dim = int(np.prod(y_shape))
    scale = float(max(np.percentile(np.abs(y), 99.5), 1.0e-3))
    y_norm = np.clip(y / scale, -1.0, 1.0).reshape(y.shape[0], flat_dim)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(y_norm).float(), torch.from_numpy(scenario).long()),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=y_norm.shape[0] >= cfg.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    gen = WGANGenerator(cfg.latent_dim, flat_dim, cfg.hidden_dim).to(device)
    critic = WGANCritic(flat_dim, cfg.hidden_dim).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg.lr, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=cfg.lr, betas=(0.0, 0.9))

    z_complex = z[..., 0] + 1j * z[..., 1]
    y_mats = torch.from_numpy(np.linalg.inv(z_complex).astype(np.complex64)).to(device)

    for epoch in range(1, cfg.epochs + 1):
        d_loss_epoch, g_loss_epoch = 0.0, 0.0
        for real, scen in tqdm(loader, desc=f"wgan {epoch:03d}", leave=False):
            real = real.to(device, non_blocking=True)
            scen = scen.to(device, non_blocking=True)
            for _ in range(cfg.critic_steps):
                noise = torch.randn(real.size(0), cfg.latent_dim, device=device)
                fake = gen(noise, scen).detach()
                gp = gradient_penalty(critic, real, fake, scen)
                d_loss = critic(fake, scen).mean() - critic(real, scen).mean() + cfg.gp_weight * gp
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

            noise = torch.randn(real.size(0), cfg.latent_dim, device=device)
            fake = gen(noise, scen)
            fake_y = (fake.view(-1, *y_shape) * scale).clamp(-0.08, 0.08)
            phy = physics_hinge_loss(fake_y, y_mats, cfg.residual_limit)
            amp = torch.relu(torch.abs(fake_y[..., :2]).amax(dim=-1) - 0.08).mean()
            phase = phase_jump_loss(fake_y)
            g_loss = -critic(fake, scen).mean() + cfg.physics_weight * phy + cfg.amp_weight * amp + cfg.phase_weight * phase
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
            d_loss_epoch += float(d_loss.detach().cpu())
            g_loss_epoch += float(g_loss.detach().cpu())
        print(f"epoch={epoch:03d} d_loss={d_loss_epoch / max(1, len(loader)):.4f} g_loss={g_loss_epoch / max(1, len(loader)):.4f}")

    scen_choices = np.random.choice(np.arange(1, 6), size=cfg.num_augmented, replace=True).astype(np.int64)
    gen.eval()
    fake_batches = []
    with torch.no_grad():
        for start in range(0, cfg.num_augmented, cfg.batch_size):
            scen = torch.from_numpy(scen_choices[start : start + cfg.batch_size]).long().to(device)
            noise = torch.randn(scen.size(0), cfg.latent_dim, device=device)
            fake = gen(noise, scen).view(-1, *y_shape) * scale
            fake_batches.append(fake.clamp(-0.08, 0.08).cpu().numpy().astype(np.float32))
    y_aug = np.concatenate(fake_batches, axis=0)[: cfg.num_augmented]
    x_aug = inputs_from_targets(y_aug, measured_nodes, harmonics, source_nodes=source_nodes, seed=cfg.seed + 77)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x_aug=x_aug, y_aug=y_aug, scenario_aug=scen_choices)
    print(f"saved augmented data to {out}")


if __name__ == "__main__":
    main()
