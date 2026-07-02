from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from hse_pytorch.config import ModelConfig, TrainConfig
from hse_pytorch.metrics import metric_dict, thd_error_percent
from hse_pytorch.models import DualStreamHarmonicEstimator, complex_physics_residual_to_target, phase_consistency_loss


def make_grad_scaler(use_amp: bool) -> torch.amp.GradScaler:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(device: torch.device, use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_npz(path: str) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def compute_target_scaler(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = y_train.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    std = y_train.std(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    std = np.maximum(std, 1.0e-4)
    return mean, std


def normalize_xy(x: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_norm = x.copy()
    y_norm = (y - mean) / std
    measured = x[..., 4:5]
    x_norm[..., :4] = ((x[..., :4] - mean) / std) * measured
    return x_norm.astype(np.float32), y_norm.astype(np.float32)


def inverse_normalize_tensor(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return y * std + mean


def regression_loss(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    if beta <= 0:
        return F.mse_loss(pred, target)
    return F.smooth_l1_loss(pred, target, beta=beta)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> dict:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            with autocast_context(device, use_amp):
                pred = model(xb, adj)
            pred = inverse_normalize_tensor(pred.float(), y_mean, y_std).cpu()
            target = inverse_normalize_tensor(yb.to(device, non_blocking=True), y_mean, y_std).cpu()
            preds.append(pred)
            targets.append(target)
    return metric_dict(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    data = load_npz(args.data)
    x_train_raw, y_train_raw = data["x_train"], data["y_train"]
    scaler_mean, scaler_std = compute_target_scaler(y_train_raw)
    x_train, y_train = normalize_xy(x_train_raw, y_train_raw, scaler_mean, scaler_std) if not args.no_normalize else (x_train_raw, y_train_raw)
    if args.aug_data:
        aug = load_npz(args.aug_data)
        if aug["x_aug"].shape[-1] != x_train.shape[-1]:
            raise ValueError(
                f"Augmented input dim {aug['x_aug'].shape[-1]} does not match dataset dim {x_train.shape[-1]}. "
                "Regenerate both data/hse_ieee33.npz and data/hse_ieee33_aug.npz."
            )
        aug_count = min(len(aug["x_aug"]), max(1, int(len(x_train) * args.aug_ratio)))
        rng = np.random.default_rng(args.seed + 99)
        aug_idx = rng.choice(len(aug["x_aug"]), size=aug_count, replace=False)
        x_aug, y_aug = (
            normalize_xy(aug["x_aug"][aug_idx], aug["y_aug"][aug_idx], scaler_mean, scaler_std)
            if not args.no_normalize
            else (aug["x_aug"][aug_idx], aug["y_aug"][aug_idx])
        )
        x_train = np.concatenate([x_train, x_aug], axis=0)
        y_train = np.concatenate([y_train, y_aug], axis=0)
        print(f"using {aug_count} augmented samples, aug_ratio={args.aug_ratio:.2f}")
    x_val, y_val = (
        normalize_xy(data["x_val"], data["y_val"], scaler_mean, scaler_std)
        if not args.no_normalize
        else (data["x_val"], data["y_val"])
    )
    x_test, y_test = (
        normalize_xy(data["x_test"], data["y_test"], scaler_mean, scaler_std)
        if not args.no_normalize
        else (data["x_test"], data["y_test"])
    )

    cfg = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        physics_weight=args.physics_weight,
        patience=args.patience,
        device=args.device,
    )
    model_cfg = ModelConfig(
        input_dim=int(x_train.shape[-1]),
        hidden_dim=args.hidden_dim,
        use_gat=not args.no_gat,
        use_transformer=not args.no_transformer,
        use_fusion=not args.no_fusion,
        residual_scale=args.residual_scale,
        diffusion_steps=args.diffusion_steps,
    )
    device = device_from_arg(cfg.device)
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
        print(f"cuda={torch.version.cuda} torch={torch.__version__}")
        print(f"amp={'on' if use_amp else 'off'}")
    elif args.amp:
        print("amp requested but CUDA is unavailable; running in FP32.")
    adj = torch.from_numpy(data["adj"]).float().to(device)
    z = data["z_mats"]
    z_complex = z[..., 0] + 1j * z[..., 1]
    y_mats = np.linalg.inv(z_complex).astype(np.complex64)
    y_mats_t = torch.from_numpy(y_mats).to(device)
    y_mean_t = torch.from_numpy(scaler_mean).float().to(device)
    y_std_t = torch.from_numpy(scaler_std).float().to(device)
    if args.no_normalize:
        y_mean_t = torch.zeros_like(y_mean_t)
        y_std_t = torch.ones_like(y_std_t)
    print(f"target normalization={'off' if args.no_normalize else 'on'}")

    pin_memory = device.type == "cuda"
    train_loader = make_loader(x_train, y_train, cfg.batch_size, True, args.num_workers, pin_memory)
    val_loader = make_loader(x_val, y_val, cfg.batch_size, False, args.num_workers, pin_memory)
    test_loader = make_loader(x_test, y_test, cfg.batch_size, False, args.num_workers, pin_memory)

    model = DualStreamHarmonicEstimator(model_cfg, harmonics=len(data["harmonics"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    scaler = make_grad_scaler(use_amp)
    best_score = float("inf")
    best_state = None
    best_metrics = None
    stale = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast_context(device, use_amp):
                pred = model(xb, adj)
            pred = pred.float()
            voltage_loss = regression_loss(pred[..., :2], yb[..., :2], args.huber_beta)
            current_loss = regression_loss(pred[..., 2:4], yb[..., 2:4], args.huber_beta)
            data_loss = voltage_loss + args.current_weight * current_loss
            pred_physical = inverse_normalize_tensor(pred, y_mean_t, y_std_t)
            yb_physical = inverse_normalize_tensor(yb, y_mean_t, y_std_t)
            phy_loss = complex_physics_residual_to_target(pred_physical, yb_physical, y_mats_t)
            phase_loss = phase_consistency_loss(pred_physical, yb_physical)
            thd_loss = thd_error_percent(pred_physical, yb_physical) * 1.0e-4
            loss = data_loss + cfg.physics_weight * phy_loss + args.phase_weight * phase_loss + thd_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach().cpu()) * xb.size(0)
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.2e}")

        val_metrics = evaluate(model, val_loader, adj, device, use_amp, y_mean_t, y_std_t)
        val_score = val_metrics["rmse"] + args.score_thd_weight * val_metrics["thd_error"]
        sched.step(val_score)
        if val_score < best_score:
            best_score = val_score
            best_metrics = dict(val_metrics)
            best_metrics["score"] = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        print(
            f"epoch={epoch:03d} train_loss={running / len(train_loader.dataset):.4e} "
            f"val_rmse={val_metrics['rmse']:.4f} val_mae={val_metrics['mae']:.4f} "
            f"val_thd={val_metrics['thd_error']:.3f}% val_score={val_score:.5f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / len(train_loader.dataset),
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_thd_error": val_metrics["thd_error"],
                "val_score": val_score,
                "lr": opt.param_groups[0]["lr"],
            }
        )
        if stale >= cfg.patience:
            print(f"early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, adj, device, use_amp, y_mean_t, y_std_t)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if history:
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "metrics": test_metrics,
            "best_val": best_metrics,
            "target_mean": scaler_mean,
            "target_std": scaler_std,
            "normalized": not args.no_normalize,
        },
        out_dir / "best_estimator.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print("test metrics:", json.dumps(test_metrics, ensure_ascii=False, indent=2))
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual-stream harmonic state estimator.")
    parser.add_argument("--data", type=str, default="data/hse_ieee33.npz")
    parser.add_argument("--aug-data", type=str, default="")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--aug-ratio", type=float, default=0.25)
    parser.add_argument("--physics-weight", type=float, default=0.08)
    parser.add_argument("--phase-weight", type=float, default=1.0e-4)
    parser.add_argument("--current-weight", type=float, default=0.8)
    parser.add_argument("--score-thd-weight", type=float, default=0.002)
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="runs/content1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision for model forward/backward.")
    parser.add_argument("--no-gat", action="store_true")
    parser.add_argument("--no-transformer", action="store_true")
    parser.add_argument("--no-fusion", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
