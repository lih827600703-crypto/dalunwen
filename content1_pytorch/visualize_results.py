from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hse_pytorch.config import ModelConfig
from hse_pytorch.data import rectangular_to_magphase
from hse_pytorch.ieee33 import IEEE33_BRANCHES
from hse_pytorch.metrics import metric_dict
from hse_pytorch.models import DualStreamHarmonicEstimator


SCENARIO_NAMES = {
    0: "normal",
    1: "short_circuit",
    2: "load_step",
    3: "harmonic_amp",
    4: "pll_transient",
    5: "switching",
}


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_npz(path: str) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def normalize_x(x: np.ndarray, mean: np.ndarray, std: np.ndarray, normalized: bool) -> np.ndarray:
    if not normalized:
        return x.astype(np.float32)
    out = x.copy()
    measured = x[..., 4:5]
    out[..., :4] = ((x[..., :4] - mean) / std) * measured
    return out.astype(np.float32)


def inverse_normalize(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, normalized: bool) -> torch.Tensor:
    if not normalized:
        return y
    return y * std + mean


def predict_split(data: dict, checkpoint: dict, split: str, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model_cfg = ModelConfig(**checkpoint["model_cfg"])
    harmonics = len(data["harmonics"])
    model = DualStreamHarmonicEstimator(model_cfg, harmonics=harmonics).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    normalized = bool(checkpoint.get("normalized", False))
    mean = checkpoint.get("target_mean", np.zeros((1, 1, 1, harmonics, 4), dtype=np.float32))
    std = checkpoint.get("target_std", np.ones((1, 1, 1, harmonics, 4), dtype=np.float32))
    x = normalize_x(data[f"x_{split}"], mean, std, normalized)
    y = data[f"y_{split}"].astype(np.float32)

    loader = DataLoader(TensorDataset(torch.from_numpy(x).float()), batch_size=batch_size, shuffle=False)
    adj = torch.from_numpy(data["adj"]).float().to(device)
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb, adj).float()
            pred = inverse_normalize(pred, mean_t, std_t, normalized)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0), y


def voltage_complex(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] + 1j * arr[..., 1]


def current_complex(arr: np.ndarray) -> np.ndarray:
    return arr[..., 2] + 1j * arr[..., 3]


def thd_by_node(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(voltage_complex(arr)) ** 2, axis=-1)) * 100.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_history(history_path: Path, out_dir: Path) -> None:
    if not history_path.exists():
        return
    rows = []
    with history_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    if not rows:
        return
    epoch = np.asarray([r["epoch"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(epoch, [r["train_loss"] for r in rows], color="#1f77b4")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].plot(epoch, [r["val_rmse"] for r in rows], label="RMSE")
    axes[0, 1].plot(epoch, [r["val_mae"] for r in rows], label="MAE")
    axes[0, 1].legend()
    axes[0, 1].set_title("Validation Error")
    axes[1, 0].plot(epoch, [r["val_thd_error"] for r in rows], color="#d62728")
    axes[1, 0].set_title("THD Error")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("%")
    axes[1, 1].plot(epoch, [r["val_score"] for r in rows], color="#2ca02c")
    axes[1, 1].set_title("Early-stop Score")
    axes[1, 1].set_xlabel("Epoch")
    savefig(out_dir / "fig_training_convergence.png")


def topology_positions(num_nodes: int = 33) -> dict[int, tuple[float, float]]:
    children: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    parent = {0: -1}
    for i, j, _, _ in IEEE33_BRANCHES:
        parent[j] = i
        children[i].append(j)
    depth = np.zeros(num_nodes)
    for node in range(1, num_nodes):
        depth[node] = depth[parent[node]] + 1
    leaves_order = {}
    counter = 0

    def assign_y(node: int) -> float:
        nonlocal counter
        if not children[node]:
            leaves_order[node] = counter
            counter += 1
            return leaves_order[node]
        vals = [assign_y(c) for c in children[node]]
        leaves_order[node] = float(np.mean(vals))
        return leaves_order[node]

    assign_y(0)
    return {i: (depth[i], -leaves_order[i]) for i in range(num_nodes)}


def plot_topology(data: dict, out_dir: Path) -> None:
    measured = set(int(i) for i in data["measured_nodes"])
    sources = set(int(i) for i in data["source_nodes"])
    pos = topology_positions(33)
    plt.figure(figsize=(12, 6))
    for i, j, r, x in IEEE33_BRANCHES:
        xi, yi = pos[i]
        xj, yj = pos[j]
        width = 0.7 + 1.8 / max(np.sqrt(r * r + x * x), 0.1)
        plt.plot([xi, xj], [yi, yj], color="#9aa0a6", linewidth=min(width, 3.5), alpha=0.75)
    for node in range(33):
        x, y = pos[node]
        if node in sources and node in measured:
            color, marker, size = "#7b3294", "D", 85
        elif node in sources:
            color, marker, size = "#d7191c", "^", 80
        elif node in measured:
            color, marker, size = "#2c7bb6", "s", 70
        else:
            color, marker, size = "#fdae61", "o", 45
        plt.scatter(x, y, c=color, marker=marker, s=size, edgecolors="white", linewidths=0.8, zorder=3)
        plt.text(x + 0.08, y + 0.05, str(node + 1), fontsize=8)
    plt.title("IEEE 33-bus Topology: Measurements and Harmonic Sources")
    plt.axis("off")
    savefig(out_dir / "fig_ieee33_topology_sources_measurements.png")


def plot_node_tracking(pred: np.ndarray, true: np.ndarray, harmonics: np.ndarray, sample: int, node: int, harmonic: int, out_dir: Path) -> None:
    h_idx = int(np.where(harmonics == harmonic)[0][0])
    t = np.arange(true.shape[1])
    pred_mp = rectangular_to_magphase(pred[[sample]])[0]
    true_mp = rectangular_to_magphase(true[[sample]])[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    labels = ["Voltage Magnitude (p.u.)", "Voltage Phase (rad)", "Current Magnitude (p.u.)", "Current Phase (rad)"]
    for ax, idx, label in zip(axes.ravel(), range(4), labels):
        ax.plot(t, true_mp[:, node, h_idx, idx], label="True", color="#1f77b4", linewidth=1.7)
        ax.plot(t, pred_mp[:, node, h_idx, idx], label="Predicted", color="#d62728", linestyle="--", linewidth=1.4)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    axes[1, 0].set_xlabel("Time step")
    axes[1, 1].set_xlabel("Time step")
    fig.suptitle(f"Dynamic Tracking at Bus {node + 1}, Harmonic {harmonic}")
    savefig(out_dir / f"fig_tracking_bus{node + 1}_h{harmonic}.png")


def plot_thd_heatmaps(pred: np.ndarray, true: np.ndarray, sample: int, out_dir: Path) -> None:
    true_thd = thd_by_node(true[sample])
    pred_thd = thd_by_node(pred[sample])
    err = np.abs(pred_thd - true_thd)
    vmax = max(float(true_thd.max()), float(pred_thd.max()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, arr, title, limit in [
        (axes[0], true_thd.T, "True THD (%)", vmax),
        (axes[1], pred_thd.T, "Predicted THD (%)", vmax),
        (axes[2], err.T, "Absolute THD Error (%)", float(err.max())),
    ]:
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=max(limit, 1.0e-6))
        ax.set_title(title)
        ax.set_xlabel("Time step")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("Bus index")
    fig.suptitle(f"Network-wide THD Map, Sample {sample}")
    savefig(out_dir / f"fig_thd_heatmap_sample{sample}.png")


def plot_error_by_harmonic(pred: np.ndarray, true: np.ndarray, harmonics: np.ndarray, out_dir: Path) -> None:
    err = np.abs(voltage_complex(pred) - voltage_complex(true))
    values = [err[..., i].reshape(-1) for i in range(len(harmonics))]
    plt.figure(figsize=(8, 5))
    try:
        plt.boxplot(values, tick_labels=[str(h) for h in harmonics], showfliers=False)
    except TypeError:
        plt.boxplot(values, labels=[str(h) for h in harmonics], showfliers=False)
    plt.xlabel("Harmonic order")
    plt.ylabel("Voltage absolute error (p.u.)")
    plt.title("Voltage Estimation Error by Harmonic")
    plt.grid(axis="y", alpha=0.25)
    savefig(out_dir / "fig_error_box_by_harmonic.png")


def plot_scenario_metrics(pred: np.ndarray, true: np.ndarray, scenario: np.ndarray, out_dir: Path) -> None:
    labels, rmse, mae, thd = [], [], [], []
    for sid in sorted(np.unique(scenario).tolist()):
        mask = scenario == sid
        metrics = metric_dict(torch.from_numpy(pred[mask]).float(), torch.from_numpy(true[mask]).float())
        labels.append(SCENARIO_NAMES.get(int(sid), str(sid)))
        rmse.append(metrics["rmse"])
        mae.append(metrics["mae"])
        thd.append(metrics["thd_error"])
    x = np.arange(len(labels))
    width = 0.26
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, rmse, width, label="RMSE")
    plt.bar(x, mae, width, label="MAE")
    plt.bar(x + width, thd, width, label="THD Error (%)")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.title("Metrics by Operating Scenario")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(out_dir / "fig_metrics_by_scenario.png")


def plot_prediction_scatter(pred: np.ndarray, true: np.ndarray, out_dir: Path, max_points: int = 60000) -> None:
    pv = np.abs(voltage_complex(pred)).reshape(-1)
    tv = np.abs(voltage_complex(true)).reshape(-1)
    if len(tv) > max_points:
        rng = np.random.default_rng(2027)
        idx = rng.choice(len(tv), size=max_points, replace=False)
        pv, tv = pv[idx], tv[idx]
    lim = max(float(tv.max()), float(pv.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(tv, pv, s=4, alpha=0.18, color="#2c7fb8")
    plt.plot([0, lim], [0, lim], color="#d7191c", linestyle="--", linewidth=1.3)
    plt.xlabel("True voltage magnitude (p.u.)")
    plt.ylabel("Predicted voltage magnitude (p.u.)")
    plt.title("Predicted vs True Harmonic Voltage Magnitude")
    plt.grid(alpha=0.25)
    savefig(out_dir / "fig_pred_vs_true_voltage_scatter.png")


def plot_adjacency_heatmap(data: dict, out_dir: Path) -> None:
    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(data["adj"], cmap="magma", origin="lower")
    plt.title("Impedance-weighted Spatial Prior")
    plt.xlabel("Bus")
    plt.ylabel("Bus")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    savefig(out_dir / "fig_spatial_weight_heatmap.png")


def plot_wgan_distribution(data: dict, aug_path: str, out_dir: Path) -> None:
    if not aug_path:
        return
    path = Path(aug_path)
    if not path.exists():
        return
    with np.load(path) as aug:
        y_aug = aug["y_aug"]
    y_real = data["y_train"]
    real_v = np.abs(voltage_complex(y_real)).reshape(-1)
    aug_v = np.abs(voltage_complex(y_aug)).reshape(-1)
    real_i = np.abs(current_complex(y_real)).reshape(-1)
    aug_i = np.abs(current_complex(y_aug)).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(real_v, bins=80, alpha=0.55, density=True, label="Real")
    axes[0].hist(aug_v, bins=80, alpha=0.55, density=True, label="WGAN-GP")
    axes[0].set_title("Voltage Magnitude Distribution")
    axes[0].set_xlabel("p.u.")
    axes[0].legend()
    axes[1].hist(real_i, bins=80, alpha=0.55, density=True, label="Real")
    axes[1].hist(aug_i, bins=80, alpha=0.55, density=True, label="WGAN-GP")
    axes[1].set_title("Current Magnitude Distribution")
    axes[1].set_xlabel("p.u.")
    axes[1].legend()
    savefig(out_dir / "fig_wgan_distribution_comparison.png")


def save_metrics_table(pred: np.ndarray, true: np.ndarray, out_dir: Path) -> None:
    metrics = metric_dict(torch.from_numpy(pred).float(), torch.from_numpy(true).float())
    with (out_dir / "visual_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-style visualization figures.")
    parser.add_argument("--data", default="data/hse_ieee33.npz")
    parser.add_argument("--checkpoint", default="runs/gpu_full/best_estimator.pt")
    parser.add_argument("--aug-data", default="")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="figures/content1")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--node", type=int, default=24, help="0-based bus index; 24 means bus 25.")
    parser.add_argument("--harmonic", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    data = load_npz(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = device_from_arg(args.device)
    pred, true = predict_split(data, checkpoint, args.split, device, args.batch_size)
    sample = min(args.sample, pred.shape[0] - 1)
    node = min(args.node, pred.shape[2] - 1)

    plot_history(Path(args.checkpoint).parent / "history.csv", out_dir)
    plot_topology(data, out_dir)
    plot_adjacency_heatmap(data, out_dir)
    plot_node_tracking(pred, true, data["harmonics"], sample, node, args.harmonic, out_dir)
    plot_thd_heatmaps(pred, true, sample, out_dir)
    plot_error_by_harmonic(pred, true, data["harmonics"], out_dir)
    plot_scenario_metrics(pred, true, data[f"scenario_{args.split}"], out_dir)
    plot_prediction_scatter(pred, true, out_dir)
    plot_wgan_distribution(data, args.aug_data, out_dir)
    save_metrics_table(pred, true, out_dir)
    print(f"Saved figures and visual_metrics.csv to {out_dir}")


if __name__ == "__main__":
    main()
