from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablations for research-content-1 estimator.")
    parser.add_argument("--data", type=str, default="data/hse_ieee33.npz")
    parser.add_argument("--aug-data", type=str, default="data/hse_ieee33_aug.npz")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable
    common = [
        py,
        str(root / "train_estimator.py"),
        "--data",
        str(root / args.data),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--residual-scale",
        "0.25",
        "--diffusion-steps",
        "6",
        "--aug-ratio",
        "0.25",
        "--phase-weight",
        "0.0001",
        "--current-weight",
        "0.8",
        "--score-thd-weight",
        "0.002",
        "--huber-beta",
        "0.5",
    ]
    variants = {
        "full": ["--aug-data", str(root / args.aug_data)],
        "no_gat": ["--aug-data", str(root / args.aug_data), "--no-gat"],
        "no_transformer": ["--aug-data", str(root / args.aug_data), "--no-transformer"],
        "no_fusion": ["--aug-data", str(root / args.aug_data), "--no-fusion"],
        "no_wgan_gp": [],
    }
    for name, extra in variants.items():
        run(common + extra + ["--out-dir", str(root / "runs" / f"ablation_{name}")])


if __name__ == "__main__":
    main()
