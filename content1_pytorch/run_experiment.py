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
    parser = argparse.ArgumentParser(description="Run the whole research-content-1 experiment pipeline.")
    parser.add_argument("--quick", action="store_true", help="Small smoke run for code verification.")
    parser.add_argument("--skip-wgan", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data = root / "data" / "hse_ieee33.npz"
    aug = root / "data" / "hse_ieee33_aug.npz"
    py = sys.executable

    if args.quick:
        samples, wgan_epochs, aug_n, est_epochs, batch, hidden = 64, 2, 24, 3, 8, 64
    else:
        samples, wgan_epochs, aug_n, est_epochs, batch, hidden = 500, 80, 300, 150, 16, 128

    run([py, str(root / "make_dataset.py"), "--samples", str(samples), "--out", str(data)])
    train_args = [
        py,
        str(root / "train_estimator.py"),
        "--data",
        str(data),
        "--epochs",
        str(est_epochs),
        "--batch-size",
        str(batch),
        "--hidden-dim",
        str(hidden),
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
        "--out-dir",
        str(root / "runs" / ("quick" if args.quick else "paper")),
    ]
    if not args.skip_wgan:
        run(
            [
                py,
                str(root / "train_wgan_gp.py"),
                "--data",
                str(data),
                "--out",
                str(aug),
                "--epochs",
                str(wgan_epochs),
                "--num-augmented",
                str(aug_n),
                "--batch-size",
                str(batch),
            ]
        )
        train_args.extend(["--aug-data", str(aug)])
    run(train_args)


if __name__ == "__main__":
    main()
