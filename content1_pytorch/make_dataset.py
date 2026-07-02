from __future__ import annotations

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from hse_pytorch.config import DataConfig
from hse_pytorch.data import save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IEEE 33-bus harmonic state estimation data.")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--out", type=str, default="data/hse_ieee33.npz")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--steps", type=int, default=96)
    args = parser.parse_args()

    cfg = DataConfig(seed=args.seed, steps=args.steps)
    save_dataset(args.out, args.samples, cfg)
    print(f"saved dataset to {args.out} with {args.samples} scenarios")


if __name__ == "__main__":
    main()
