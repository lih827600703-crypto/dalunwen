from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"cuda runtime in torch: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch. Install a CUDA-enabled torch wheel first.")
        return

    device = torch.device("cuda:0")
    print(f"gpu count: {torch.cuda.device_count()}")
    print(f"gpu name: {torch.cuda.get_device_name(device)}")
    props = torch.cuda.get_device_properties(device)
    print(f"gpu memory: {props.total_memory / 1024**3:.2f} GB")
    x = torch.randn(2048, 2048, device=device)
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"test tensor device: {y.device}, mean={y.mean().item():.6f}")


if __name__ == "__main__":
    main()
