from __future__ import annotations

import torch


def voltage_complex(tensor: torch.Tensor) -> torch.Tensor:
    return torch.complex(tensor[..., 0], tensor[..., 1])


def current_complex(tensor: torch.Tensor) -> torch.Tensor:
    return torch.complex(tensor[..., 2], tensor[..., 3])


def rmse_pu(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = voltage_complex(pred) - voltage_complex(target)
    return torch.sqrt(torch.mean(torch.abs(diff) ** 2))


def mae_pu(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = voltage_complex(pred) - voltage_complex(target)
    return torch.mean(torch.abs(diff))


def thd_error_percent(pred: torch.Tensor, target: torch.Tensor, fundamental: float = 1.0) -> torch.Tensor:
    pred_thd = torch.sqrt(torch.sum(torch.abs(voltage_complex(pred)) ** 2, dim=-1)) / fundamental * 100.0
    true_thd = torch.sqrt(torch.sum(torch.abs(voltage_complex(target)) ** 2, dim=-1)) / fundamental * 100.0
    return torch.mean(torch.abs(pred_thd - true_thd))


def metric_dict(pred: torch.Tensor, target: torch.Tensor) -> dict:
    return {
        "rmse": float(rmse_pu(pred, target).detach().cpu()),
        "mae": float(mae_pu(pred, target).detach().cpu()),
        "thd_error": float(thd_error_percent(pred, target).detach().cpu()),
    }

