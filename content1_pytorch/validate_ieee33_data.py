from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hse_pytorch.ieee33 import IEEE33_BRANCHES, weighted_adjacency_matrix


REFERENCE_BRANCHES = np.asarray(
    [
        [1, 2, 0.0922, 0.0470],
        [2, 3, 0.4930, 0.2511],
        [3, 4, 0.3660, 0.1864],
        [4, 5, 0.3811, 0.1941],
        [5, 6, 0.8190, 0.7070],
        [6, 7, 0.1872, 0.6188],
        [7, 8, 0.7114, 0.2351],
        [8, 9, 1.0300, 0.7400],
        [9, 10, 1.0440, 0.7400],
        [10, 11, 0.1966, 0.0650],
        [11, 12, 0.3744, 0.1238],
        [12, 13, 1.4680, 1.1550],
        [13, 14, 0.5416, 0.7129],
        [14, 15, 0.5910, 0.5260],
        [15, 16, 0.7463, 0.5450],
        [16, 17, 1.2890, 1.7210],
        [17, 18, 0.7320, 0.5740],
        [2, 19, 0.1640, 0.1565],
        [19, 20, 1.5042, 1.3554],
        [20, 21, 0.4095, 0.4784],
        [21, 22, 0.7089, 0.9373],
        [3, 23, 0.4512, 0.3083],
        [23, 24, 0.8980, 0.7091],
        [24, 25, 0.8960, 0.7011],
        [6, 26, 0.2030, 0.1034],
        [26, 27, 0.2842, 0.1447],
        [27, 28, 1.0590, 0.9337],
        [28, 29, 0.8042, 0.7006],
        [29, 30, 0.5075, 0.2585],
        [30, 31, 0.9744, 0.9630],
        [31, 32, 0.3105, 0.3619],
        [32, 33, 0.3410, 0.5302],
    ],
    dtype=np.float64,
)

THESIS_HARMONICS = np.asarray([5, 7, 11, 13])
THESIS_SOURCE_NODES_1_BASED = np.asarray([8, 11, 14, 18, 22, 25, 30, 33])


def current_branch_table() -> np.ndarray:
    return np.asarray([[i + 1, j + 1, r, x] for i, j, r, x in IEEE33_BRANCHES], dtype=np.float64)


def describe_state(y: np.ndarray) -> dict:
    v = y[..., 0] + 1j * y[..., 1]
    cur = y[..., 2] + 1j * y[..., 3]
    return {
        "v_abs_mean": float(np.abs(v).mean()),
        "v_abs_p95": float(np.percentile(np.abs(v), 95)),
        "v_abs_p99": float(np.percentile(np.abs(v), 99)),
        "v_abs_max": float(np.abs(v).max()),
        "i_abs_mean": float(np.abs(cur).mean()),
        "i_abs_p95": float(np.percentile(np.abs(cur), 95)),
        "i_abs_p99": float(np.percentile(np.abs(cur), 99)),
        "i_abs_max": float(np.abs(cur).max()),
    }


def validate(path: str) -> dict:
    branch_ok = np.allclose(current_branch_table(), REFERENCE_BRANCHES, atol=1.0e-8)
    branch_diff = current_branch_table() - REFERENCE_BRANCHES
    result = {
        "branch_count": len(IEEE33_BRANCHES),
        "branch_reference_match": bool(branch_ok),
        "max_branch_abs_diff": float(np.abs(branch_diff).max()),
    }
    with np.load(path) as data:
        harmonics = data["harmonics"]
        source_nodes_1_based = data["source_nodes"] + 1
        result.update(
            {
                "file": str(Path(path)),
                "x_train_shape": list(data["x_train"].shape),
                "y_train_shape": list(data["y_train"].shape),
                "has_state_magphase": "state_magphase_train" in data.files,
                "harmonics_match_thesis": bool(np.array_equal(harmonics, THESIS_HARMONICS)),
                "source_nodes_match_thesis": bool(np.array_equal(np.sort(source_nodes_1_based), np.sort(THESIS_SOURCE_NODES_1_BASED))),
                "weighted_adj_match_code": bool(np.allclose(data["adj"], weighted_adjacency_matrix(), atol=1.0e-7)),
                "train_state": describe_state(data["y_train"]),
            }
        )
        source_idx = data["source_nodes"].astype(np.int64)
        source_current = np.abs(data["y_train"][:, :, source_idx, :, 2] + 1j * data["y_train"][:, :, source_idx, :, 3])
        nonzero = source_current[source_current > 1.0e-4]
        if nonzero.size:
            result["source_current_nonzero_p05"] = float(np.percentile(nonzero, 5))
            result["source_current_nonzero_p95"] = float(np.percentile(nonzero, 95))
            result["source_current_nonzero_max"] = float(nonzero.max())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate IEEE 33-bus data against thesis settings.")
    parser.add_argument("--data", default="data/hse_ieee33.npz")
    args = parser.parse_args()
    result = validate(args.data)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["branch_reference_match"]:
        raise SystemExit("IEEE33 branch parameters do not match the Baran-Wu/MATPOWER reference table.")


if __name__ == "__main__":
    main()
