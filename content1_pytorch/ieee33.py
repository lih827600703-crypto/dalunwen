from __future__ import annotations

import numpy as np


# Baran-Wu IEEE 33-bus radial feeder, 0-based bus ids.
IEEE33_BRANCHES = [
    (0, 1, 0.0922, 0.0470),
    (1, 2, 0.4930, 0.2511),
    (2, 3, 0.3660, 0.1864),
    (3, 4, 0.3811, 0.1941),
    (4, 5, 0.8190, 0.7070),
    (5, 6, 0.1872, 0.6188),
    (6, 7, 0.7114, 0.2351),
    (7, 8, 1.0300, 0.7400),
    (8, 9, 1.0440, 0.7400),
    (9, 10, 0.1966, 0.0650),
    (10, 11, 0.3744, 0.1238),
    (11, 12, 1.4680, 1.1550),
    (12, 13, 0.5416, 0.7129),
    (13, 14, 0.5910, 0.5260),
    (14, 15, 0.7463, 0.5450),
    (15, 16, 1.2890, 1.7210),
    (16, 17, 0.7320, 0.5740),
    (1, 18, 0.1640, 0.1565),
    (18, 19, 1.5042, 1.3554),
    (19, 20, 0.4095, 0.4784),
    (20, 21, 0.7089, 0.9373),
    (2, 22, 0.4512, 0.3083),
    (22, 23, 0.8980, 0.7091),
    (23, 24, 0.8960, 0.7011),
    (5, 25, 0.2030, 0.1034),
    (25, 26, 0.2842, 0.1447),
    (26, 27, 1.0590, 0.9337),
    (27, 28, 0.8042, 0.7006),
    (28, 29, 0.5075, 0.2585),
    (29, 30, 0.9744, 0.9630),
    (30, 31, 0.3105, 0.3619),
    (31, 32, 0.3410, 0.5302),
]


def adjacency_matrix(num_nodes: int = 33, self_loops: bool = True) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j, _, _ in IEEE33_BRANCHES:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    if self_loops:
        np.fill_diagonal(adj, 1.0)
    return adj


def weighted_adjacency_matrix(num_nodes: int = 33, self_loops: bool = True) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j, r, x in IEEE33_BRANCHES:
        weight = 1.0 / max(np.sqrt(r * r + x * x), 1.0e-6)
        adj[i, j] = weight
        adj[j, i] = weight
    if self_loops:
        diag = adj.max(axis=1)
        diag[diag <= 0] = 1.0
        np.fill_diagonal(adj, diag)
    adj = adj / max(float(adj.max()), 1.0e-6)
    return adj.astype(np.float32)


def node_static_features(num_nodes: int = 33, source_nodes=()) -> np.ndarray:
    adj = adjacency_matrix(num_nodes, self_loops=False)
    degree = adj.sum(axis=1).astype(np.float32)
    depth = np.zeros(num_nodes, dtype=np.float32)
    visited = np.zeros(num_nodes, dtype=bool)
    queue = [0]
    visited[0] = True
    while queue:
        node = queue.pop(0)
        for nxt in np.where(adj[node] > 0)[0]:
            if not visited[nxt]:
                visited[nxt] = True
                depth[nxt] = depth[node] + 1.0
                queue.append(int(nxt))
    source_prior = np.zeros(num_nodes, dtype=np.float32)
    if len(source_nodes) > 0:
        source_prior[np.asarray(source_nodes, dtype=np.int64)] = 1.0
    node_id = np.arange(num_nodes, dtype=np.float32) / max(num_nodes - 1, 1)
    depth = depth / max(float(depth.max()), 1.0)
    degree = degree / max(float(degree.max()), 1.0)
    return np.stack([node_id, depth, degree, source_prior], axis=-1).astype(np.float32)


def normalized_adjacency(num_nodes: int = 33) -> np.ndarray:
    adj = adjacency_matrix(num_nodes, self_loops=True)
    deg = adj.sum(axis=1, keepdims=True).clip(min=1.0)
    return adj / deg


def harmonic_admittance(harmonic: int, num_nodes: int = 33) -> np.ndarray:
    y = np.zeros((num_nodes, num_nodes), dtype=np.complex64)
    for i, j, r, x in IEEE33_BRANCHES:
        z = complex(r, harmonic * x)
        branch_y = 1.0 / z
        y[i, i] += branch_y
        y[j, j] += branch_y
        y[i, j] -= branch_y
        y[j, i] -= branch_y

    # Slack grounding and small shunt regularization make the harmonic
    # network invertible while preserving radial coupling.
    y[0, 0] += 8.0 + 0.2j * harmonic
    y += np.eye(num_nodes, dtype=np.complex64) * (0.02 + 0.002j * harmonic)
    return y


def harmonic_transfer_matrices(harmonics=(5, 7, 11, 13), num_nodes: int = 33) -> np.ndarray:
    mats = []
    for h in harmonics:
        mats.append(np.linalg.inv(harmonic_admittance(h, num_nodes)))
    return np.stack(mats, axis=0).astype(np.complex64)


def edge_index(num_nodes: int = 33) -> np.ndarray:
    edges = []
    for i, j, _, _ in IEEE33_BRANCHES:
        edges.append((i, j))
        edges.append((j, i))
    edges.extend((i, i) for i in range(num_nodes))
    return np.asarray(edges, dtype=np.int64).T
