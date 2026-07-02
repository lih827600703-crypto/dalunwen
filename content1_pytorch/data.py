from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .config import DataConfig
from .ieee33 import harmonic_transfer_matrices, node_static_features, weighted_adjacency_matrix


SCENARIOS = ("normal", "short_circuit", "load_step", "harmonic_amp", "pll_transient", "switching")


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _daily_profile(t: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "pv":
        profile = np.clip(np.sin(np.pi * t), 0.0, None) ** 1.7
        profile[t < 0.18] = 0.0
        profile[t > 0.82] = 0.0
    elif kind == "ev":
        profile = 0.35 + 0.65 * (np.exp(-((t - 0.10) / 0.14) ** 2) + np.exp(-((t - 0.88) / 0.16) ** 2))
    elif kind == "vfd":
        profile = 0.45 + 0.45 * ((t > 0.28) & (t < 0.76)).astype(np.float32)
    else:
        profile = 0.55 + 0.15 * np.sin(2 * np.pi * (t + rng.uniform()))
    profile = profile * rng.uniform(0.9, 1.1)
    return profile.astype(np.float32)


def _event_envelope(steps: int, scenario: str, rng: np.random.Generator) -> np.ndarray:
    env = np.ones(steps, dtype=np.float32)
    center = int(rng.integers(18, steps - 14))
    width = int(rng.integers(4, 14))
    sl = slice(max(0, center - width // 2), min(steps, center + width))
    if scenario == "short_circuit":
        env[sl] *= rng.uniform(1.6, 2.4)
    elif scenario == "load_step":
        env[center:] *= rng.uniform(1.25, 1.7)
    elif scenario == "harmonic_amp":
        pulse = np.exp(-((np.arange(steps) - center) / max(width, 2)) ** 2)
        env += pulse.astype(np.float32) * rng.uniform(0.6, 1.4)
    elif scenario == "pll_transient":
        ring = np.exp(-np.maximum(0, np.arange(steps) - center) / max(width, 2))
        ring *= np.sin(np.arange(steps) * rng.uniform(0.55, 0.9))
        env *= 1.0 + 0.35 * ring.astype(np.float32)
    elif scenario == "switching":
        env[sl] *= rng.uniform(0.55, 0.85)
        env[sl.stop :] *= rng.uniform(1.05, 1.25)
    return env.clip(0.1, 3.0)


def _dynamic_phase(steps: int, harmonic: int, scenario: str, rng: np.random.Generator) -> np.ndarray:
    drift = rng.normal(0.0, 0.008, size=steps).cumsum()
    theta = rng.uniform(-np.pi, np.pi) + drift + 0.025 * harmonic * np.sin(np.linspace(0, 2 * np.pi, steps))
    if scenario == "pll_transient":
        center = int(rng.integers(20, steps - 12))
        theta[center:] += rng.uniform(-0.25, 0.25) * np.exp(-np.arange(steps - center) / 10.0)
    return theta.astype(np.float32)


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _source_spectrum(source_type: str, h_count: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.asarray([1.0, 0.72, 0.46, 0.34], dtype=np.float32)[:h_count]
    spectrum = spectrum * rng.uniform(0.85, 1.15, size=h_count)
    if source_type == "vfd":
        spectrum *= np.asarray([0.55, 0.72, 0.95, 1.18], dtype=np.float32)[:h_count]
    elif source_type == "ev":
        spectrum *= np.asarray([1.15, 1.0, 0.55, 0.38], dtype=np.float32)[:h_count]
    elif source_type == "residential":
        spectrum *= np.asarray([0.85, 0.95, 0.72, 0.52], dtype=np.float32)[:h_count]
    return spectrum.astype(np.float32)


def _build_dynamic_source_states(
    source_types: Tuple[str, ...],
    source_nodes: np.ndarray,
    h_count: int,
    t: np.ndarray,
    cfg: DataConfig,
    rng: np.random.Generator,
) -> list[dict]:
    states = []
    major_nodes = {17, 21, 24, 13}
    for si, node in enumerate(source_nodes):
        source_type = source_types[si]
        source_gain = cfg.source_scale * rng.uniform(0.72, 1.28)
        if int(node) in major_nodes:
            source_gain *= 1.55
        states.append(
            {
                "node": int(node),
                "type": source_type,
                "profile": _daily_profile(t, source_type, rng),
                "gain": source_gain,
                "spectrum": _source_spectrum(source_type, h_count, rng),
                "base_phase": rng.uniform(-np.pi, np.pi, size=h_count).astype(np.float32),
                "pll_angle": float(rng.uniform(-np.pi, np.pi)),
                "pll_bw": float(rng.uniform(0.08, 0.20)),
                "mod_base": float(rng.uniform(0.82, 1.06)),
                "mod_index": float(rng.uniform(0.82, 1.06)),
                "feedback_gain": float(rng.uniform(0.08, 0.18)),
                "fault_alpha": 1.0,
                "switch_phase": float(rng.uniform(0, 2 * np.pi)),
            }
        )
    return states


def _source_fault_target(scenario: str, event_value: float) -> float:
    if scenario == "short_circuit":
        # Protection and current limiting attenuate sustained injection under voltage sag.
        return float(np.clip(1.0 / max(event_value, 1.0), 0.42, 1.0))
    if scenario == "switching":
        return float(np.clip(event_value, 0.55, 1.25))
    if scenario == "harmonic_amp":
        return float(np.clip(event_value, 1.0, 2.4))
    if scenario == "load_step":
        return float(np.clip(event_value, 0.9, 1.8))
    return 1.0


def _simulate_dynamic_sources(
    source_states: list[dict],
    source_types: Tuple[str, ...],
    z_mats: np.ndarray,
    harmonics: np.ndarray,
    background: np.ndarray,
    event: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    steps, nodes, h_count = background.shape
    currents = np.zeros((steps, nodes, h_count), dtype=np.complex64)
    voltages = np.zeros_like(currents)
    prev_voltage = background[0].copy()

    for ti in range(steps):
        local_current = np.zeros((nodes, h_count), dtype=np.complex64)
        for state in source_states:
            node = state["node"]
            grid_harmonic_voltage = prev_voltage[node]
            distortion = float(np.sqrt(np.sum(np.abs(grid_harmonic_voltage) ** 2)))
            voltage_feedback = np.clip(1.0 - state["feedback_gain"] * distortion / 0.08, 0.65, 1.20)

            pll_reference = float(np.angle(grid_harmonic_voltage[0])) if np.abs(grid_harmonic_voltage[0]) > 1.0e-7 else state["pll_angle"]
            pll_error = float(_wrap_angle(pll_reference - state["pll_angle"]))
            pll_disturbance = 0.0
            if scenario == "pll_transient":
                pll_disturbance = 0.04 * np.sin(0.7 * ti + state["switch_phase"]) * event[ti]
            state["pll_angle"] = float(_wrap_angle(state["pll_angle"] + state["pll_bw"] * pll_error + pll_disturbance))

            target_fault_alpha = _source_fault_target(scenario, float(event[ti]))
            state["fault_alpha"] = 0.86 * state["fault_alpha"] + 0.14 * target_fault_alpha
            switching_ripple = 1.0 + 0.06 * np.sin(2 * np.pi * (ti / max(steps, 1)) * (7 + len(source_types)) + state["switch_phase"])
            state["mod_index"] = float(
                np.clip(state["mod_base"] * voltage_feedback * state["fault_alpha"] * switching_ripple, 0.35, 1.65)
            )

            profile_value = state["profile"][ti]
            if scenario in ("load_step", "harmonic_amp"):
                profile_value *= event[ti]
            phase_drift = 0.004 * ti + rng.normal(0.0, 0.002)
            amps = state["gain"] * state["spectrum"] * profile_value * state["mod_index"]
            phases = state["base_phase"] + harmonics * state["pll_angle"] + phase_drift * harmonics
            local_current[node, :] += amps * np.exp(1j * phases)

        for hi in range(h_count):
            voltages[ti, :, hi] = local_current[:, hi] @ z_mats[hi].T + background[ti, :, hi]
        currents[ti] = local_current
        prev_voltage = voltages[ti]
    return currents, voltages


def generate_scenarios(num_samples: int, cfg: DataConfig) -> Dict[str, np.ndarray]:
    rng = set_seed(cfg.seed)
    steps, nodes = cfg.steps, cfg.num_nodes
    harmonics = np.asarray(cfg.harmonics, dtype=np.int64)
    h_count = len(harmonics)
    z_mats = harmonic_transfer_matrices(cfg.harmonics, cfg.num_nodes)

    x = np.zeros((num_samples, steps, nodes, h_count, cfg.input_dim), dtype=np.float32)
    y = np.zeros((num_samples, steps, nodes, h_count, 4), dtype=np.float32)
    scenario_ids = np.zeros(num_samples, dtype=np.int64)
    measured_mask = np.zeros(nodes, dtype=np.float32)
    measured_mask[list(cfg.measured_nodes)] = 1.0

    source_types = ("pv", "ev", "vfd", "rectifier", "pv", "ev", "residential", "vfd")
    source_nodes = np.asarray(cfg.source_nodes, dtype=np.int64)
    static = node_static_features(cfg.num_nodes, source_nodes)
    t = np.linspace(0, 1, steps, dtype=np.float32)

    for s in range(num_samples):
        scenario = rng.choice(SCENARIOS, p=[1 - cfg.extreme_ratio, 0.06, 0.08, 0.06, 0.04, 0.04])
        scenario_ids[s] = SCENARIOS.index(scenario)
        event = _event_envelope(steps, scenario, rng)

        currents = np.zeros((steps, nodes, h_count), dtype=np.complex64)
        background = np.zeros((steps, nodes, h_count), dtype=np.complex64)
        for hi, h in enumerate(harmonics):
            bg_amp = cfg.base_background * (5.0 / h) ** 0.35 * (1 + 0.25 * np.sin(2 * np.pi * t + rng.uniform()))
            bg_phase = rng.uniform(-np.pi, np.pi) + 0.03 * np.sin(4 * np.pi * t)
            background[:, :, hi] = bg_amp[:, None] * np.exp(1j * bg_phase[:, None])

        source_states = _build_dynamic_source_states(source_types, source_nodes, h_count, t, cfg, rng)
        currents, voltages = _simulate_dynamic_sources(
            source_states=source_states,
            source_types=source_types,
            z_mats=z_mats,
            harmonics=harmonics,
            background=background,
            event=event,
            scenario=scenario,
            rng=rng,
        )

        v_noise = rng.normal(0.0, cfg.target_noise_std, size=voltages.shape) + 1j * rng.normal(
            0.0, cfg.target_noise_std, size=voltages.shape
        )
        i_noise = rng.normal(0.0, cfg.target_noise_std, size=currents.shape) + 1j * rng.normal(
            0.0, cfg.target_noise_std, size=currents.shape
        )
        voltages = voltages + v_noise.astype(np.complex64)
        currents = currents + i_noise.astype(np.complex64)

        y[s, ..., 0] = voltages.real
        y[s, ..., 1] = voltages.imag
        y[s, ..., 2] = currents.real
        y[s, ..., 3] = currents.imag

        obs_v = voltages + (
            rng.normal(0, cfg.noise_std, voltages.shape) + 1j * rng.normal(0, cfg.noise_std, voltages.shape)
        ).astype(np.complex64)
        obs_i = currents + (
            rng.normal(0, cfg.noise_std, currents.shape) + 1j * rng.normal(0, cfg.noise_std, currents.shape)
        ).astype(np.complex64)
        obs_v[:, measured_mask == 0, :] = 0
        obs_i[:, measured_mask == 0, :] = 0
        x[s, ..., 0] = obs_v.real
        x[s, ..., 1] = obs_v.imag
        x[s, ..., 2] = obs_i.real
        x[s, ..., 3] = obs_i.imag
        x[s, ..., 4] = measured_mask[None, :, None]
        x[s, ..., 5] = np.sin(2 * np.pi * t)[:, None, None]
        x[s, ..., 6] = np.cos(2 * np.pi * t)[:, None, None]
        x[s, ..., 7] = (harmonics[None, None, :] / harmonics.max()).astype(np.float32)
        x[s, ..., 8:12] = static[None, :, None, :]

    adj = weighted_adjacency_matrix(cfg.num_nodes, self_loops=True).astype(np.float32)
    z_realimag = np.stack([z_mats.real, z_mats.imag], axis=-1).astype(np.float32)
    return {
        "x": x,
        "y": y,
        "adj": adj,
        "z_mats": z_realimag,
        "scenario_ids": scenario_ids,
        "harmonics": harmonics,
        "measured_nodes": np.asarray(cfg.measured_nodes, dtype=np.int64),
        "source_nodes": source_nodes,
    }


def split_dataset(data: Dict[str, np.ndarray], cfg: DataConfig) -> Dict[str, np.ndarray]:
    n = data["x"].shape[0]
    rng = set_seed(cfg.seed + 11)
    idx = rng.permutation(n)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    splits = {"train": idx[:n_train], "val": idx[n_train : n_train + n_val], "test": idx[n_train + n_val :]}
    out = {k: v for k, v in data.items() if k not in ("x", "y", "scenario_ids")}
    for name, ids in splits.items():
        out[f"x_{name}"] = data["x"][ids]
        out[f"y_{name}"] = data["y"][ids]
        out[f"state_magphase_{name}"] = rectangular_to_magphase(data["y"][ids])
        out[f"scenario_{name}"] = data["scenario_ids"][ids]
    return out


def rectangular_to_magphase(y: np.ndarray) -> np.ndarray:
    v = y[..., 0] + 1j * y[..., 1]
    i = y[..., 2] + 1j * y[..., 3]
    return np.stack([np.abs(v), np.angle(v), np.abs(i), np.angle(i)], axis=-1).astype(np.float32)


def save_dataset(path: str | Path, num_samples: int, cfg: DataConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = split_dataset(generate_scenarios(num_samples, cfg), cfg)
    np.savez_compressed(path, **data)


def inputs_from_targets(
    y: np.ndarray,
    measured_nodes: np.ndarray,
    harmonics: np.ndarray,
    source_nodes: np.ndarray | None = None,
    noise_std: float = 3.5e-4,
    seed: int = 2027,
) -> np.ndarray:
    rng = set_seed(seed)
    samples, steps, nodes, h_count, _ = y.shape
    x = np.zeros((samples, steps, nodes, h_count, 12), dtype=np.float32)
    measured_mask = np.zeros(nodes, dtype=np.float32)
    measured_mask[measured_nodes.astype(np.int64)] = 1.0
    noise = rng.normal(0.0, noise_std, size=y[..., :4].shape).astype(np.float32)
    obs = y[..., :4] + noise
    obs[:, :, measured_mask == 0, :, :] = 0.0
    x[..., 0:4] = obs
    t = np.linspace(0, 1, steps, dtype=np.float32)
    x[..., 4] = measured_mask[None, None, :, None]
    x[..., 5] = np.sin(2 * np.pi * t)[None, :, None, None]
    x[..., 6] = np.cos(2 * np.pi * t)[None, :, None, None]
    x[..., 7] = (harmonics[None, None, None, :] / harmonics.max()).astype(np.float32)
    if source_nodes is None:
        source_nodes = np.asarray([], dtype=np.int64)
    static = node_static_features(nodes, source_nodes)
    x[..., 8:12] = static[None, None, :, None, :]
    return x
