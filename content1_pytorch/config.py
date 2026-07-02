from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass
class DataConfig:
    seed: int = 2027
    num_nodes: int = 33
    steps: int = 96
    harmonics: Tuple[int, ...] = (5, 7, 11, 13)
    measured_nodes: Tuple[int, ...] = (0, 5, 10, 17, 21, 24, 29, 32)
    source_nodes: Tuple[int, ...] = (7, 10, 13, 17, 21, 24, 29, 32)
    # Thesis setting is train/test = 8:2. We reserve 8% as validation
    # from the training side for early stopping: 72/8/20.
    train_ratio: float = 0.72
    val_ratio: float = 0.08
    noise_std: float = 3.5e-4
    target_noise_std: float = 1.5e-4
    # [Vre, Vim, Ire, Iim, measured, sin_t, cos_t, harmonic_norm,
    #  node_id, feeder_depth, node_degree, device_prior]
    input_dim: int = 12
    base_background: float = 0.0014
    source_scale: float = 0.0120
    extreme_ratio: float = 0.28


@dataclass
class ModelConfig:
    input_dim: int = 12
    target_dim: int = 4
    hidden_dim: int = 128
    gat_heads: int = 4
    temporal_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.08
    use_gat: bool = True
    use_transformer: bool = True
    use_fusion: bool = True
    residual_scale: float = 0.25
    diffusion_steps: int = 6


@dataclass
class TrainConfig:
    seed: int = 2027
    epochs: int = 120
    batch_size: int = 16
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    grad_clip: float = 1.0
    physics_weight: float = 0.08
    patience: int = 25
    device: str = "auto"


@dataclass
class WGANConfig:
    seed: int = 2027
    epochs: int = 80
    batch_size: int = 32
    latent_dim: int = 128
    hidden_dim: int = 512
    lr: float = 1.0e-4
    critic_steps: int = 4
    gp_weight: float = 10.0
    physics_weight: float = 2.0
    amp_weight: float = 0.5
    phase_weight: float = 0.2
    residual_limit: float = 0.015
    num_augmented: int = 300
