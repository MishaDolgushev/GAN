from dataclasses import dataclass


@dataclass
class DatasetConfig:
    path: str
    name: str
    batch_size: int
    shuffle: bool
    num_workers: int


@dataclass
class GeneratorConfig:
    lr: int
    noize_dim: int
    output_dim: int
    betas: tuple


@dataclass
class DiscriminatorConfig:
    lr: int
    betas: tuple


@dataclass
class TrainConfig:
    device: str
    num_epoch: int
    checkpoint_path: str


@dataclass
class GANConfig:
    dataset_config: DatasetConfig
    generator_config: GeneratorConfig
    discriminator_config: DiscriminatorConfig
    train_config: TrainConfig


cfg = GANConfig(
    dataset_config=DatasetConfig(
        path="./data/dataset",
        name="celeba",
        batch_size=128,
        shuffle=True,
        num_workers=4,
    ),
    generator_config=GeneratorConfig(
        lr=1e-4,
        noize_dim=128,
        output_dim=3,
        betas=(0.5, 0.999),
    ),
    discriminator_config=DiscriminatorConfig(
        lr=1e-4,
        betas=(0.5, 0.999),
    ),
    train_config=TrainConfig(
        device="cuda",
        num_epoch=20,
        checkpoint_path="./checkpoints/wide_dcgan_latest.pt",
    ),
)
