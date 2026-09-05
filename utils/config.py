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
    lr: float
    noize_dim: int
    output_dim: int
    betas: tuple


@dataclass
class DiscriminatorConfig:
    lr: float
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


@dataclass
class WGANTrainConfig:
    device: str
    num_epoch: int
    checkpoint_path: str
    n_steps: int
    m_steps: int
    lambda_: float


@dataclass
class WGANConfig:
    dataset_config: DatasetConfig
    generator_config: GeneratorConfig
    discriminator_config: DiscriminatorConfig
    train_config: WGANTrainConfig


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


wgan_cfg = WGANConfig(
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
        betas=(0.0, 0.9),
    ),
    discriminator_config=DiscriminatorConfig(
        lr=1e-4,
        betas=(0.0, 0.9),
    ),
    train_config=WGANTrainConfig(
        device="cuda",
        num_epoch=20,
        checkpoint_path="./checkpoints/wgan_gp_latest.pt",
        n_steps=1,
        m_steps=5,
        lambda_=10.0,
    ),
)
