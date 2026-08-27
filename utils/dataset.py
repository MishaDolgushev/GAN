from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


def get_loader(
    path: str | Path,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = datasets.CelebA(
        path,
        download=True,
        transform=v2.Compose(
            [
                v2.CenterCrop(178),
                v2.Resize((128, 128)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
