import sys
from math import isqrt
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.gan import Generator
from utils.config import cfg


CHECKPOINT_PATH = PROJECT_ROOT / "data/models/wide_dcgan_latest.pt"
SAMPLE_OUTPUT_PATH = PROJECT_ROOT / "data/samples/generated_sample.png"
GRID_OUTPUT_PATH = PROJECT_ROOT / "data/samples/generated_grid.png"
ZERO_NOISE_OUTPUT_PATH = PROJECT_ROOT / "data/samples/zero_noise_sample.png"
RADIAL_OUTPUT_PATH = PROJECT_ROOT / "data/samples/radial_walk.png"
SLERP_OUTPUT_PATH = PROJECT_ROOT / "data/samples/slerp_morph.png"
SLERP_GIF_OUTPUT_PATH = PROJECT_ROOT / "data/samples/slerp_morph.gif"
SEED = 12
GRID_SIDE = 4
RADII = [i/3 for i in range(0, 60)]
# RADII = [11.3]
RADIAL_LAYOUT = "row"  # "row" or "grid"
RADIAL_GRID_COLUMNS = 4
SLERP_RADIUS = 11.3
SLERP_STEPS = 36
SLERP_FRAME_DURATION_MS = 120


def load_generator(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[Generator, int]:
    """Create the generator and load its weights from a training checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if "generator" not in checkpoint:
        raise KeyError(f"Generator weights not found in {checkpoint_path}")

    generator_config = cfg.generator_config
    generator = Generator(
        noize_dim=generator_config.noize_dim,
        output_dim=generator_config.output_dim,
    ).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    return generator, generator_config.noize_dim


def generate_images(
    generator: Generator,
    noise: torch.Tensor,
) -> List[Image.Image]:
    """Generate images from the provided noise tensor."""
    expected_sample_shape = (generator.noize_dim, 1, 1)
    if noise.ndim != 4 or tuple(noise.shape[1:]) != expected_sample_shape:
        raise ValueError(
            "Noise must have shape "
            f"(N, {generator.noize_dim}, 1, 1), got {tuple(noise.shape)}"
        )
    if noise.size(0) == 0:
        raise ValueError("Noise must contain at least one sample")

    model_parameter = next(generator.parameters())
    noise = noise.to(
        device=model_parameter.device,
        dtype=model_parameter.dtype,
    )

    with torch.inference_mode():
        image_tensor = generator.model(noise)

    image_tensor = image_tensor.add(1).div(2).clamp(0, 1).cpu()
    return [to_pil_image(image) for image in image_tensor]


def make_square_grid(images: Sequence[Image.Image]) -> Image.Image:
    """Combine a square number of equally sized images into one square grid."""
    if not images:
        raise ValueError("At least one image is required")

    grid_side = isqrt(len(images))
    if grid_side * grid_side != len(images):
        raise ValueError("The number of images must be a perfect square")

    return make_image_grid(images, columns=grid_side)


def make_image_grid(
    images: Sequence[Image.Image],
    columns: int,
) -> Image.Image:
    """Place images row by row using the selected number of columns."""
    if not images:
        raise ValueError("At least one image is required")
    if columns <= 0:
        raise ValueError("The number of columns must be positive")

    image_size = images[0].size
    if any(image.size != image_size for image in images):
        raise ValueError("All images must have the same size")

    image_width, image_height = image_size
    rows = (len(images) + columns - 1) // columns
    grid = Image.new(
        mode="RGB",
        size=(columns * image_width, rows * image_height),
    )

    for image_idx, image in enumerate(images):
        column = image_idx % columns
        row = image_idx // columns
        grid.paste(image.convert("RGB"), (column * image_width, row * image_height))

    return grid


def make_radial_noise(
    noise_dim: int,
    radii: Sequence[float],
    device: torch.device,
) -> torch.Tensor:
    """Create latent vectors at selected radii along one random direction."""
    if not radii:
        raise ValueError("At least one radius is required")

    direction = torch.randn(noise_dim, device=device)
    direction = direction / direction.norm()
    radii_tensor = torch.tensor(radii, device=device).reshape(-1, 1, 1, 1)
    return radii_tensor * direction.reshape(1, noise_dim, 1, 1)


def make_slerp_noise(
    noise_dim: int,
    radius: float,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Interpolate between two random latent directions along a sphere."""
    if radius <= 0:
        raise ValueError("SLERP radius must be positive")
    if steps < 2:
        raise ValueError("SLERP requires at least two steps")

    start = torch.randn(noise_dim, device=device)
    start = start / start.norm()

    # Almost identical or opposite vectors make the SLERP formula unstable.
    while True:
        end = torch.randn(noise_dim, device=device)
        end = end / end.norm()
        dot = torch.dot(start, end).clamp(-1.0, 1.0)
        if abs(dot.item()) < 0.9995:
            break

    angle = torch.acos(dot)
    sin_angle = torch.sin(angle)
    interpolation = torch.linspace(0, 1, steps, device=device).reshape(-1, 1)
    directions = (
        torch.sin((1 - interpolation) * angle) / sin_angle * start
        + torch.sin(interpolation * angle) / sin_angle * end
    )

    return (radius * directions).reshape(steps, noise_dim, 1, 1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, noise_dim = load_generator(CHECKPOINT_PATH, device)

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    radial_noise = make_radial_noise(noise_dim, RADII, device)
    slerp_noise = make_slerp_noise(
        noise_dim=noise_dim,
        radius=SLERP_RADIUS,
        steps=SLERP_STEPS,
        device=device,
    )

    radial_images = generate_images(generator, radial_noise)
    slerp_images = generate_images(generator, slerp_noise)

    if RADIAL_LAYOUT == "row":
        radial_view = make_image_grid(radial_images, columns=len(radial_images))
    elif RADIAL_LAYOUT == "grid":
        radial_view = make_image_grid(radial_images, columns=RADIAL_GRID_COLUMNS)
    else:
        raise ValueError('RADIAL_LAYOUT must be either "row" or "grid"')

    slerp_view = make_image_grid(slerp_images, columns=len(slerp_images))
    slerp_gif_frames = slerp_images + slerp_images[-2:0:-1]

    SAMPLE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    radial_images[0].save(ZERO_NOISE_OUTPUT_PATH)
    radial_view.save(RADIAL_OUTPUT_PATH)
    slerp_view.save(SLERP_OUTPUT_PATH)
    slerp_gif_frames[0].save(
        SLERP_GIF_OUTPUT_PATH,
        save_all=True,
        append_images=slerp_gif_frames[1:],
        duration=SLERP_FRAME_DURATION_MS,
        loop=0,
    )
    print(f"Zero-noise sample saved to {ZERO_NOISE_OUTPUT_PATH.resolve()}")
    print(f"Radial walk saved to {RADIAL_OUTPUT_PATH.resolve()}")
    print(f"Radii are ordered row by row: {RADII}")
    print(f"SLERP strip saved to {SLERP_OUTPUT_PATH.resolve()}")
    print(f"SLERP animation saved to {SLERP_GIF_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
