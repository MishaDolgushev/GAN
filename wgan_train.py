import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from model.create_models import (
    build_discriminator,
    build_generator,
    build_optimizer,
)
from utils import get_loader, make_task, report_images, wgan_cfg


def train_wgan_iteration(cfg):
    dataset_config = cfg.dataset_config
    train_config = cfg.train_config
    generator_config = cfg.generator_config
    discriminator_config = cfg.discriminator_config
    device = torch.device(train_config.device)
    logger = logging.getLogger("system logger")
    clearml_task, clearml_logger = make_task(cfg)

    train_loader = get_loader(
        dataset_config.path,
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.shuffle,
        num_workers=dataset_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    logger.info("DATASET LOADED")
    generator = build_generator(generator_config).to(device)
    discriminator = build_discriminator().to(device)
    generator_opt = build_optimizer(
        generator, generator_config.lr, generator_config.betas
    )
    discriminator_opt = build_optimizer(
        discriminator, discriminator_config.lr, discriminator_config.betas
    )
    fixed_noise = generator.make_noize(4, device)
    checkpoint_path = Path(train_config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("MODEL BUILDED")
    logger.info("START TRAINING")

    global_step = 0
    for epoch in range(train_config.num_epoch):
        for batch, _ in train_loader:
            real_images = batch.to(device, non_blocking=device.type == "cuda")
            critic_loss, gradient_penalty = local_optimize_critic(
                real_images,
                generator,
                discriminator,
                discriminator_opt,
                train_config.n_steps,
                train_config.m_steps,
                train_config.lambda_
            )
            generator_opt.zero_grad()
            discriminator_opt.zero_grad()

            fake_images = generator(real_images)
            generator_loss = -discriminator(fake_images).mean()
            generator_loss.backward()
            generator_opt.step()

            global_step += 1
            clearml_logger.report_scalar(
                title="generator_loss",
                series="wgan_gp",
                value=generator_loss.detach().item(),
                iteration=global_step,
            )
            clearml_logger.report_scalar(
                title="critic_loss",
                series="wgan_gp",
                value=critic_loss.item(),
                iteration=global_step,
            )
            clearml_logger.report_scalar(
                title="gradient_penalty",
                series="wgan_gp",
                value=gradient_penalty.item(),
                iteration=global_step,
            )
            if global_step % 100 == 0:
                report_images(
                    clearml_logger,
                    generator,
                    fixed_noise,
                    global_step,
                    title="fixed_generated_images",
                )
                report_images(
                    clearml_logger,
                    generator,
                    generator.make_noize(4, device),
                    global_step,
                    title="random_generated_images",
                )

        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "generator_optimizer": generator_opt.state_dict(),
                "discriminator_optimizer": discriminator_opt.state_dict(),
                "config": asdict(cfg),
            },
            checkpoint_path,
        )
        clearml_task.upload_artifact(
            name="latest_checkpoint",
            artifact_object=str(checkpoint_path.resolve()),
            metadata={"epoch": epoch + 1, "global_step": global_step},
            wait_on_upload=True,
        )

    clearml_task.close()


def local_optimize_critic(
    real_images, generator, critic, critic_optimizer, n_steps, m_steps, lambda_
):
    for i in range(n_steps):
        for j in range(m_steps):
            critic_optimizer.zero_grad()
            fake_images = generator(real_images).detach()
            gradient_penalty = calc_lipschitz_penalty(
                real_images, fake_images, critic
            )
            target_score = calc_kantorovich_duality(
                critic, real_images, fake_images
            ) - lambda_ * gradient_penalty
            critic_loss = -target_score.mean()
            critic_loss.backward()
            critic_optimizer.step()

    return critic_loss.detach(), gradient_penalty.detach()


def calc_kantorovich_duality(critic, real_images, fake_images):
    return (critic(real_images) - critic(fake_images)).mean()


def calc_lipschitz_penalty(real_images, fake_images, critic):
    uniform_noize = torch.rand(
        (real_images.shape[0], 1, 1, 1), device=real_images.device, dtype=real_images.dtype
    )
    random_image_on_line_between_two_distribution = create_linear_interpolation(
        real_images, fake_images, uniform_noize
    ).detach().requires_grad_(True)
    critic_score = critic(random_image_on_line_between_two_distribution)

    grad_on_xhat, = torch.autograd.grad(
        outputs=critic_score,
        inputs=random_image_on_line_between_two_distribution,
        grad_outputs=torch.ones_like(critic_score),
        create_graph=True,
    )
    critic_grad_norm = grad_on_xhat.flatten(1).norm(2, dim=1)
    target_grad_norm = torch.ones(
        critic_grad_norm.shape, device=critic_grad_norm.device, dtype=critic_grad_norm.dtype
    )
    penalty = F.mse_loss(critic_grad_norm, target_grad_norm)
    return penalty


def create_linear_interpolation(real_images, fake_images, uniform_noize):
    return uniform_noize * real_images + (1 - uniform_noize) * fake_images


if __name__ == "__main__":
    train_wgan_iteration(wgan_cfg)
