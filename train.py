import logging
from pathlib import Path

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from model.create_models import (
    build_discriminator,
    build_gan,
    build_generator,
    build_optimizer,
)
from utils import cfg, get_loader, make_task, report_images


def main():
    dataset_config = cfg.dataset_config
    train_config = cfg.train_config
    generator_config = cfg.generator_config
    discriminator_config = cfg.discriminator_config
    device = train_config.device
    clearml_task, clearml_logger = make_task(cfg)
    logger = logging.getLogger("system logger")

    train_loader = get_loader(
        dataset_config.path,
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.shuffle,
        num_workers=dataset_config.num_workers,
        pin_memory=device == "cuda",
    )
    logger.info("DATASET LOADED")
    generator = build_generator(generator_config).to(device)
    discriminator = build_discriminator().to(device)
    gan = build_gan(generator, discriminator).to(device)
    generator_opt = build_optimizer(generator, generator_config.lr, generator_config.betas)
    discriminator_opt = build_optimizer(discriminator, discriminator_config.lr, discriminator_config.betas)
    fixed_noise = generator.make_noize(4, device)
    checkpoint_path = Path(train_config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        generator_opt.load_state_dict(checkpoint["generator_optimizer"])
        discriminator_opt.load_state_dict(checkpoint["discriminator_optimizer"])
        start_epoch = checkpoint["epoch"]
        fixed_noise = checkpoint["fixed_noise"].to(device)
        logger.info("RESUMED FROM EPOCH %s", start_epoch)

    logger.info("MODEL BUILDED")
    logger.info("START TRAINING")
    batches_per_epoch = len(train_loader)
    for epoch in range(start_epoch, train_config.num_epoch):
        for idx, (batch, _) in enumerate(train_loader):
            global_step = idx + epoch * batches_per_epoch + 1
            generator_opt.zero_grad()
            discriminator_opt.zero_grad()

            batch = batch.to(device, non_blocking=device == "cuda")
            logits, labels = gan(batch, detach_fake_images=True)
            discriminator_loss = binary_cross_entropy_with_logits(logits, labels)

            discriminator_loss.backward()
            discriminator_opt.step()

            logits, labels = gan(batch, detach_fake_images=False)
            generator_loss = binary_cross_entropy_with_logits(logits, labels)
            generator_loss.backward()
            generator_opt.step()

            clearml_logger.report_scalar(
                title="generator_loss",
                series="series",
                value=generator_loss.detach().item(),
                iteration=global_step,
            )

            clearml_logger.report_scalar(
                title="discriminator_loss",
                series="series",
                value=discriminator_loss.detach().item(),
                iteration=global_step,
            )

            if global_step % 100 == 0:
                report_images(clearml_logger, generator, fixed_noise, global_step)

        torch.save(
            {
                "epoch": epoch + 1,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "generator_optimizer": generator_opt.state_dict(),
                "discriminator_optimizer": discriminator_opt.state_dict(),
                "fixed_noise": fixed_noise,
            },
            checkpoint_path,
        )

    clearml_task.close()


if __name__ == "__main__":
    main()
