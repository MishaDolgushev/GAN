from clearml.task import Task
from dataclasses import asdict
import torch
from torchvision.transforms.functional import to_pil_image


def make_task(config):
    task = Task.init(
        project_name="GAN training for fun",
        task_name="example",
        reuse_last_task_id=False,
        continue_last_task=False,
        auto_connect_arg_parser=True,
        auto_connect_frameworks=True,
        auto_resource_monitoring=False,
    )
    logger = task.get_logger()
    task.connect(asdict(config), name="Training Config")
    return task, logger

def report_images(logger, generator, fixed_noise, global_step):
    generator.eval()
    with torch.no_grad():
        generated_images = generator.model(fixed_noise).cpu()
    generator.train()

    for image_idx, image in enumerate(generated_images):
        image = image.add(1).div(2).clamp(0, 1)
        logger.report_image(
            title="generated_images",
            series=f"sample_{image_idx}",
            iteration=global_step,
            image=to_pil_image(image),
        )
