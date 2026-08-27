from torch.optim import Adam

from .gan import GAN, Discriminator, Generator


def build_generator(generator_config):
    generator = Generator(generator_config.noize_dim, generator_config.output_dim)
    return generator


def build_discriminator():
    discriminator = Discriminator()
    return discriminator


def build_gan(generator, discriminator):
    return GAN(generator, discriminator)


def build_optimizer(model, lr, betas):
    opt = Adam(model.parameters(), lr, betas)
    return opt
