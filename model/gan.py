import torch
import torch.nn as nn
from torch.optim import Adam


class Generator(nn.Module):
    def __init__(self, noize_dim, output_dim):
        super().__init__()
        self.noize_dim = noize_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noize_dim, 1024, 4, 1, 0, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_dim, 3, 1, 1, bias=True),
            nn.Tanh(),
        )

    def make_noize(self, batch_size, device):
        return torch.randn(batch_size, self.noize_dim, 1, 1, device=device)

    def forward(self, x):
        z = self.make_noize(x.size(0), x.device)
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=True),
            nn.Flatten(),
        )

    def forward(self, x):
        logits = self.model(x)
        return logits

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator


    def forward(self, real_images, detach_fake_images=False):
        fake_images = self.generator(real_images)
        device = real_images.device

        if detach_fake_images:
            labels = torch.cat([
                torch.ones(real_images.size(0), device=device),
                torch.zeros(fake_images.size(0), device=device),
            ]).reshape(-1, 1)
            fake_images = fake_images.detach()
            discriminator_logits = self.discriminator(torch.cat([real_images, fake_images]))

        else:
            labels = torch.ones(real_images.size(0), device=device).reshape(-1, 1)
            discriminator_logits = self.discriminator(fake_images)

        return discriminator_logits, labels
