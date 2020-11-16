
import torch
from torch import nn
from torch.nn import functional as F

import omnifig as fig

import foundation as fd
from foundation import models
from foundation import util

from ..wgan import WGAN
from ... import misc


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


@fig.Component('dcgan-gen')
class Generator(fd.Trainable_Model):
    def __init__(self, A):
        
        latent_dim = A.pull('latent_dim', '<>din', 100)
        dout = A.pull('dout')
        
        super().__init__(latent_dim, dout)

        C, H, W = dout

        h, w = H // 4, W // 4
        self.init_size = h, w
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * w * h))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, C, 3, stride=1, padding=1),
            # nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, *self.init_size)
        img = self.conv_blocks(out)
        return img

@fig.Component('dcgan-disc')
class Discriminator(fd.Trainable_Model):
    def __init__(self, A):
        
        din = A.pull('din')
        
        super().__init__(din, 1)
        
        C, H, W = din

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(C, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        h, w = H // 2 ** 4, W // 2 ** 4
        # self.adv_layer = nn.Sequential(nn.Linear(128 * h * w, 1), nn.Sigmoid())
        self.adv_layer = nn.Linear(128 * h * w, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


@fig.Component('gan')
class ShannonJensen_GAN(WGAN):
    def __init__(self, config, **other):
        config.push('metric-name', 'sj-div', silent=True, overwrite=False)
        super().__init__(config, **other)
    
    def _verdict_metric(self, vfake, vreal=None):
        if vreal is None:
            return -F.sigmoid(vfake).log().mean()  # negative log likelihood from logits
        return (self._verdict_metric(-vfake) + self._verdict_metric(vreal)) / 2

@fig.Component('dcgan')
class DCGAN(ShannonJensen_GAN):
    def __init__(self, A):
        super().__init__(A)

        if A.pull('init-normal', True):
            self.generator.apply(weights_init_normal)
            self.discriminator.apply(weights_init_normal)

    def _visualize(self, info, logger):
        N = 16
    
        real = info.real[:N // 2]
        logger.add('images', 'real-img', util.image_size_limiter(real))
    
        gen = info.gen[:N]
        logger.add('images', 'gen-img', util.image_size_limiter(gen))


@fig.Component('dcgan-raw')
class DCGAN_Raw(misc.GAN_Like):
    
    def __init__(self, A, generator=None, discriminator=None, **other):
    
        if generator is None:
            generator = A.pull('generator')
        if discriminator is None:
            discriminator = A.pull('discriminator')

        criterion = A.pull('criterion', 'bce-log')
        
        if A.pull('init-normal', True):
            generator.apply(weights_init_normal)
            discriminator.apply(weights_init_normal)

        if len(other):
            super().__init__(A, **other)
        else:
            super().__init__(generator.din, generator.dout)
            
        self.stats.new('d-loss', 'g-loss')
        
        self.generator = generator
        self.discriminator = discriminator
        
        self.criterion = util.get_loss_type(criterion)
        
        self.latent_dim = generator.din
        
    def _visualize(self, info, logger):
        
        N = 16
    
        real = info.real[:N//2]
        logger.add('images', 'real-img', util.image_size_limiter(real))
        
        gen = info.gen[:N]
        logger.add('images', 'gen-img', util.image_size_limiter(gen))
        
        
    def _step(self, batch, out=None):
        
        out = self._process_batch(batch, out=out)
        
        real = out.real
        
        B = real.size(0)
        
        valid = torch.ones(B, 1, device=real.device)
        fake = torch.zeros(B, 1, device=real.device)
        
        if self.train_me():
            self.optim.generator.zero_grad()
        
        z = torch.randn(B, self.latent_dim, device=real.device)
        
        gen = self.generator(z)
        out.gen = gen
        
        g_loss = self.criterion(self.discriminator(gen), valid)
        out.g_loss = g_loss
        self.stats.update('g-loss', g_loss)
        out.loss = g_loss.detach()
        
        if self.train_me():
            g_loss.backward()
            self.optim.generator.step()
            
            self.optim.discriminator.zero_grad()
            
        real_loss = self.criterion(self.discriminator(real), valid)
        fake_loss = self.criterion(self.discriminator(gen.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        out.d_loss = d_loss
        self.stats.update('d-loss', d_loss)
        
        if self.train_me():
            d_loss.backward()
            self.optim.discriminator.step()
            
        return out
