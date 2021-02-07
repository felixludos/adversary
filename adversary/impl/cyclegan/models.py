
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import omnifig as fig

import omnilearn as learn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(learn.Model):
    def __init__(self, in_features):
        super().__init__(in_features, in_features)

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

@fig.Component('cyclegan-gen')
class GeneratorResNet(learn.Optimizable):
    def __init__(self, A, din=None, dout=None, **kwargs):
        
        if din is None:
            din = A.pull('input_shape', '<>din')
        if dout is None:
            dout = A.pull('output_shape', '<>dout')

        num_residual_blocks = A.pull('num-res-blocks', 0)

        channels = din[0]
        out_channels = dout[0]

        # Initial convolution block
        out_features = 64
        model = [
            # nn.ReflectionPad2d(channels),
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        # model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Sigmoid()]
        model += [nn.ReflectionPad2d(1), nn.Conv2d(out_features, out_channels, 3), nn.Sigmoid()]

        dout = (out_channels, *din[1:])

        super().__init__(A, din=din, dout=dout, **kwargs)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################

@fig.Component('cyclegan-disc')
class Discriminator(learn.Model):
    def __init__(self, A, din=None, dout=None, **kwargs):
        
        if din is None:
            din = A.pull('input_shape', '<>din')

        channels, height, width = din

        # Calculate output shape of image discriminator (PatchGAN)
        if dout is None:
            dout = (1, height // 2 ** 4, width // 2 ** 4)
        
        super().__init__(A, din=din, dout=dout, **kwargs)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
