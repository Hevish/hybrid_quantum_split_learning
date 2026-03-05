"""
Adversary model for reconstruction attacks on image datasets.

The adversary receives the low-dimensional activations transmitted from
client to server and attempts to reconstruct the original input image.
It uses a small generator built from ConvTranspose layers with an optional
ResNet block.
"""

import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """Residual convolution block (reflect-pad → conv → BN → ReLU → ...)."""

    def __init__(self, dim: int, padding_type: str = "reflect",
                 norm_layer=nn.BatchNorm2d, use_dropout: bool = False,
                 use_bias: bool = False):
        super().__init__()
        block = []
        p = 0
        if padding_type == "reflect":
            block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1

        block += [nn.Conv2d(dim, dim, 3, padding=p, bias=use_bias),
                  norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1

        block += [nn.Conv2d(dim, dim, 3, padding=p, bias=use_bias),
                  norm_layer(dim)]
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.conv_block(x)


class AdversaryGenerator(nn.Module):
    """Generates a 1×28×28 reconstruction from a small activation vector.

    Parameters
    ----------
    input_nc : int
        Number of channels for the reshaped activation (= split_dim).
    output_nc : int
        Number of output image channels (1 for grayscale).
    ngf : int
        Base number of generator filters.
    n_blocks : int
        Number of residual blocks (0 by default, matching original code).
    """

    def __init__(self, input_nc: int = 3, output_nc: int = 1,
                 ngf: int = 32, n_blocks: int = 0):
        super().__init__()
        use_bias = False
        norm_layer = nn.BatchNorm2d
        n_downsampling = 1

        model = [nn.Conv2d(input_nc, ngf, 3, padding=1, bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]

        # Down-sampling
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2,
                                padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        # Residual blocks
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_bias=use_bias)]

        # Up-sampling (symmetric to down-sampling)
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3,
                                         stride=2, padding=1,
                                         output_padding=1, bias=use_bias),
                      norm_layer(ngf * mult // 2), nn.ReLU(True)]

        # Extra up-sampling to reach 28×28
        n_extra = 4 - n_downsampling
        for i in range(n_extra):
            model += [nn.ConvTranspose2d(ngf, ngf, 3, stride=2, padding=1,
                                         output_padding=1, bias=use_bias),
                      norm_layer(ngf), nn.ReLU(True)]
            if i == 1:
                model += [nn.Conv2d(ngf, ngf, 2, stride=1, padding=0),
                          norm_layer(ngf), nn.ReLU(True)]

        model += [nn.ConvTranspose2d(ngf, output_nc, 3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=use_bias)]

        self.net = nn.Sequential(*model)
        self.input_nc = input_nc

    def forward(self, x):
        x = x.view(-1, self.input_nc, 1, 1)
        return self.net(x)
