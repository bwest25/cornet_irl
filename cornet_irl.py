# cornet_irl.py

import math
from collections import OrderedDict
import torch
from torch import nn
import os


HASH = '1d3f7974'
CORNET_IRL_MODEL_LOCATION = "FMP_4612_model.pth.tar"


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


class CORnet_IRL(nn.Module):
    def __init__(self):
        super(CORnet_IRL, self).__init__()

        self.V1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                                bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())]))

        self.V2 = CORblock_S(64, 128, times=2)

        self.V4 = CORblock_S(128, 256, times=4)

        self.IT = CORblock_S(256, 512, times=2)

        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())]))

        # normals_branch branches from V4, predicts surface normals
        self.normals_branch = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear1', nn.Linear(256, 512)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(p=0.5)),
            ('linear2', nn.Linear(512, 9408))]))

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, inp):
        shared = inp
        shared = self.V1(shared)
        shared = self.V2(shared)
        shared = self.V4(shared)
        classification = self.IT(shared)
        classification = self.decoder(classification)
        normals = self.normals_branch(shared)
        return normals, classification


def __cornet_irl(pretrained=None, map_location=None):
    """
    This method returns an instance of cornet_irl.

    Args:
        pretrained (string): Denotes whether model should take weights from
            pretrained cornet_s, from pretrained cornet_irl, or start
            from scratch.
        map_location (string): Location of the map
    """
    model = CORnet_IRL()
    model = torch.nn.DataParallel(model)

    if pretrained is None:
        pass
    elif pretrained == "cornet_s":
        url = f"https://s3.amazonaws.com/cornet-models/cornet_s-{HASH}.pth"
        ckpt_data =\
            torch.utils.model_zoo.load_url(url, map_location=map_location)
        model.load_state_dict(ckpt_data['state_dict'], strict=False)
    else:
        assert pretrained == "cornet_irl"
        url = f"https://958.s3.us-east-2.amazonaws.com/FMP_4612_model.pth.tar"
        cornet_irl_data =\
            torch.utils.model_zoo.load_url(url, map_location=map_location)
        model.load_state_dict(cornet_irl_data['state_dict'], strict=True)

    return model


def get_cornet_irl(pretrained=None, n_gpus=0):
    """
    This method returns an instance of cornet_irl.

    Args:
        pretrained (string): Denotes whether model should take weights from
            pretrained cornet_s, from pretrained cornet_irl, or start
            from scratch.
        n_gpus: (int): Number of gpus
    """
    assert pretrained is None or pretrained == "cornet_s"\
        or pretrained == "cornet_irl"

    if n_gpus == 0:
        model = __cornet_irl(pretrained=pretrained, map_location='cpu')
        model = model.module  # remove DataParallel
    else:
        model = __cornet_irl(pretrained=pretrained, map_location=None)
        model = model.cuda()
    return model
