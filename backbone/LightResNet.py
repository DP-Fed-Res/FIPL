import torch
import torch.nn as nn
import torch.nn.functional as F


class WSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight

        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight_std = weight.std(dim=(1, 2, 3), unbiased=False, keepdim=True)

        standardized_weight = (weight - weight_mean) / (weight_std + 1e-5)

        return F.conv2d(x, standardized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        WSConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=16, num_channels=out_channels),
        nn.ReLU()
    ]
    if pool:
        layers.append(nn.AvgPool2d(2))
    return nn.Sequential(*layers)


class LightResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LightResNet, self).__init__()
        self.name = 'lightresnet'
        base_channels = 16

        self.conv1 = conv_block(in_channels, base_channels)
        self.conv2 = conv_block(base_channels, base_channels * 2, pool=True)

        self.res1 = nn.Sequential(
            conv_block(base_channels * 2, base_channels * 2),
            conv_block(base_channels * 2, base_channels * 2)
        )

        self.conv3 = conv_block(base_channels * 2, base_channels * 4, pool=True)
        self.conv4 = conv_block(base_channels * 4, base_channels * 8, pool=True)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def forward(self, xb, return_features: bool = False):
        out = self.conv1(xb)
        out = self.conv2(out)

        out = self.res1(out) + out

        out = self.conv3(out)
        out = self.conv4(out)

        out = self.pool(out)
        feature = self.flatten(out)
        out = self.fc(feature)
        if return_features:
            return feature, out
        return out
