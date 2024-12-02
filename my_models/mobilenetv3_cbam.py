""" 
PyTorch implementation of Searching for MobileNetV3

As described in https://arxiv.org/pdf/1905.02244

MobileNetV3 is tuned to mobile phone CPUs through a combination of hardwareaware 
network architecture search (NAS) complemented by the NetAdapt algorithm and then 
subsequently improved through novel architecture advances
"""


import torch
from torch import nn
from my_models.cbam import CBAM

class InvertedResidualWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, reduction=16):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        # Expand
        self.expand = nn.Conv2d(in_channels, hidden_dim, 1, bias=False) if expand_ratio != 1 else None
        self.expand_bn = nn.BatchNorm2d(hidden_dim) if self.expand else None
        self.expand_relu = nn.ReLU6(inplace=True) if self.expand else None

        # Depthwise Convolution
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        self.depthwise_relu = nn.ReLU6(inplace=True)

        # Project
        self.project = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

        # Initialize CBAM only if reduction is specified
        self.cbam = CBAM(out_channels, reduction) if reduction else None

    def forward(self, x):
        identity = x
        if self.expand:
            x = self.expand_relu(self.expand_bn(self.expand(x)))
        x = self.depthwise_relu(self.depthwise_bn(self.depthwise(x)))
        x = self.project_bn(self.project(x))
        
        if self.cbam:  # Apply CBAM only if initialized
            x = self.cbam(x)
            
        if self.use_residual:
            return x + identity
        return x


    
class MobileNetV3WithCBAM(nn.Module):
    def __init__(self, mode='large', num_classes=1000, reduction=16):
        super(MobileNetV3WithCBAM, self).__init__()
        self.cfgs = self._get_cfgs(mode)
        input_channel = 16
        last_channel = 1280 if mode == 'large' else 1024

        # First Layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.Hardswish()
        )

        # Inverted Residual Blocks with CBAM
        self.blocks = nn.ModuleList()
        for k, t, c, use_cbam, s in self.cfgs:
            out_channel = c
            self.blocks.append(
                InvertedResidualWithCBAM(input_channel, out_channel, s, t, reduction=reduction) if use_cbam else
                InvertedResidualWithCBAM(input_channel, out_channel, s, t, reduction=None)
            )
            input_channel = out_channel

        # Final Layers
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.Hardswish()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_conv(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    @staticmethod
    def _get_cfgs(mode):
        if mode == 'large':
            return [
                # (kernel_size, expand_ratio, out_channels, use_cbam, stride)
                (3, 1, 16, False, 1),
                (3, 4, 24, False, 2),
                (3, 3, 24, False, 1),
                (5, 3, 40, True, 2),
                (5, 3, 40, True, 1),
                (3, 6, 80, True, 2),
                (3, 6, 80, True, 1),
                (3, 6, 112, True, 1),
                (3, 6, 160, True, 2),
                (3, 6, 160, True, 1),
            ]
        else:
            return [
                # Small configuration
                (3, 1, 16, False, 2),
                (3, 4.5, 24, False, 2),
                (3, 3.67, 40, True, 2),
                (5, 6, 48, True, 1),
                (5, 6, 96, True, 2),
            ]
