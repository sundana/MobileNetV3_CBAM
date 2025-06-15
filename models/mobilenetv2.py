import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)
        


class MobileNetV2(nn.Module):
    def __init__(self, num_classes = 1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        inverted_residual_config = [
            # t, c, n, s
            [1, 16, 1, 1],  # t: expansion factor, c: output channels, n: number of blocks, s: stride
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # First layer: regular conv
        input_channel = int(32 * width_mult)
        layers = [nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(input_channel),
                  nn.ReLU6(inplace=True)]

        # Inverted Residual Block
        for t, c, n, s in inverted_residual_config:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidualBlock(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Last few layers
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1200
        layers.append(nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(last_channel))
        layers.append(nn.ReLU6(inplace=True))

        self.features = nn.Sequential(*layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_channel, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
