import torch
import torch.nn as nn
import torch.nn.functional as F

class FireModule(nn.Module):
    """
    Fire Module: Squeeze + Expand Layers
    """
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.expand_1x1(x), self.expand_3x3(x)], dim=1)

class SqueezeNet(nn.Module):
    """
    SqueezeNet Architecture
    """
    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            # Initial Convolution Layer
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # Fire Modules
            FireModule(96, 16, 64),
            FireModule(128, 16, 64),
            FireModule(128, 32, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128),
            FireModule(256, 48, 192),
            FireModule(384, 48, 192),
            FireModule(384, 64, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256),
            
            # Final Convolution Layer
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)