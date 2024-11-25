import torch
from torch import nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
  def __init__(self, input_channels, reduction_ratio=4):
    super().__init__()
    reduced_channels = input_channels // reduction_ratio
    self.fc1 = nn.Conv2d(input_channels, reduced_channels, kernel_size=1)
    self.fc2 = nn.Conv2d(reduced_channels, input_channels, kernel_size=1)
  def forward(self, x):
    scale = F.adaptive_avg_pool2d(x, 1)  # Global Average Pooling
    scale = F.relu(self.fc1(scale))
    scale = torch.sigmoid(self.fc2(scale))
    return x * scale


class MobileNetV3Block(nn.Module):
    """Inverted Residual Block with optional SE and H-Swish."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se, activation):
        super().__init__()
        self.use_se = use_se
        hidden_dim = int(in_channels * expand_ratio) # Ensure hidden_dim is an integer
        self.expand = in_channels != hidden_dim
        self.activation = activation

        # Expand phase
        if self.expand:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)


        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # Squeeze-and-Excitation
        if self.use_se:
            self.se = SqueezeExcitation(hidden_dim)

        # Pointwise convolution
        self.pointwise_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.use_res_connect = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        if self.expand:
            x = self.activation(self.bn1(self.expand_conv(x)))
        x = self.activation(self.bn2(self.depthwise_conv(x)))
        if self.use_se:
            x = self.se(x)
        x = self.bn3(self.pointwise_conv(x))
        if self.use_res_connect:
            return x + identity
        else:
            return x

class MobileNetV3(nn.Module):
    """MobileNetV3 (small or large)."""
    def __init__(self, num_classes=1000, mode='large'):
        super().__init__()
        self.mode = mode

        # Configuration (based on Table 1 and 2 in the paper)
        config = [
            # in_channels, out_channels, kernel, stride, expand, SE, activation
            [16, 16, 3, 1, 1, False, nn.ReLU()],
            [16, 24, 3, 2, 4, False, nn.ReLU()],
            [24, 24, 3, 1, 3, False, nn.ReLU()],
            [24, 40, 5, 2, 3, True, nn.Hardswish()],
            [40, 40, 5, 1, 3, True, nn.Hardswish()],
            [40, 80, 3, 2, 6, False, nn.Hardswish()],
            [80, 80, 3, 1, 2.5, False, nn.Hardswish()],
            [80, 112, 3, 1, 6, True, nn.Hardswish()],
            [112, 160, 5, 2, 6, True, nn.Hardswish()],
        ]

        # First convolution
        self.first_conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.first_bn = nn.BatchNorm2d(16)
        self.first_activation = nn.Hardswish()

        # Building MobileNetV3 Blocks
        layers = []
        in_channels = 16
        for c in config:
            layers.append(
                MobileNetV3Block(
                    in_channels=in_channels,
                    out_channels=c[1],
                    kernel_size=c[2],
                    stride=c[3],
                    expand_ratio=c[4],
                    use_se=c[5],
                    activation=c[6]
                )
            )
            in_channels = c[1]
        self.blocks = nn.Sequential(*layers)

        # Final layers
        self.last_conv = nn.Conv2d(160, 960, kernel_size=1, bias=False)
        self.last_bn = nn.BatchNorm2d(960)
        self.last_activation = nn.Hardswish()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(960, num_classes)

    def forward(self, x):
        x = self.first_activation(self.first_bn(self.first_conv(x)))
        x = self.blocks(x)
        x = self.last_activation(self.last_bn(self.last_conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x