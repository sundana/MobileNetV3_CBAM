import torch
import torch.nn as nn
from my_models.channel_shuffle import ChannelShuffle


class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, groups = 2):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.channel_split = lambda x: torch.split(x, x.size(1) // 2, dim=1)

        # Layers for one branch
        self.branch_transform = nn.Sequential(
            # Pointwise convolution (expand)
            nn.Conv2d(in_channels // 2, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # Pointwise convolution (project)
            nn.Conv2d(hidden_dim, out_channels // 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels // 2),
        )

        # Channel shuffle
        self.channel_shuffle = ChannelShuffle(groups)

    def forward(self, x):
        # Split channels
        x1, x2 = self.channel_split(x)

        # Transform one branch
        x1 = self.branch_transform(x1)

        # Concatenate and shuffle channels
        out = torch.cat((x1, x2), dim=1)
        out = self.channel_shuffle(out)

        # Residual connection if applicable
        if self.use_residual:
            out += x
        return out
    


class MobileShuffleNet(nn.Module):
    def __init__(self, inverted_residual_setting, last_channels=1024, drop=0.2, num_classes=1000):
        super().__init__()
        features = []
        features.append(nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        ))
        for ic, ec, oc, ks, s, act, se in inverted_residual_setting:
            features.append(InvertedBottleneck(in_channels=ic, 
                                               out_channels=oc,
                                               stride=s,
                                               expand_ratio=ec))
        lastconv_in_channels = inverted_residual_setting[-1][2]
        lastconv_out_channels = 6 * lastconv_in_channels
        features.append(nn.Sequential(
            nn.Conv2d(lastconv_in_channels, lastconv_out_channels, kernel_size=1),
            nn.BatchNorm2d(lastconv_out_channels),
            nn.Hardswish(inplace=True)
            ))
        self.featuers = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(lastconv_out_channels, last_channels),
            nn.Hardswish(inplace=True),
            nn.Dropout(drop, inplace=True),
            nn.Linear(last_channels, num_classes)
        )

    def forward(self, x):
        x = self.featuers(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x


mobilenetv3_small_setting = [
    # input_channels, expand_channel, output_channel, kernel size, 
    # stride, activation, use_se
    [16, 16, 16, 3, 2, "RE", True],
    [16, 72, 24, 3, 2, "RE", False],
    [24, 88, 24, 3, 1, "RE", False],
    [24, 96, 40, 5, 2, "HS", True],
    [40, 240, 40, 5, 1, "HS", True],
    [40, 240, 40, 5, 1, "HS", True],
    [40, 120, 48, 5, 1, "HS", True],
    [48, 144, 48, 5, 1, "HS", True],
    [48, 288, 96, 5, 2, "HS", True],
    [96, 576, 96, 5, 1, "HS", True],
    [96, 576, 96, 5, 1, "HS", True]

]
mobilenetv3_large_setting = [
    [16, 16, 16, 3, 1, "RE", False],
    [16, 64, 24, 3, 2, "RE", False],
    [24, 72, 24, 3, 1, "RE", False],
    [24, 72, 40, 5, 2, "RE", True],
    [40, 120, 40, 5, 1, "RE", True],
    [40, 120, 40, 5, 1, "RE", True],
    [40, 240, 80, 3, 2, "HS", False],
    [80, 200, 80, 3, 1, "HS", False],
    [80, 184, 80, 3, 1, "HS", False],
    [80, 184, 80, 3, 1, "HS", False],
    [80, 480, 112, 3, 1, "HS", True],
    [112, 672, 112, 3, 1, "HS", True],
    [112, 672, 160, 5, 2, "HS", True],
    [160, 960, 160, 5, 1, "HS", True],
    [160, 960, 160, 5, 1, "HS", True],
]
mobilenetv3_large_setting = [
    [16, 4, 16, 3, 1, "RE", False],
    [16, 4, 24, 3, 2, "RE", False],
    [24, 4, 24, 3, 1, "RE", False],
    [24, 4, 40, 5, 2, "RE", True],
    [40, 4, 40, 5, 1, "RE", True],
    [40, 4, 40, 5, 1, "RE", True],
    [40, 4, 80, 3, 2, "HS", False],
    [80, 6, 80, 3, 1, "HS", False],
    [80, 6, 80, 3, 1, "HS", False],
    [80, 6, 80, 3, 1, "HS", False],
    [80, 6, 112, 3, 1, "HS", True],
    [112, 6, 112, 3, 1, "HS", True],
    [112, 6, 160, 5, 2, "HS", True],
    [160, 6, 160, 5, 1, "HS", True],
    [160, 6, 160, 5, 1, "HS", True],
]

def mobileshufflenet_small(num_classes=1000):
    model = MobileShuffleNet(mobilenetv3_small_setting, last_channels=1024,
                        num_classes=num_classes)
    return model

def mobileshufflenet_large(num_classes=1000):
    model = MobileShuffleNet(mobilenetv3_large_setting, last_channels=1280,
                        num_classes=num_classes)
    return model

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = mobileshufflenet_small()
    y = model(x)
    print(y.shape)