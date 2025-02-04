""" 
PyTorch implementation of Searching for MobileNetV3

As described in https://arxiv.org/pdf/1905.02244

MobileNetV3 is tuned to mobile phone CPUs through a combination of hardwareaware 
network architecture search (NAS) complemented by the NetAdapt algorithm and then 
subsequently improved through novel architecture advances
"""


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from my_models.cbam import CBAM

class SELayer(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, hidden_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, expand_channels, out_channels,
                 kernel_size, stride, act="HS", use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, expand_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(expand_channels)
        self.act = nn.Hardswish() if act == "HS" else nn.ReLU()

        self.conv2 = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size,
                               stride=stride, padding=(kernel_size - 1) // 2, groups=expand_channels)
        self.bn2 = nn.BatchNorm2d(expand_channels)
        
        self.conv3 = nn.Conv2d(expand_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = stride == 1 and in_channels == out_channels
        self.se = SELayer(expand_channels, int(expand_channels // 4)) if use_se else None
    
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        if self.se is not None:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.shortcut:
            out += x
        return out


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, module, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.module = module
        
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.module != None:
            out = self.module(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.Hardswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), CBAM(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), CBAM(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), CBAM(40), 1),
            Block(3, 40, 240, 80, nn.Hardswish(), None, 2),
            Block(3, 80, 200, 80, nn.Hardswish(), None, 1),
            Block(3, 80, 184, 80, nn.Hardswish(), None, 1),
            Block(3, 80, 184, 80, nn.Hardswish(), None, 1),
            Block(3, 80, 480, 112, nn.Hardswish(), CBAM(112), 1),
            Block(3, 112, 672, 112, nn.Hardswish(), CBAM(112), 1),
            Block(5, 112, 672, 160, nn.Hardswish(), CBAM(160), 1),
            Block(5, 160, 672, 160, nn.Hardswish(), CBAM(160), 2),
            Block(5, 160, 960, 160, nn.Hardswish(), CBAM(160), 1),
        )
        
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = nn.Hardswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = nn.Hardswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out
    

class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.Hardswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), CBAM(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, nn.Hardswish(), CBAM(40), 2),
            Block(5, 40, 240, 40, nn.Hardswish(), CBAM(40), 1),
            Block(5, 40, 240, 40, nn.Hardswish(), CBAM(40), 1),
            Block(5, 40, 120, 48, nn.Hardswish(), CBAM(48), 1),
            Block(5, 48, 144, 48, nn.Hardswish(), CBAM(48), 1),
            Block(5, 48, 288, 96, nn.Hardswish(), CBAM(96), 2),
            Block(5, 96, 576, 96, nn.Hardswish(), CBAM(96), 1),
            Block(5, 96, 576, 96, nn.Hardswish(), CBAM(96), 1),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = nn.Hardswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = nn.Hardswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    
    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out
    


def test():
    net = MobileNetV3_Small()
    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    print(y.size())