import torch
import torch.nn as nn

# Channel Attention (CA)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(-2, -1), keepdim=True)  # Global Avg Pooling
        max_pool = torch.amax(x, dim=(-2, -1), keepdim=True)  # Global Max Pooling
        out = self.fc1(avg_pool) + self.fc1(max_pool)
        return torch.sigmoid(self.fc2(self.relu(out))) * x

# Spatial Attention (SA)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Channel Avg Pooling
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # Channel Max Pooling
        out = torch.cat([avg_pool, max_pool], dim=1)
        return torch.sigmoid(self.conv(out)) * x

# CBAM Module
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
