import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        assert num_channels % self.groups == 0
        group_channels = num_channels // self.groups
        x = x.view(batch_size, self.groups, group_channels, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x
