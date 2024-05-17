import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChannelAttention3D(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.shared_MLP = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        return x * channel_attention

class SpatialAttention3D(nn.Module):
    def __init__(self):
        super(SpatialAttention3D, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.conv(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        return x * spatial_attention

class CBAM3D(nn.Module):
    def __init__(self, channel, reduction=2):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(channel, reduction)
        self.spatial_attention = SpatialAttention3D()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x