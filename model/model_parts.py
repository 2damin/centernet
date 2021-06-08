import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 7, stride = 2, padding = 3)

    def forward(self, x):
        return self.conv(x)

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
    
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.residualBlock = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size = 1),
                                        nn.BatchNorm2d(mid_channels),
                                        nn.ReLU(),
                                        nn.Conv2d(mid_channels, mid_channels, kernel_size = 3),
                                        nn.BatchNorm2d(mid_channels),
                                        nn.ReLU(),
                                        nn.Conv2d(mid_channels, out_channels, kernel_size = 1),
                                        nn.BatchNorm2d(out_channels))

        self.conv_pointwise = PointwiseConv(in_channels, out_channels)
    
    def forward(self, x):
        shortcut = self.conv_pointwise(x)
        return nn.Relu(torch.add(self.residualBlock(x) + shortcut))
