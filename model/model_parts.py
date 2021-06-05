import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 7, stride = 2, padding = 1)

    def forward(self, x):
        return self.conv(x)

class ConvTriple(nn.Module):
    def __init__(self, in_channels, midOne_channels, midTwo_channels, out_channels):
        super().__init__()
        self.convTriple = nn.Sequential(nn.Conv2d(in_channels, midOne_channels, kernel_size = 1),
                                        nn.Conv2d(midOne_channels, midTwo_channels, kernel_size = 3),
                                        nn.Conv2d(midTwo_channels, out_channels, kernel_size = 1))
    
    def forward(self, x):
        return self.convTriple(x)