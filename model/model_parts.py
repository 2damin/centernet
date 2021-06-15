import torch
import torch.nn as nn
from typing import Optional

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 7, stride = 2, padding = 3)

    def forward(self, x):
        return self.conv(x)

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, bias = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias=bias)
    
    def forward(self, x):
        return self.conv(x)

class BottleNeck(nn.Module):

    expansion: int = 4

    def __init__(self,
                inplanes,
                planes,
                stride :int = 1,
                downsample: Optional[nn.Module] = None,
                groups: int = 1,
                base_width :int = 64):
        super().__init__()

        self.conv1 = PointwiseConv(inplanes, planes)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        outplanes = planes * self.expansion

        self.conv3 = PointwiseConv(planes, outplanes)

        self.bn3 = nn.BatchNorm2d(outplanes)

        self.downsample = downsample

    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out
