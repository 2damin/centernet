import torch
import torch.nn as nn

from .model_parts import *

class ResNet(nn.Module):
    def __init__(self, n_channels, n_classes ):
        super(ResNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv1 = Conv(3, 64)

        self.max_pool1 = nn.MaxPool2d(3, stride=2)

        self.conv2_x = ResidualBlock(64, 64, 256)

        self.conv3_x = ResidualBlock(256, 128, 512)

        self.conv4_x = ResidualBlock(512, 256, 1024)

        self.conv5_x = ResidualBlock(1024, 512, 2048)
    
        self.avgPool = nn.AvgPool2d(4)
        
        self.fc = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        for _ in range(3):
            x = self.conv2_x(x)
        for _ in range(4):
            x = self.conv3_x(x)
        for _ in range(6):
            x = self.conv4_x(x)
        for _ in range(3):
            x = self.conv5_x(x)
        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.Softmax(x)
        return x

        


