import torch
import torch.nn as nn

from .model_parts import BottleNeck, Conv, PointwiseConv

BN_MOMENTUM = 0.1

class CenterNet_ResNet(nn.Module):
    def __init__(self, n_channels, n_classes, heads ):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.deconv_with_bias = False
        self.inplanes = 64
        self.heads = heads

        self.conv1 = Conv(3, 64)

        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        self.relu = nn.ReLU(inplace=True)

        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self.make_layer(BottleNeck, 64, num_blocks = 3, stride = 1)

        self.conv3_x = self.make_layer(BottleNeck, 128, num_blocks = 4, stride = 2)

        self.conv4_x = self.make_layer(BottleNeck, 256, num_blocks = 6, stride = 2)

        self.conv5_x = self.make_layer(BottleNeck, 512, num_blocks = 3, stride = 2)

        self.deconv_layers = self.make_deconv_layer(
            3,
            [256,256,256],
            [4,4,4],
        )
        self.inplanes = 512 * BottleNeck.expansion

        for head in sorted(self.heads):
            num_output = self.heads[head]

            fc = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Conv2d(64, num_output, kernel_size = 1, padding = 0, stride = 1)
            )
            self.__setattr__(head, fc)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.deconv_layers(x)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)

        return [ret]
    
    def get_deconv_cfg(self, deconv_kernel, index):
        padding = output_padding = 0
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def make_deconv_layer(self, num_layers, filters, kernels ):
        layers = []
        in_channel = self.inplanes
        for i in range(num_layers):
            kernel, padding, output_padding = self.get_deconv_cfg(kernels[i], i)

            planes = filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channel,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias = self.deconv_with_bias)
                )
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            in_channel = planes

        return nn.Sequential(*layers)

    def make_layer(self, block, planes, num_blocks: int, stride: int):
        downsample = None

        if stride != 1 or planes * block.expansion != self.inplanes:
            downsample = nn.Sequential(
                PointwiseConv(self.inplanes, planes * block.expansion, stride = stride),
                nn.BatchNorm2d(planes * block.expansion))

        layer = []
        layer.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        for i in range(1, num_blocks):
            layer.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layer)
        






        


