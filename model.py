import torch
import torch.nn as nn
from config import architecture_config


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_normalization = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_normalization(x)
        x - self.leaky_relu(x)
        return x


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture_config = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture_config)
        self.fc = self._create_fully_connected(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def _create_conv_layers(self, architecture_config):
        layers = []
        in_channels = self.in_channels

        for config in architecture_config:
            if type(config) == tuple:
                layers += [CNNBlock(in_channels, out_channels=config[1], kernel_size=config[0], stride=config[2],
                                    padding=config[3])]
                in_channels = config[1]

            elif type(config) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]

            elif type(config) == list:
                conv1 = config[0]
                conv2 = config[1]
                num_repeats = config[2]

                for num in range(num_repeats):
                    layers += [CNNBlock(in_channels, out_channels=conv1[1], kernel_size=conv1[0], stride=conv1[2],
                                        padding=conv1[3])]
                    layers += [
                        CNNBlock(in_channels=conv1[1], out_channels=conv2[1], kernel_size=conv2[0], stride=conv2[2],
                                 padding=conv2[3])]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fully_connected(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(nn.Flatten(), nn.Linear(1024 * S * S, 4096), nn.Dropout(0.0), nn.LeakyReLU(0.1),
                             nn.Linear(4096, S * S * (B * 5 + C)))



