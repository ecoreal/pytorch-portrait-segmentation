import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, high, low):
        x1 = self.up(high)
        x1 = F.interpolate(x1, size=(low.size(2), low.size(3)), mode="bilinear", align_corners=False)
        x1 = torch.cat([x1, low], dim=1)
        x1 = self.conv_relu(x1)
        return x1


class UNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.decoder4 = Decoder(1024, 512)
        self.decoder3 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder1 = Decoder(128, 64)
        self.last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        layer1 = self.layer1(input)
        layer2 = self.layer2(self.maxpool(layer1))
        layer3 = self.layer3(self.maxpool(layer2))
        layer4 = self.layer4(self.maxpool(layer3))
        layer5 = self.layer5(self.maxpool(layer4))

        layer6 = self.decoder4(layer5, layer4)
        layer7 = self.decoder3(layer6, layer3)
        layer8 = self.decoder2(layer7, layer2)
        layer9 = self.decoder1(layer8, layer1)
        output = self.last(layer9)

        return output
