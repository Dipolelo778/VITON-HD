import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Basic Residual Block."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class UNetResNet(nn.Module):
    """Advanced U-Net with ResNet Blocks."""

    def __init__(self, input_nc=6, output_nc=3, num_filters=64):
        super(UNetResNet, self).__init__()

        # Encoder
        self.enc1 = ResBlock(input_nc, num_filters)
        self.enc2 = ResBlock(num_filters, num_filters * 2)
        self.enc3 = ResBlock(num_filters * 2, num_filters * 4)
        self.enc4 = ResBlock(num_filters * 4, num_filters * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResBlock(num_filters * 8, num_filters * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(num_filters * 16, num_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ResBlock(num_filters * 16, num_filters * 8)

        self.up3 = nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ResBlock(num_filters * 8, num_filters * 4)

        self.up2 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResBlock(num_filters * 4, num_filters * 2)

        self.up1 = nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=2, stride=2)
        self.dec1 = ResBlock(num_filters * 2, num_filters)

        self.final = nn.Conv2d(num_filters, output_nc, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return torch.tanh(out)
