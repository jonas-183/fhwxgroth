import torch
import torch.nn as nn
from torchvision import transforms

class DeepUNet(nn.Module):
    def __init__(self, num_classes=4):
        super(DeepUNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 64)  # Eingabekanal auf 3 setzen
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)  # Tieferer Encoder

        # Decoder
        self.upconv5 = self.upconv_block(1024, 512)
        self.upconv4 = self.upconv_block(512, 256)
        self.upconv3 = self.upconv_block(256, 128)
        self.upconv2 = self.upconv_block(128, 64)
        self.upconv1 = self.upconv_block(64, num_classes, final=True)

        # Bottleneck
        self.bottleneck = nn.Conv2d(1024, 1024, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Hilfsfunktion für Convolutional Blocks"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def upconv_block(self, in_channels, out_channels, final=False):
        """Hilfsfunktion für Upsampling-Blocks"""
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        ]
        if not final:
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(enc5)

        # Decoder
        dec5 = self.upconv5(bottleneck)
        dec4 = self.upconv4(dec5 + enc4)  # Skip Connection
        dec3 = self.upconv3(dec4 + enc3)  # Skip Connection
        dec2 = self.upconv2(dec3 + enc2)  # Skip Connection
        dec1 = self.upconv1(dec2 + enc1)  # Skip Connection

        return dec1