import torch
import torch.nn as nn
from torchvision import transforms

import torch.nn.functional as F
import torchvision.models as models

# Residual Block für komplexere Modellarchitektur
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

        # 1x1 Convolution, um Dimensionen anzupassen, falls notwendig
        if in_channels != out_channels:
            self.residual_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)

        # Residual-Anpassung
        if self.residual_conv:
            residual = self.residual_conv(residual)

        return self.relu(x + residual)

# U-Net
class UNet(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(UNet, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), torch.nn.ReLU(),
            torch.nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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


class DeeperUNet(nn.Module):
    def __init__(self, num_classes=4):
        super(DeeperUNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 64)  # Eingabekanal auf 3 setzen
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)
        self.enc6 = self.conv_block(1024, 2048)
        self.enc7 = self.conv_block(2048, 4096)  # Tieferer Encoder

        # Decoder
        self.upconv7 = self.upconv_block(4096, 2048)
        self.upconv6 = self.upconv_block(2048, 1024)
        self.upconv5 = self.upconv_block(1024, 512)
        self.upconv4 = self.upconv_block(512, 256)
        self.upconv3 = self.upconv_block(256, 128)
        self.upconv2 = self.upconv_block(128, 64)
        self.upconv1 = self.upconv_block(64, num_classes, final=True)

        # Bottleneck
        self.bottleneck = nn.Conv2d(4096, 4096, kernel_size=1)

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

    def center_crop(self, enc, dec):
        """Schneidet den Encoder-Ausgabe-Tensor auf die Größe des Decoder-Eingabe-Tensors zurecht"""
        _, _, h, w = dec.size()
        enc = transforms.CenterCrop([h, w])(enc)
        return enc

    def channel_match(self, enc, dec):
        """Passt die Kanäle der Encoder-Ausgabe an die des Decoders an."""
        if enc.size(1) != dec.size(1):  # Wenn die Kanäle nicht übereinstimmen
            conv = nn.Conv2d(enc.size(1), dec.size(1), kernel_size=1).to(enc.device)  # 1x1-Conv auf das gleiche Gerät verschieben
            enc = conv(enc)  # Kanäle anpassen
        return self.center_crop(enc, dec)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # 64 Kanäle
        enc2 = self.enc2(enc1)  # 128 Kanäle
        enc3 = self.enc3(enc2)  # 256 Kanäle
        enc4 = self.enc4(enc3)  # 512 Kanäle
        enc5 = self.enc5(enc4)  # 1024 Kanäle
        enc6 = self.enc6(enc5)  # 2048 Kanäle
        enc7 = self.enc7(enc6)  # 4096 Kanäle

        # Bottleneck
        bottleneck = self.bottleneck(enc7)  # 4096 Kanäle

        # Decoder mit Kanalanpassung
        dec7 = self.upconv7(bottleneck)  # 2048 Kanäle
        dec7 = dec7 + self.channel_match(enc7, dec7)  # Skip-Connection (2048 Kanäle)

        dec6 = self.upconv6(dec7)  # 1024 Kanäle
        dec6 = dec6 + self.channel_match(enc6, dec6)  # Skip-Connection (1024 Kanäle)

        dec5 = self.upconv5(dec6)  # 512 Kanäle
        dec5 = dec5 + self.channel_match(enc5, dec5)  # Skip-Connection (512 Kanäle)

        dec4 = self.upconv4(dec5)  # 256 Kanäle
        dec4 = dec4 + self.channel_match(enc4, dec4)  # Skip-Connection (256 Kanäle)

        dec3 = self.upconv3(dec4)  # 128 Kanäle
        dec3 = dec3 + self.channel_match(enc3, dec3)  # Skip-Connection (128 Kanäle)

        dec2 = self.upconv2(dec3)  # 64 Kanäle
        dec2 = dec2 + self.channel_match(enc2, dec2)  # Skip-Connection (64 Kanäle)

        dec1 = self.upconv1(dec2)  # Endausgabe

        return dec1