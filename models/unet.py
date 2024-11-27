import torch

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
    def __init__(self, num_classes=3):
        super(UNet, self).__init__()
        self.encoder = torch.nn.Sequential(
            ResidualBlock(3, 64),
            ResidualBlock(64, 128),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
