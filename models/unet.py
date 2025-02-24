import torch
import torch.nn as nn
from torchvision import transforms

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DeepUNet(nn.Module):
    def __init__(self, num_classes=4):
        super(DeepUNet, self).__init__()
        self.dropout = nn.Dropout2d(0.3)
        
        # Encoder
        self.enc1 = self._make_encoder_block(3, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        self.enc5 = self._make_encoder_block(512, 1024)
        self.enc6 = self._make_encoder_block(1024, 2048)
        
        # Decoder
        self.upconv6 = self.upconv_block(2048, 1024)
        self.upconv5 = self.upconv_block(1024, 512)
        self.upconv4 = self.upconv_block(512, 256)
        self.upconv3 = self.upconv_block(256, 128)
        self.upconv2 = self.upconv_block(128, 64)
        self.upconv1 = self.upconv_block(64, num_classes, final=True)

        # Fix attention gates with correct channel dimensions
        # F_g is the number of channels from the higher level (decoder)
        # F_l is the number of channels from the skip connection (encoder)
        # F_int is the intermediate channel dimension
        self.attention5 = AttentionGate(F_g=1024, F_l=1024, F_int=512)
        self.attention4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.attention3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.attention2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.attention1 = AttentionGate(F_g=64, F_l=64, F_int=32)

        # Bottleneck
        self.bottleneck = nn.Conv2d(2048, 2048, kernel_size=1)
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_encoder_block(self, in_channels, out_channels):
        """Convolutional block with residual connection"""
        class ResBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
                self.norm1 = nn.InstanceNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
                self.norm2 = nn.InstanceNorm2d(out_ch)
                self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
                self.dropout = nn.Dropout2d(0.2)
                self.downsample = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

            def forward(self, x):
                identity = self.downsample(x)
                out = self.conv1(x)
                out = self.norm1(out)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.conv2(out)
                out = self.norm2(out)
                out += identity
                out = self.relu(out)
                return out

        class SEBlock(nn.Module):
            def __init__(self, channel, reduction=16):
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, x):
                b, c, _, _ = x.size()
                y = self.avg_pool(x).view(b, c)
                y = self.fc(y).view(b, c, 1, 1)
                return x * y.expand_as(x)

        return nn.Sequential(
            ResBlock(in_channels, out_channels),
            SEBlock(out_channels),
            self.dropout
        )

    def upconv_block(self, in_channels, out_channels, final=False):
        """Helper function for upsampling blocks"""
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        ]
        if not final:
            layers.extend([
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(0.2)
            ])
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True) if not final else nn.Identity()
        ])
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)         # 64 channels
        enc2 = self.enc2(enc1)      # 128 channels
        enc3 = self.enc3(enc2)      # 256 channels
        enc4 = self.enc4(enc3)      # 512 channels
        enc5 = self.enc5(enc4)      # 1024 channels
        enc6 = self.enc6(enc5)      # 2048 channels

        # Bottleneck
        bottleneck = self.bottleneck(enc6)  # 2048 channels

        # Decoder with attention
        dec6 = self.upconv6(bottleneck)     # 1024 channels
        dec5 = self.upconv5(self.attention5(dec6, enc5))  # 512 channels
        dec4 = self.upconv4(self.attention4(dec5, enc4))  # 256 channels
        dec3 = self.upconv3(self.attention3(dec4, enc3))  # 128 channels
        dec2 = self.upconv2(self.attention2(dec3, enc2))  # 64 channels
        dec1 = self.upconv1(self.attention1(dec2, enc1))  # num_classes channels

        return dec1