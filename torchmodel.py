import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.encoder_block(in_channels, 32)
        self.encoder2 = self.encoder_block(32, 64)
        self.encoder3 = self.encoder_block(64, 128)
        self.encoder4 = self.encoder_block(128, 256)
        self.encoder5 = self.encoder_block(256, 512)

        # Decoder
        self.decoder1 = self.decoder_block(512, 256)
        self.decoder2 = self.decoder_block(256, 128)
        self.decoder3 = self.decoder_block(128, 64)
        self.decoder4 = self.decoder_block(64, 32)

        # Output layer
        self.output_layer = nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        x = self.decoder1(e5)
        x = self.decoder2(x + e4)
        x = self.decoder3(x + e3)
        x = self.decoder4(x + e2)
        output = self.output_layer(x)
        return torch.sigmoid(output)