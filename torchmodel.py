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
        self.decoder2 = self.decoder_block(256*2, 128)
        self.decoder3 = self.decoder_block(128*2, 64)
        self.decoder4 = self.decoder_block(64*2, 32)

        # Output layer
        self.output = self.output_layer(32*2,16*2,out_channels)


    def output_layer(self,in_channels,middle_layer,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, middle_layer, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_layer),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_layer, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.Sigmoid()
        )

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
        x = self.decoder2(torch.cat((x,e4),1))
        x = self.decoder3(torch.cat((x,e3),1))
        x = self.decoder4(torch.cat((x,e2),1))
        
        output = self.output(torch.cat((x,e1),1))

        return output