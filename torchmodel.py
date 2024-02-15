import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.f = [32,64,128,256,512]
        # Encoder
        self.encoder1 = self.encoder_block(in_channels, self.f[0])
        self.encoder2 = self.encoder_block(self.f[0], self.f[1])
        self.encoder3 = self.encoder_block(self.f[1], self.f[2])
        self.encoder4 = self.encoder_block(self.f[2], self.f[3])
        self.encoder5 = self.encoder_block(self.f[3], self.f[4])

        # Decoder
        self.decoder1 = self.decoder_block(self.f[4], self.f[3])
        self.decoder2 = self.decoder_block(self.f[3]*2, self.f[2])
        self.decoder3 = self.decoder_block(self.f[2]*2, self.f[1])
        self.decoder4 = self.decoder_block(self.f[1]*2, self.f[0])

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
        
        print(e1.shape)
        print(e2.shape)
        print(e3.shape)
        print(e4.shape)
        print(e5.shape)

        x = self.decoder1(e5)
        print(x.shape)
        x = self.decoder2(torch.cat((x,e4),1))
        print(x.shape)
        x = self.decoder3(torch.cat((x,e3),1))
        print(x.shape)
        x = self.decoder4(torch.cat((x,e2),1))
        
        output = self.output(torch.cat((x,e1),1))

        return output