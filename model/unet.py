import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class CNNBlock(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int, 
                 kernel_size: int,
                 stride: int = 1, 
                 apply_bn: bool = True, 
                 apply_relu: bool = True):
        
        super().__init__()
        
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=kernel_size // 2)]
        if apply_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if apply_relu:
            layers.append(nn.ReLU())
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, init_ch: int):
        
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.init_ch = init_ch

        self.inc = nn.Sequential(
            CNNBlock(self.in_channels, self.init_ch, 5),
            CNNBlock(self.init_ch, self.init_ch, 5))

        self.down_00 = nn.Sequential(
            CNNBlock(self.init_ch, self.init_ch * 2, 5),
            CNNBlock(self.init_ch * 2, self.init_ch * 2, 5))

        self.down_01 = nn.Sequential(
            CNNBlock(self.init_ch * 2, self.init_ch * 4, 5),
            CNNBlock(self.init_ch * 4, self.init_ch * 4, 5))
        
        self.down_02 = nn.Sequential(
            CNNBlock(self.init_ch * 4, self.init_ch * 4, 5),
            CNNBlock(self.init_ch * 4, self.init_ch * 4, 5))
        
        self.up_00 = nn.Sequential(
            CNNBlock(self.init_ch * 8, self.init_ch * 4, 5),
            CNNBlock(self.init_ch * 4, self.init_ch * 2, 5))
        
        self.up_01 = nn.Sequential(
            CNNBlock(self.init_ch * 4, self.init_ch * 2, 5),
            CNNBlock(self.init_ch * 2, self.init_ch, 5))
        
        self.up_02 = nn.Sequential(
            CNNBlock(self.init_ch * 2, self.init_ch, 5),
            CNNBlock(self.init_ch, self.init_ch, 5))

        self.outc = nn.Conv2d(self.init_ch, self.in_channels, 1, 1, 0)

        self.upconv_00_to_01 = nn.ConvTranspose2d(self.init_ch * 4, self.init_ch * 4, 2, 2, 0)
        self.upconv_01_to_02 = nn.ConvTranspose2d(self.init_ch * 2, self.init_ch * 2, 2, 2, 0)
        self.upconv_02_to_out = nn.ConvTranspose2d(self.init_ch, self.init_ch, 2, 2, 0)
        
        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        inc_out = self.inc(x)
        inc_pool = self.pooling(inc_out)
        
        down_00_out = self.down_00(inc_pool)
        down_00_pool = self.pooling(down_00_out)
        
        down_01_out = self.down_01(down_00_pool)
        down_01_pool = self.pooling(down_01_out)
        
        down_02_out = self.down_02(down_01_pool)
        down_02_upconv = self.upconv_00_to_01(down_02_out)

        up_00_out = self.up_00(torch.cat([down_02_upconv, down_01_out], dim=1))
        up_00_upconv = self.upconv_01_to_02(up_00_out)
        
        up_01_out = self.up_01(torch.cat([up_00_upconv, down_00_out], dim=1))
        up_01_upconv = self.upconv_02_to_out(up_01_out)
        
        up_02_out = self.up_02(torch.cat([up_01_upconv, inc_out], dim=1))
        out = self.outc(up_02_out)
        return out
