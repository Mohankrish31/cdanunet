class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
        
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Match size (for odd image dims)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CDAN_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_ch=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
    
        self.bottom = DoubleConv(base_ch*8, base_ch*16)
        self.cdan_bottom = CDAN(base_ch*16)

        self.up3 = Up(base_ch*16, base_ch*8)
        self.up2 = Up(base_ch*8, base_ch*4)
        self.up1 = Up(base_ch*4, base_ch*2)
        self.up0 = Up(base_ch*2, base_ch)

        self.outc = nn.Conv2d(base_ch, out_channels, kernel_size=1)

        # Add attention after each encoder/decoder block
        self.att1 = CDAN(base_ch*2)
        self.att2 = CDAN(base_ch*4)
        self.att3 = CDAN(base_ch*8)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.att1(x2)
        x3 = self.down2(x2)
        x3 = self.att2(x3)
        x4 = self.down3(x3)
        x4 = self.att3(x4)

        xb = self.bottom(x4)
        xb = self.cdan_bottom(xb)

        x = self.up3(xb, x4)
        x = self.up2(x, x3)
        x = self.up1(x, x2)
        x = self.up0(x, x1)
        out = self.outc(x)
        return out
