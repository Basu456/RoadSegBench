
import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class UNetBinary(nn.Module):
    """Reference binary-road U-Net; not a substitute for a separately specified variant."""
    def __init__(self, in_channels=3, out_channels=1, base=32):
        super().__init__()
        self.e1 = DoubleConv(in_channels, base)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = DoubleConv(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)
        self.e4 = DoubleConv(base*4, base*8)
        self.p4 = nn.MaxPool2d(2)
        self.b = DoubleConv(base*8, base*16)
        self.u4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.d4 = DoubleConv(base*16, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.d3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.d2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.d1 = DoubleConv(base*2, base)
        self.head = nn.Conv2d(base, out_channels, 1)
    def forward(self, x):
        a = self.e1(x); b = self.e2(self.p1(a))
        c = self.e3(self.p2(b)); d = self.e4(self.p3(c))
        z = self.b(self.p4(d))
        x = self.d4(torch.cat([self.u4(z), d], 1))
        x = self.d3(torch.cat([self.u3(x), c], 1))
        x = self.d2(torch.cat([self.u2(x), b], 1))
        x = self.d1(torch.cat([self.u1(x), a], 1))
        return self.head(x)
