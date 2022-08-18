import math
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, stride=1, padding=None, groups=1, act=True, bias=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, stride, autopad(k, padding), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, kernel_size=1, padding=None,  stride=1, act=True, bias=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, kernel_size, stride, padding=padding, groups=math.gcd(c1, c2), act=act, bias=bias)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution class
    def __init__(self, c1, c2, k=1, stride=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, stride, p1, p2, groups=math.gcd(c1, c2))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias, padding_mode=padding_mode)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, groups=1, bias=bias, padding_mode=padding_mode)
        self.device = device
        self.dtype = dtype

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
