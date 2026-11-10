import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(concat)
        return self.sigmoid(att) * x


class CBAM(nn.Module):
    """Channel attention followed by spatial attention."""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DoubleConv(nn.Module):
    """(Conv + BN + ReLU) x 2."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DFeatureExtract(nn.Module):
    """
    Shared encoder E.

    Returns h1 (64 ch, 1/1), h2 (128, 1/2), h3 (256, 1/4), h4 (512, 1/8) and h5 (512, 1/16).
    h1-h3 feed the reconstruction branch G_R, h1-h5 feed the segmentation branch G_S.
    """
    def __init__(self, in_channels=3, base_filters=64, use_cbam=True):
        super(DFeatureExtract, self).__init__()
        self.use_cbam = use_cbam

        self.conv1 = DoubleConv(in_channels, base_filters)
        self.cbam1 = CBAM(base_filters) if use_cbam else nn.Identity()

        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base_filters, base_filters * 2)
        self.cbam2 = CBAM(base_filters * 2) if use_cbam else nn.Identity()

        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base_filters * 2, base_filters * 4)
        self.cbam3 = CBAM(base_filters * 4) if use_cbam else nn.Identity()

        self.maxpool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(base_filters * 4, base_filters * 8)
        self.cbam4 = CBAM(base_filters * 8) if use_cbam else nn.Identity()

        # bottleneck without CBAM, ASPP is applied in G_S
        self.maxpool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(base_filters * 8, base_filters * 8)

    def forward(self, x):
        h1 = self.cbam1(self.conv1(x))

        x = self.maxpool1(h1)
        h2 = self.cbam2(self.conv2(x))

        x = self.maxpool2(h2)
        h3 = self.cbam3(self.conv3(x))

        x = self.maxpool3(h3)
        h4 = self.cbam4(self.conv4(x))

        x = self.maxpool4(h4)
        h5 = self.conv5(x)

        return h1, h2, h3, h4, h5
