import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate attention (Hou et al., CVPR 2021)."""
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class CALayer(nn.Module):
    """Channel attention used in RCAB."""
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """Residual channel attention block."""
    def __init__(self, channels, reduction=16):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            CALayer(channels, reduction)
        )

    def forward(self, x):
        return x + self.body(x)


class TextureDecoder(nn.Module):
    """
    Reconstruction branch G_R.

    Reconstructs the reference image I_H from encoder features h1, h2 and h3.
    Output size is scale x the input size; scale = 1 in this work.
    """
    def __init__(self, scale=4, in_ch_h1=64, in_ch_h2=128,  in_ch_h3=256, base_ch=64, num_rcab=8, out_channels=4, use_coordatt=True):
        super().__init__()
        self.scale = scale
        self.use_coordatt = use_coordatt

        # feature alignment to the h1 resolution
        self.align_h2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch_h2, base_ch, 1)
        )
        self.align_h3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch_h3, base_ch, 1)
        )
        self.align_h1 = nn.Conv2d(in_ch_h1, base_ch, 1)

        self.fusion = nn.Conv2d(base_ch * 3, base_ch, 3, padding=1)
        self.coord_att = CoordAtt(base_ch, base_ch) if use_coordatt else nn.Identity()

        self.recon_trunk = nn.Sequential(*[RCAB(base_ch) for _ in range(num_rcab)])
        self.trunk_conv = nn.Conv2d(base_ch, base_ch, 3, padding=1)

        # pixel-shuffle upsampling, identity when scale = 1
        self.upsample = self._make_espcns(base_ch, scale)

        self.img_recon = nn.Sequential(
            nn.Conv2d(base_ch, out_channels, 3, padding=1),
            nn.Sigmoid()  # output in [0, 1]
        )

    def _make_espcns(self, in_ch, scale):
        """x2 pixel-shuffle stages, scale must be a power of 2."""
        layers = []
        cur_ch = in_ch
        cur_scale = scale
        while cur_scale > 1:
            out_ch = cur_ch
            layers.append(nn.Conv2d(cur_ch, cur_ch * 4, 3, padding=1))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.ReLU(inplace=True))
            cur_scale //= 2
        return nn.Sequential(*layers)

    def forward(self, h1, h2, h3):
        h1_aligned = self.align_h1(h1)
        h2_up = self.align_h2(h2)
        h3_up = self.align_h3(h3)

        fused = self.fusion(torch.cat([h1_aligned, h2_up, h3_up], dim=1))
        fused = self.coord_att(fused)

        recon = self.recon_trunk(fused)
        sr_feat = fused + self.trunk_conv(recon)

        up_feat = self.upsample(sr_feat)
        sr_img = self.img_recon(up_feat)

        return sr_img, sr_feat


class PatchGANDiscriminator(nn.Module):
    """
    SN-PatchGAN discriminator D.

    Outputs patch logits without sigmoid for the hinge loss, uses spectral
    normalization and no BatchNorm, and can return intermediate features
    for the feature-matching loss.
    """
    def __init__(self, in_channels=3, ndf=64, n_layers=3, use_spectral_norm=True):
        super(PatchGANDiscriminator, self).__init__()
        self.use_spectral_norm = use_spectral_norm

        def conv(in_c, out_c, kernel_size=4, stride=2, padding=1):
            layer = nn.Conv2d(in_c, out_c, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=True)
            if use_spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            return layer

        self.blocks = nn.ModuleList()
        nf_mult = 1

        self.blocks.append(nn.Sequential(
            conv(in_channels, ndf, stride=2),
            nn.LeakyReLU(0.2, inplace=True)
        ))

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.blocks.append(nn.Sequential(
                conv(ndf * nf_mult_prev, ndf * nf_mult, stride=2),
                nn.LeakyReLU(0.2, inplace=True)
            ))

        # stride-1 block before the final logits
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.blocks.append(nn.Sequential(
            conv(ndf * nf_mult_prev, ndf * nf_mult, stride=1),
            nn.LeakyReLU(0.2, inplace=True)
        ))

        self.final = conv(ndf * nf_mult, 1, stride=1)

    def forward(self, x, return_features=False):
        features = []
        for block in self.blocks:
            x = block(x)
            if return_features:
                features.append(x)
        logits = self.final(x)
        if return_features:
            return logits, features
        return logits
