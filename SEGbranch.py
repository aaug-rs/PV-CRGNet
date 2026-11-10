import torch
import torch.nn as nn
import torch.nn.functional as F


def get_group_norm(channels, max_groups=8):
    """GroupNorm with the largest group count (<= max_groups) that divides channels."""
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvGNAct(nn.Module):
    """Conv + GroupNorm + ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            get_group_norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_channels, out_channels, 3),
            ConvGNAct(out_channels, out_channels, 3)
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=(1, 6, 12, 18)):
        super().__init__()

        self.branches = nn.ModuleList()
        for rate in atrous_rates:
            if rate == 1:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                        get_group_norm(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, out_channels,
                            kernel_size=3, padding=rate, dilation=rate, bias=False
                        ),
                        get_group_norm(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            get_group_norm(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1, bias=False),
            get_group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]
        outs = [branch(x) for branch in self.branches]

        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)
        outs.append(gp)

        x = torch.cat(outs, dim=1)
        return self.project(x)


class FullScaleFusionBlock(nn.Module):
    """UNet3+-style full-scale fusion: project each input, resize to ref_feat and fuse."""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                get_group_norm(out_channels),
                nn.ReLU(inplace=True)
            )
            for c in in_channels_list
        ])

        self.fuse = nn.Sequential(
            DoubleConv(out_channels * len(in_channels_list), out_channels),
            nn.Dropout2d(0.1)
        )

    def forward(self, feats, ref_feat):
        assert len(feats) == len(self.proj), "Number of input features does not match in_channels_list"
        target_size = ref_feat.shape[2:]
        outs = []
        for x, p in zip(feats, self.proj):
            x = p(x)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        return self.fuse(x)


class PVSegmentationHeadUNet3Plus(nn.Module):
    """
    Segmentation branch G_S.

    Inputs: encoder features h1 (1/1) to h5 (1/16).
    Returns main logits [B, num_classes, H, W] and auxiliary logits {d4, d3, d2}
    for deep supervision, all at the input resolution.
    """
    def __init__(self, num_classes=1, decoder_channels=64, h1_channels=64, h2_channels=128,
                 h3_channels=256, h4_channels=512, h5_channels=512, use_aspp=True):
        super().__init__()

        self.use_aspp = use_aspp

        if use_aspp:
            self.bottleneck = ASPP(h5_channels, decoder_channels)
            bottleneck_out_channels = decoder_channels
        else:
            self.bottleneck = nn.Sequential(
                DoubleConv(h5_channels, decoder_channels),
                nn.Dropout2d(0.1)
            )
            bottleneck_out_channels = decoder_channels

        # d4 (1/8): h1-h5
        self.fuse_d4 = FullScaleFusionBlock(
            [h1_channels, h2_channels, h3_channels, h4_channels, bottleneck_out_channels],
            decoder_channels
        )

        # d3 (1/4): h1-h5, d4
        self.fuse_d3 = FullScaleFusionBlock(
            [h1_channels, h2_channels, h3_channels, h4_channels, bottleneck_out_channels, decoder_channels],
            decoder_channels
        )

        # d2 (1/2): h1-h5, d4, d3
        self.fuse_d2 = FullScaleFusionBlock(
            [h1_channels, h2_channels, h3_channels, h4_channels, bottleneck_out_channels, decoder_channels, decoder_channels],
            decoder_channels
        )

        # d1 (1/1): h1-h5, d4, d3, d2
        self.fuse_d1 = FullScaleFusionBlock(
            [h1_channels, h2_channels, h3_channels, h4_channels, bottleneck_out_channels, decoder_channels, decoder_channels, decoder_channels],
            decoder_channels
        )

        self.final_fuse = nn.Sequential(
            DoubleConv(decoder_channels, decoder_channels),
            nn.Dropout2d(0.1)
        )

        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

        # deep-supervision heads
        self.aux_d4 = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        self.aux_d3 = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        self.aux_d2 = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(self, h1, h2, h3, h4, h5):
        h5b = self.bottleneck(h5)

        d4 = self.fuse_d4([h1, h2, h3, h4, h5b], ref_feat=h4)

        d4_up = F.interpolate(d4, size=h3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.fuse_d3([h1, h2, h3, h4, h5b, d4_up], ref_feat=h3)

        d4_up2 = F.interpolate(d4, size=h2.shape[2:], mode='bilinear', align_corners=False)
        d3_up = F.interpolate(d3, size=h2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.fuse_d2([h1, h2, h3, h4, h5b, d4_up2, d3_up], ref_feat=h2)

        d4_up3 = F.interpolate(d4, size=h1.shape[2:], mode='bilinear', align_corners=False)
        d3_up2 = F.interpolate(d3, size=h1.shape[2:], mode='bilinear', align_corners=False)
        d2_up = F.interpolate(d2, size=h1.shape[2:], mode='bilinear', align_corners=False)

        d1 = self.fuse_d1(
            [h1, h2, h3, h4, h5b, d4_up3, d3_up2, d2_up],
            ref_feat=h1
        )

        out = self.final_fuse(d1)
        main_logits = self.classifier(out)

        aux_logits = {
            "d4": self.aux_d4(d4),
            "d3": self.aux_d3(d3),
            "d2": self.aux_d2(d2)
        }

        if main_logits.shape[2:] != h1.shape[2:]:
            main_logits = F.interpolate(main_logits, size=h1.shape[2:], mode='bilinear', align_corners=False)

        for k in aux_logits:
            aux_logits[k] = F.interpolate(aux_logits[k], size=h1.shape[2:], mode='bilinear', align_corners=False)

        return main_logits, aux_logits  # raw logits, no sigmoid
