import logging
import rasterio
rasterio.log.setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

import shutil
from pathlib import Path
from tqdm import tqdm
from math import exp

from DFeatureExtract import DFeatureExtract
from SRbranch_HingeFM import TextureDecoder, PatchGANDiscriminator
from SEGbranch import PVSegmentationHeadUNet3Plus
from utils import get_logger, AverageMeter, save_checkpoint, log_csv


# ==============================================================================
# Configuration
# ==============================================================================
class Config:
    lr = 1e-4
    d_lr = 5e-5
    num_bands = 3
    batch_size = 4
    epochs = 200
    cuda = True
    num_workers = 4

    save_dir = Path(__file__).resolve().parent / 'runs' / 'pv_crgnet'
    data_dir = Path(__file__).resolve().parent / 'data'  # data/{train,val,test}/{LR,HR,label}
    image_size = [512, 512]
    resume = False

    use_cbam = True
    use_coordatt = True

    # hinge GAN with spectral normalization and feature matching
    use_gan = True                      # False: Joint Rec.-Seg. w/o D
    use_feature_matching = True
    use_spectral_norm = True
    gan_start_epoch = 20
    gan_ramp_epochs = 30
    lambda_adv_max = 0.02

    use_pv_aware_discriminator = True   # True: D([I; y; b]), False: D(I), i.e. Joint Rec.-Seg.

    lambda_fm = 0.05
    d_steps = 1
    grad_clip_g = 1.0
    grad_clip_d = 1.0


opt = Config()

torch.manual_seed(2020)
if opt.cuda and torch.cuda.is_available():
    torch.cuda.manual_seed_all(2020)


# ==============================================================================
# Dataset
# ==============================================================================
class PatchDataset(Dataset):
    """Paired I_L (LR), I_H (HR) and PV mask tiles, matched by sorted file order."""
    def __init__(self, lr_folder, hr_folder, label_folder):
        self.lr_files = sorted(list(Path(lr_folder).rglob('*.tif')))
        self.hr_files = sorted(list(Path(hr_folder).rglob('*.tif')))
        self.label_files = sorted(list(Path(label_folder).rglob('*.png')))
        assert len(self.lr_files) == len(self.hr_files) == len(self.label_files), \
            "Numbers of LR, HR and label tiles do not match"

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr = rasterio.open(self.lr_files[idx]).read().astype('float32') / 255
        hr = rasterio.open(self.hr_files[idx]).read().astype('float32') / 255
        label = rasterio.open(self.label_files[idx]).read(1).astype('float32')
        lr = lr[:3, :, :]
        hr = hr[:3, :, :]
        label[label > 0] = 1.0
        label = label[None, :, :]
        return torch.tensor(lr), torch.tensor(hr), torch.tensor(label)


# ==============================================================================
# Model
# ==============================================================================
class PVJointModel(nn.Module):
    def __init__(self, use_cbam=True, use_coordatt=True):
        super(PVJointModel, self).__init__()
        self.shared_encoder = DFeatureExtract(
            in_channels=opt.num_bands, base_filters=64, use_cbam=use_cbam
        )
        # scale=1: I_L and I_H tiles share the same 512 x 512 grid
        self.texture_decoder = TextureDecoder(
            scale=1, in_ch_h1=64, in_ch_h2=128, in_ch_h3=256,
            out_channels=opt.num_bands, use_coordatt=use_coordatt
        )
        self.seg_head = PVSegmentationHeadUNet3Plus(num_classes=1, decoder_channels=64)
        discriminator_in_channels = opt.num_bands + 2 if opt.use_pv_aware_discriminator else opt.num_bands
        self.discriminator = PatchGANDiscriminator(
            in_channels=discriminator_in_channels,
            use_spectral_norm=opt.use_spectral_norm
        )

    def forward(self, lr_img):
        """Inference path: returns the reconstruction and the PV probability map."""
        h1, h2, h3, h4, h5 = self.shared_encoder(lr_img)
        sr_img, _ = self.texture_decoder(h1, h2, h3)
        main_seg_logits, _ = self.seg_head(h1, h2, h3, h4, h5)
        return sr_img, torch.sigmoid(main_seg_logits)


# ==============================================================================
# Losses
# ==============================================================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val
    else:
        L = val_range
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

class MSSSIMLoss(nn.Module):
    """1 - MS-SSIM."""
    def __init__(self, normalize=True):
        super(MSSSIMLoss, self).__init__()
        self.normalize = normalize

    def forward(self, x, y):
        # float32 avoids NaN from variance subtraction under AMP
        return 1.0 - msssim(x.float(), y.float(), normalize=self.normalize)


class FocalTverskyLoss(nn.Module):
    """alpha weights FN, beta weights FP."""
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        target = target.view(-1)

        TP = (probs * target).sum()
        FP = ((1 - target) * probs).sum()
        FN = (target * (1 - probs)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        focal_tversky = torch.clamp(1 - tversky, min=1e-6) ** self.gamma
        return focal_tversky

class PVJointSegLoss(nn.Module):
    """Segmentation loss: 0.5 * BCE + Focal Tversky."""
    def __init__(self, weight_bce=0.5, weight_ft=1.0):
        super(PVJointSegLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ft = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
        self.weight_bce = weight_bce
        self.weight_ft = weight_ft

    def forward(self, logits, target):
        loss_bce = self.bce(logits, target)
        loss_ft = self.ft(logits, target)
        return self.weight_bce * loss_bce + self.weight_ft * loss_ft


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        # float32 keeps eps ** 2 from underflowing under AMP
        x = x.float()
        y = y.float()
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

class VGGPerceptualLoss(nn.Module):
    """MSE between shallow VGG16 features, inputs in [0, 1]."""
    def __init__(self, feature_layer=2):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.loss_network = nn.Sequential(*list(vgg.children())[:feature_layer+1]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        sr = sr[:, :3, :, :]
        hr = hr[:, :3, :, :]
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        sr_feat = self.loss_network(sr)
        hr_feat = self.loss_network(hr)
        return F.mse_loss(sr_feat, hr_feat)


class GradientLoss(nn.Module):
    """L1 difference of horizontal and vertical image gradients."""
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, gen_img, gt_img):
        gen_rgb = gen_img[:, :3, :, :]
        gt_rgb = gt_img[:, :3, :, :]

        grad_gen_x = torch.abs(gen_rgb[:, :, :, 1:] - gen_rgb[:, :, :, :-1])
        grad_gen_y = torch.abs(gen_rgb[:, :, 1:, :] - gen_rgb[:, :, :-1, :])
        grad_gt_x = torch.abs(gt_rgb[:, :, :, 1:] - gt_rgb[:, :, :, :-1])
        grad_gt_y = torch.abs(gt_rgb[:, :, 1:, :] - gt_rgb[:, :, :-1, :])

        loss = torch.mean(torch.abs(grad_gen_x - grad_gt_x)) + \
               torch.mean(torch.abs(grad_gen_y - grad_gt_y))
        return loss

class CosineSimilarityLoss(nn.Module):
    """1 - cosine similarity between RGB vectors."""
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, gen_img, gt_img):
        gen_rgb = gen_img[:, :3, :, :].float()
        gt_rgb = gt_img[:, :3, :, :].float()
        cos_sim = F.cosine_similarity(gen_rgb, gt_rgb, dim=1, eps=1e-8)
        return 1.0 - torch.mean(cos_sim)


# ==============================================================================
# Training
# ==============================================================================
class Experiment:
    def __init__(self, option):
        self.device = torch.device('cuda' if torch.cuda.is_available() and option.cuda else 'cpu')
        self.opt = option
        self.save_dir = option.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'
        self.best = self.train_dir / 'best.pth'
        self.last_model = self.train_dir / 'joint_model.pth'

        self.logger = get_logger()
        self.logger.info('Initializing models...')

        self.model = PVJointModel(
            use_cbam=self.opt.use_cbam,
            use_coordatt=self.opt.use_coordatt
        ).to(self.device)

        self.criterion_char = CharbonnierLoss().to(self.device)
        self.criterion_percep = VGGPerceptualLoss().to(self.device)
        self.criterion_msssim = MSSSIMLoss().to(self.device)
        self.criterion_seg = PVJointSegLoss().to(self.device)
        self.criterion_grad = GradientLoss().to(self.device)
        self.criterion_cos = CosineSimilarityLoss().to(self.device)

        params_G = (
            list(self.model.shared_encoder.parameters()) +
            list(self.model.texture_decoder.parameters()) +
            list(self.model.seg_head.parameters())
        )
        self.g_optimizer = optim.AdamW(
            params_G, lr=option.lr, betas=(0.9, 0.99), weight_decay=1e-4
        )
        self.d_optimizer = optim.AdamW(
            self.model.discriminator.parameters(),
            lr=option.d_lr, betas=(0.0, 0.9), weight_decay=0.0
        )

        self.scaler = GradScaler()
        self.start_epoch = 0

        if self.opt.resume:
            self.load_checkpoint()

    def load_checkpoint(self):
        """Resume from the latest checkpoint (joint_model.pth)."""
        if self.last_model.exists():
            self.logger.info(f"==> Loading checkpoint '{self.last_model}'")
            checkpoint = torch.load(self.last_model, map_location=self.device)

            self.model.load_state_dict(checkpoint['state_dict'])

            if 'g_optimizer' in checkpoint:
                self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            if 'd_optimizer' in checkpoint:
                self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])

            if 'scaler' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1

            self.logger.info(f"==> Resuming training from epoch {self.start_epoch}")
        else:
            self.logger.info(f"==> No checkpoint found at '{self.last_model}', starting from scratch.")

    # ---------------- adversarial helpers ----------------
    @staticmethod
    def make_pv_boundary(label):
        """Boundary prior b = dilate(y) - erode(y) with a 3 x 3 kernel."""
        dilated = F.max_pool2d(label.float(), kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-label.float(), kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def build_discriminator_input(self, image, label):
        """Returns [image; y; b] for the PV-aware D, or the image alone."""
        if not self.opt.use_pv_aware_discriminator:
            return image
        mask = label.float()
        if mask.shape[-2:] != image.shape[-2:]:
            mask = F.interpolate(mask, size=image.shape[-2:], mode='nearest')
        boundary = self.make_pv_boundary(mask)
        return torch.cat([image, mask.detach(), boundary.detach()], dim=1)

    @staticmethod
    def hinge_d_loss(real_logits, fake_logits):
        return 0.5 * (F.relu(1.0 - real_logits).mean() +
                      F.relu(1.0 + fake_logits).mean())

    @staticmethod
    def hinge_g_loss(fake_logits):
        return (-fake_logits).mean()

    @staticmethod
    def feature_matching_loss(real_features, fake_features):
        if len(real_features) != len(fake_features):
            raise RuntimeError('Discriminator feature lists have different lengths.')
        total = 0.0
        for real_f, fake_f in zip(real_features, fake_features):
            total = total + F.l1_loss(fake_f, real_f.detach())
        return total / max(1, len(real_features))

    def get_gan_weight(self, epoch):
        """lambda_adv: 0 before gan_start_epoch, then linear ramp to lambda_adv_max."""
        if (not self.opt.use_gan) or epoch < self.opt.gan_start_epoch:
            return 0.0
        progress = (epoch - self.opt.gan_start_epoch + 1) / max(1, self.opt.gan_ramp_epochs)
        progress = min(1.0, max(0.0, progress))
        return self.opt.lambda_adv_max * progress

    def train_on_epoch(self, epoch, data_loader):
        self.model.train()
        epg_loss = AverageMeter(); eppd_loss = AverageMeter()
        epseg_loss = AverageMeter(); epg_error = AverageMeter()
        ep_adv = AverageMeter(); ep_fm = AverageMeter()

        lambda_seg = min(1.0, max(0.0, (epoch - 10) / 40.0))
        lambda_adv = self.get_gan_weight(epoch)

        pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                    desc=f"Epoch {epoch}", ncols=120)

        for idx, data in pbar:
            lr, hr, label = [d.to(self.device, non_blocking=True) for d in data]

            # 1) discriminator
            d_loss = torch.tensor(0.0, device=self.device)
            if self.opt.use_gan and lambda_adv > 0:
                for p in self.model.discriminator.parameters():
                    p.requires_grad_(True)
                for _ in range(self.opt.d_steps):
                    self.d_optimizer.zero_grad(set_to_none=True)
                    with autocast():
                        with torch.no_grad():
                            h1d, h2d, h3d, _, _ = self.model.shared_encoder(lr)
                            fake_d, _ = self.model.texture_decoder(h1d, h2d, h3d)
                        real_input = self.build_discriminator_input(hr, label)
                        fake_input = self.build_discriminator_input(fake_d.detach(), label)
                        real_logits = self.model.discriminator(real_input)
                        fake_logits = self.model.discriminator(fake_input)
                        d_loss = self.hinge_d_loss(real_logits, fake_logits)
                    self.scaler.scale(d_loss).backward()
                    self.scaler.unscale_(self.d_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.discriminator.parameters(), self.opt.grad_clip_d)
                    self.scaler.step(self.d_optimizer)
            else:
                for p in self.model.discriminator.parameters():
                    p.requires_grad_(False)

            # 2) encoder, reconstruction and segmentation branches
            for p in self.model.discriminator.parameters():
                p.requires_grad_(False)
            self.g_optimizer.zero_grad(set_to_none=True)

            with autocast():
                h1, h2, h3, h4, h5 = self.model.shared_encoder(lr)
                sr_img, _ = self.model.texture_decoder(h1, h2, h3)
                main_logits, aux_logits = self.model.seg_head(h1, h2, h3, h4, h5)

                loss_char = self.criterion_char(sr_img, hr)
                loss_perc = self.criterion_percep(sr_img, hr)
                loss_msssim = self.criterion_msssim(sr_img, hr)
                loss_grad = self.criterion_grad(sr_img, hr)
                loss_cos = self.criterion_cos(sr_img, hr)

                loss_adv = torch.tensor(0.0, device=self.device)
                loss_fm = torch.tensor(0.0, device=self.device)
                if self.opt.use_gan and lambda_adv > 0:
                    fake_input = self.build_discriminator_input(sr_img, label)
                    fake_logits, fake_features = self.model.discriminator(
                        fake_input, return_features=True)
                    loss_adv = self.hinge_g_loss(fake_logits)
                    if self.opt.use_feature_matching:
                        with torch.no_grad():
                            real_input = self.build_discriminator_input(hr, label)
                            _, real_features = self.model.discriminator(
                                real_input, return_features=True)
                        loss_fm = self.feature_matching_loss(
                            real_features, fake_features)

                # lambda_FM follows the lambda_adv ramp
                loss_sr_total = (
                    1.0 * loss_char +
                    0.30 * loss_msssim +
                    0.15 * loss_grad +
                    0.10 * loss_cos +
                    0.05 * loss_perc +
                    lambda_adv * loss_adv +
                    (self.opt.lambda_fm * (lambda_adv / max(self.opt.lambda_adv_max, 1e-8))) * loss_fm
                )

                if lambda_seg > 0:
                    loss_seg_main = self.criterion_seg(main_logits, label)
                    loss_seg_aux = (
                        self.criterion_seg(aux_logits['d4'], label) * 0.2 +
                        self.criterion_seg(aux_logits['d3'], label) * 0.3 +
                        self.criterion_seg(aux_logits['d2'], label) * 0.4)
                    seg_loss = loss_seg_main + loss_seg_aux
                else:
                    seg_loss = torch.tensor(0.0, device=self.device)

                g_total = loss_sr_total + lambda_seg * seg_loss

            self.scaler.scale(g_total).backward()
            self.scaler.unscale_(self.g_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.shared_encoder.parameters()) +
                list(self.model.texture_decoder.parameters()) +
                list(self.model.seg_head.parameters()),
                max_norm=self.opt.grad_clip_g)
            self.scaler.step(self.g_optimizer)
            self.scaler.update()

            mse = F.mse_loss(sr_img.detach(), hr).item()
            epg_loss.update(g_total.item()); eppd_loss.update(d_loss.item())
            epseg_loss.update(seg_loss.item() if lambda_seg > 0 else 0.0)
            epg_error.update(mse); ep_adv.update(loss_adv.item()); ep_fm.update(loss_fm.item())
            pbar.set_postfix(G=f'{g_total.item():.4f}', D=f'{d_loss.item():.4f}',
                             Seg=f'{seg_loss.item() if lambda_seg > 0 else 0.0:.4f}',
                             Adv=f'{loss_adv.item():.4f}', FM=f'{loss_fm.item():.4f}',
                             MSE=f'{mse:.5f}')

        self.logger.info(
            f"Epoch[{epoch}] - G:{epg_loss.avg:.4f} D:{eppd_loss.avg:.4f} "
            f"Seg:{epseg_loss.avg:.4f} Adv:{ep_adv.avg:.4f} FM:{ep_fm.avg:.4f} "
            f"MSE:{epg_error.avg:.6f} lambda_adv:{lambda_adv:.5f}"
        )
        return epg_loss.avg, eppd_loss.avg, epseg_loss.avg, epg_error.avg, ep_adv.avg, ep_fm.avg, lambda_adv

    @torch.no_grad()
    def test_on_epoch(self, data_loader):
        """Validation: reconstruction MSE and dataset-level IoU, precision, recall, F1."""
        self.model.eval()
        epoch_error = AverageMeter()
        inter = union = tp = fp = fn = 0.0
        for lr, hr, label in data_loader:
            lr, hr, label = lr.to(self.device), hr.to(self.device), label.to(self.device)
            sr_img, seg_probs = self.model(lr)
            epoch_error.update(F.mse_loss(sr_img, hr).item())
            pred = (seg_probs >= 0.5).float()
            inter += (pred * label).sum().item()
            union += ((pred + label) > 0).float().sum().item()
            tp += (pred * label).sum().item()
            fp += (pred * (1 - label)).sum().item()
            fn += ((1 - pred) * label).sum().item()
        iou = inter / max(union, 1e-8)
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return epoch_error.avg, iou, precision, recall, f1

    def train(self, train_loader, val_loader, epochs=200):
        best_iou = -float('inf')
        best_mse = float('inf')
        for epoch in range(self.start_epoch, epochs):
            (g_loss, d_loss, seg_loss, g_error, adv_loss, fm_loss, lambda_adv) = \
                self.train_on_epoch(epoch, train_loader)
            val_mse, val_iou, val_precision, val_recall, val_f1 = self.test_on_epoch(val_loader)

            log_csv(
                self.history,
                [epoch, g_loss, d_loss, seg_loss, g_error, val_mse, val_iou,
                 val_precision, val_recall, val_f1, adv_loss, fm_loss, lambda_adv],
                header=['epoch', 'g_loss', 'd_loss', 'seg_loss', 'train_mse',
                        'val_mse', 'val_iou', 'val_precision', 'val_recall',
                        'val_f1', 'adv_loss', 'fm_loss', 'lambda_adv'])

            # best.pth is selected by validation IoU
            save_checkpoint(self.model, self.g_optimizer, self.d_optimizer,
                            self.scaler, epoch, self.last_model)
            if val_iou > best_iou:
                best_iou = val_iou
                shutil.copy(str(self.last_model), str(self.best))
                self.logger.info(f"New best segmentation model: IoU={val_iou:.4f}, F1={val_f1:.4f}")
            if val_mse < best_mse:
                best_mse = val_mse
                shutil.copy(str(self.last_model), str(self.train_dir / 'best_reconstruction.pth'))

            if (epoch + 1) % 5 == 0:
                hist_model = self.train_dir / f'joint_model_epoch{epoch + 1}.pth'
                save_checkpoint(self.model, self.g_optimizer, self.d_optimizer,
                                self.scaler, epoch, hist_model)
                print(f"Saved checkpoint: epoch {epoch + 1}")
            print(
                f"\nEpoch {epoch}: G={g_loss:.4f} D={d_loss:.4f} Seg={seg_loss:.4f} "
                f"Adv={adv_loss:.4f} FM={fm_loss:.4f} lambda_adv={lambda_adv:.5f} | "
                f"Val MSE={val_mse:.6f} IoU={val_iou:.4f} P={val_precision:.4f} "
                f"R={val_recall:.4f} F1={val_f1:.4f}\n")


if __name__ == '__main__':
    train_dataset = PatchDataset(
        opt.data_dir / 'train' / 'LR',
        opt.data_dir / 'train' / 'HR',
        opt.data_dir / 'train' / 'label'
    )

    val_dataset = PatchDataset(
        opt.data_dir / 'val' / 'LR',
        opt.data_dir / 'val' / 'HR',
        opt.data_dir / 'val' / 'label'
    )

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    experiment = Experiment(opt)
    experiment.train(train_loader, val_loader, epochs=opt.epochs)
