import csv
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from jointed_train_HingeFM import Config, PVJointModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt = Config()

MODEL_KWARGS = dict(
    use_cbam=opt.use_cbam,
    use_coordatt=opt.use_coordatt,
)


CHECKPOINT_PATH = opt.save_dir / "train" / "best.pth"

INPUT_DIR = opt.data_dir / "test" / "LR"
HR_DIR = opt.data_dir / "test" / "HR"
LABEL_DIR = opt.data_dir / "test" / "label"

OUTPUT_DIR = opt.save_dir / "pred_best"
OUTPUT_VIS_DIR = OUTPUT_DIR / "vis"
OUTPUT_SR_DIR = OUTPUT_DIR / "sr"
OUTPUT_SEG_DIR = OUTPUT_DIR / "seg"
METRIC_CSV = OUTPUT_DIR / "metrics.csv"

OUTPUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_SR_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_SEG_DIR.mkdir(parents=True, exist_ok=True)


class TestDataset(Dataset):
    """Test tiles. HR and labels are optional and only used for per-tile metrics."""
    def __init__(self, lr_dir, hr_dir=None, label_dir=None):
        self.lr_files = sorted(Path(lr_dir).glob("*.tif"))
        if not self.lr_files:
            raise FileNotFoundError(f"No tif files found in {lr_dir}")

        self.hr_files = sorted(Path(hr_dir).glob("*.tif")) if hr_dir and Path(hr_dir).exists() else []
        self.label_files = sorted(Path(label_dir).glob("*.png")) if label_dir and Path(label_dir).exists() else []
        self.has_hr = len(self.hr_files) == len(self.lr_files)
        self.has_label = len(self.label_files) == len(self.lr_files)

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr = self._read_tif(self.lr_files[idx])
        hr = self._read_tif(self.hr_files[idx]) if self.has_hr else np.zeros_like(lr)
        label = self._read_label(self.label_files[idx], lr.shape[1:]) if self.has_label else np.zeros(lr.shape[1:], dtype=np.float32)
        return (
            torch.from_numpy(lr[: opt.num_bands] / 255.0).float(),
            torch.from_numpy(hr[: opt.num_bands] / 255.0).float(),
            torch.from_numpy(label[None, :, :]).float(),
            self.lr_files[idx].name,
        )

    @staticmethod
    def _read_tif(path):
        arr = tifffile.imread(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        return arr

    @staticmethod
    def _read_label(path, image_size):
        label = imageio.imread(path).astype(np.float32)
        if label.ndim == 3:
            label = label[..., 0]
        if label.max() > 1:
            label = label / 255.0
        label = (label > 0).astype(np.float32)
        if label.shape != tuple(image_size):
            raise ValueError(f"Label shape {label.shape} does not match image shape {image_size}: {path}")
        return label


def load_joint_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    model = PVJointModel(**MODEL_KWARGS).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def mse(a, b):
    return float(((a - b) ** 2).mean())


def ssim_metric(a, b):
    a_np = np.clip(a * 255, 0, 255).astype(np.uint8)
    b_np = np.clip(b * 255, 0, 255).astype(np.uint8)
    return float(np.mean([compare_ssim(a_np[c], b_np[c], data_range=255) for c in range(a_np.shape[0])]))


def iou_metric(pred, label):
    pred_bin = (pred > 0.5).astype(np.uint8).flatten()
    label_bin = (label > 0.5).astype(np.uint8).flatten()
    return float(jaccard_score(label_bin, pred_bin, zero_division=0))


def precision_recall_f1(pred, label):
    pred = (pred > 0.5).astype(np.uint8)
    label = (label > 0.5).astype(np.uint8)
    tp = float((pred * label).sum())
    fp = float((pred * (1 - label)).sum())
    fn = float(((1 - pred) * label).sum())
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def to_rgb(img):
    if img.shape[0] >= 3:
        return np.transpose(img[:3], (1, 2, 0))
    return np.transpose(np.tile(img, (3, 1, 1)), (1, 2, 0))


def save_visual(fname, lr_np, sr_np, hr_np, seg_bin, label_np, has_hr, has_label, title):
    """Side-by-side panel: input, reconstruction, reference, prediction, label."""
    h, w = lr_np.shape[1:]
    gap = 5
    images = [to_rgb(lr_np), to_rgb(sr_np)]
    if has_hr:
        images.append(to_rgb(hr_np))
    images.append(np.stack([seg_bin] * 3, axis=-1))
    if has_label:
        images.append(np.stack([label_np] * 3, axis=-1))

    canvas = np.ones((h, len(images) * w + (len(images) - 1) * gap, 3), dtype=np.float32)
    for i, img in enumerate(images):
        canvas[:, i * (w + gap): i * (w + gap) + w] = np.clip(img, 0, 1)

    plt.figure(figsize=(3 * len(images), 4))
    plt.imshow(canvas)
    plt.axis("off")
    plt.title(title)
    plt.savefig(OUTPUT_VIS_DIR / fname.replace(".tif", ".png"), bbox_inches="tight", pad_inches=0.1)
    plt.close()


def inference_and_visualize():
    model = load_joint_model(CHECKPOINT_PATH, DEVICE)
    dataset = TestDataset(INPUT_DIR, HR_DIR, LABEL_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    rows = []
    with torch.no_grad():
        for lr, hr, label, fname in tqdm(loader, total=len(loader), desc="Inference", ncols=100):
            lr = lr.to(DEVICE)
            sr, seg_probs = model(lr)

            lr_np = lr.squeeze(0).cpu().numpy()
            hr_np = hr.squeeze(0).numpy()
            sr_np = sr.squeeze(0).cpu().numpy()
            label_np = label[0, 0].numpy()
            seg_np = seg_probs[0, 0].cpu().numpy()
            seg_bin = (seg_np > 0.5).astype(np.float32)

            sr_save = np.clip(sr_np * 255, 0, 255).astype(np.uint8)
            tifffile.imwrite(OUTPUT_SR_DIR / fname[0], np.transpose(sr_save, (1, 2, 0)))
            imageio.imwrite(OUTPUT_SEG_DIR / fname[0].replace(".tif", ".png"), (seg_bin * 255).astype(np.uint8))

            row = {"file": fname[0]}
            title_parts = []
            if dataset.has_hr:
                row["mse"] = mse(sr_np, hr_np)
                row["ssim"] = ssim_metric(sr_np, hr_np)
                title_parts += [f"MSE:{row['mse']:.4f}", f"SSIM:{row['ssim']:.4f}"]
            if dataset.has_label:
                row["iou"] = iou_metric(seg_bin, label_np)
                row["precision"], row["recall"], row["f1"] = precision_recall_f1(seg_bin, label_np)
                title_parts += [
                    f"IoU:{row['iou']:.4f}",
                    f"P:{row['precision']:.4f}",
                    f"R:{row['recall']:.4f}",
                    f"F1:{row['f1']:.4f}",
                ]
            rows.append(row)

            title = " | ".join(title_parts) if title_parts else fname[0]
            tqdm.write(f"{fname[0]} | {title}")
            save_visual(fname[0], lr_np, sr_np, hr_np, seg_bin, label_np, dataset.has_hr, dataset.has_label, title)

    # per-tile metrics; use Global_IOU.py for the dataset-level scores reported in the paper
    if rows:
        keys = ["file", "mse", "ssim", "iou", "precision", "recall", "f1"]
        with open(METRIC_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved metrics to {METRIC_CSV}")


if __name__ == "__main__":
    inference_and_visualize()
