# PV-CRGNet

Official PyTorch implementation of

**PV-CRGNet: A Cross-Resolution Guided Network for Distributed Rooftop Photovoltaic Extraction from Jilin-1 Imagery**
IEEE Geoscience and Remote Sensing Letters (under review)

PV-CRGNet extracts rooftop photovoltaic (PV) panels from 0.5 m Jilin-1 imagery. During training, a reconstruction branch is guided by paired higher-resolution reference imagery, and a PV-aware conditional discriminator receives the PV mask and its boundary prior. At inference time only the Jilin-1 image is required.

## Code structure

| File | Content | Symbol in the paper |
|---|---|---|
| `DFeatureExtract.py` | Shared encoder (DoubleConv + CBAM), outputs h1–h5 | E |
| `SRbranch_HingeFM.py` | `TextureDecoder`: reconstruction branch (feature alignment, CoordAtt, RCAB × 8) | G_R |
| | `PatchGANDiscriminator`: SN-PatchGAN discriminator | D |
| `SEGbranch.py` | UNet3+-style full-scale segmentation branch with ASPP and deep supervision | G_S |
| `jointed_train_HingeFM.py` | Joint model, losses and training loop | |
| `pred_HingeFM.py` | Inference on the test set | |
| `Global_IOU.py` | Dataset-level IoU / Precision / Recall / F1 from pooled TP, FP, FN | |

## Requirements

Python ≥ 3.10 and a CUDA-capable GPU are recommended.

```
pip install -r requirements.txt
```

## Data

Place the data as follows (file names of LR, HR and label tiles are paired in sorted order):

```
data/
├── train/  LR/*.tif   HR/*.tif   label/*.png
├── val/    LR/*.tif   HR/*.tif   label/*.png
└── test/   LR/*.tif   HR/*.tif   label/*.png
```

- `LR`: 0.5 m Jilin-1 RGB tiles, 512 × 512, input image I_L
- `HR`: higher-resolution reference tiles covering the same extent on the same 512 × 512 grid, reference image I_H, used only for training
- `label`: binary PV masks, 0 = background and non-zero = PV

**Data availability.** The Jilin-1 and Google satellite imagery used in this study are subject to the providers' licensing terms and cannot be redistributed. The construction of the dataset is described in Section III-A of the paper. The PV annotations are available from the corresponding author upon reasonable request.

## Training

Edit `Config` in `jointed_train_HingeFM.py` if needed, then run

```
python jointed_train_HingeFM.py
```

Checkpoints and `history.csv` are written to `runs/pv_crgnet/train/`. `best.pth` is selected by validation IoU.

The variants with a reconstruction branch in Table II are obtained by changing two switches in `Config`:

| Variant | `use_gan` | `use_pv_aware_discriminator` |
|---|---|---|
| Joint Rec.-Seg. w/o D | `False` | – |
| Joint Rec.-Seg. | `True` | `False` |
| PV-CRGNet | `True` | `True` |

Seg. Baseline in Table II consists of the shared encoder and the segmentation branch only.

## Inference and evaluation

After training, run

```
python pred_HingeFM.py              # writes runs/pv_crgnet/pred_best/{seg,sr,vis}
python Global_IOU.py                # IoU, Precision, Recall, F1
```

## Citation

```
@article{pvcrgnet,
  title   = {PV-CRGNet: A Cross-Resolution Guided Network for Distributed Rooftop Photovoltaic Extraction from Jilin-1 Imagery},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026}
}
```
