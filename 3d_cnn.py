"""
run_adni.py
===========
ONE-FILE pipeline for the AD/MCI classification project. Replaces
models.py / data_utils.py / train.py / run_real_data.py / run_real_pipeline.py
with a single script you run directly on your real ADNI data.

What it does, end to end:
    1. Loads AD / NC / pMCI / sMCI .nii.gz scans.
    2. Trains a 3D ICAE (Inception Convolutional Autoencoder) unsupervised on AD+NC.
    3. Fine-tunes an AD-vs-NC classifier from that pretrained encoder.
    4. Transfers those encoder weights and fine-tunes a pMCI-vs-sMCI classifier.
    5. Saves both trained models as .pt files.
    6. (Optional) Generates a gradient-based saliency map showing which brain
       regions the AD-vs-NC model relies on.

Reference: Oh, K. et al. (2019). Scientific Reports, 9, 18150.

------------------------------------------------------------------------------
HOW TO USE
------------------------------------------------------------------------------
1. Edit the CONFIG block below: folder paths, volume size, epochs, batch size.
2. Run:
       python run_adni.py
3. Trained weights land in the same folder as: ad_nc_classifier.pt, mci_classifier.pt
------------------------------------------------------------------------------
"""

import os
import random
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import nibabel as nib
except ImportError:
    raise ImportError("Missing dependency: pip install nibabel")

try:
    from scipy.ndimage import affine_transform, zoom
except ImportError:
    raise ImportError("Missing dependency: pip install scipy")


# ---- Data locations ----
DATA_ROOT_AD_NC = {
    "NC": r"D:\AD_project\ad_cnn_project\data\NC",
    "AD": r"D:\AD_project\ad_cnn_project\data\AD",
}
DATA_ROOT_MCI = {
    "sMCI": r"D:\AD_project\ad_cnn_project\data\sMCI",
    "pMCI": r"D:\AD_project\ad_cnn_project\data\pMCI",
}

# ---- training settings ----
TARGET_SHAPE = (96, 120, 96)     # paper uses (120, 160, 120); shrink if memory-limited
VAL_RATIO = 0.2
SEED = 42
BATCH_SIZE = 2                   # lower to 1 if you hit out-of-memory errors

EPOCHS_PRETRAIN = 5             # unsupervised CAE/ICAE pretraining
EPOCHS_FINETUNE = 10            # supervised fine-tuning (both AD/NC and pMCI/sMCI)
LEARNING_RATE = 1e-4
L1_LAMBDA = 1e-4
L2_LAMBDA = 1e-4

# ---- Where to save trained weights ----
AD_NC_MODEL_PATH = "ad_nc_classifier.pt"
MCI_MODEL_PATH = "mci_classifier.pt"

# ---- Whether to run saliency visualization after training ----
RUN_SALIENCY = True
SALIENCY_OUTPUT_PATH = "ad_nc_saliency_map.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# DATA: preprocessing, augmentation, Dataset
# ==============================================================================

def rescale_intensity(volume: np.ndarray) -> np.ndarray:
    v_min, v_max = float(volume.min()), float(volume.max())
    if v_max - v_min < 1e-8:
        return np.zeros_like(volume)
    return (volume - v_min) / (v_max - v_min)


def resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)


def load_and_preprocess(path: str, target_shape: Tuple[int, int, int]) -> np.ndarray:
    img = nib.load(path)
    vol = np.asarray(img.get_fdata(), dtype=np.float32)
    vol = resize_volume(vol, target_shape)
    vol = rescale_intensity(vol)
    return vol


def random_affine_augment(volume: np.ndarray) -> np.ndarray:
    """Random rotation + shift + rescale, matching the paper's augmentation."""
    angle = np.deg2rad(random.uniform(-5, 5))
    scale = random.uniform(0.8, 1.2)
    shift_amt = random.uniform(0.0, 0.1)

    d, h, w = volume.shape
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]) / scale
    center = np.array([d, h, w]) / 2.0
    offset = center - rot @ center + np.array([shift_amt * d, shift_amt * h, 0])

    return affine_transform(volume, rot, offset=offset, order=1, mode="nearest").astype(np.float32)


def random_intensity_jitter(volume: np.ndarray) -> np.ndarray:
    brightness = random.uniform(-0.1, 0.1)
    contrast = random.uniform(0.9, 1.1)
    mean = volume.mean()
    jittered = (volume - mean) * contrast + mean + brightness
    return np.clip(jittered, 0.0, 1.0).astype(np.float32)


class Sample:
    __slots__ = ("path", "label")

    def __init__(self, path: str, label: int):
        self.path = path
        self.label = label


class MRIDataset(Dataset):
    def __init__(self, samples: List[Sample], augment: bool, target_shape: Tuple[int, int, int]):
        self.samples = samples
        self.augment = augment
        self.target_shape = target_shape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol = load_and_preprocess(self.samples[idx].path, self.target_shape)
        if self.augment:
            vol = random_affine_augment(vol)
            vol = random_intensity_jitter(vol)
        tensor = torch.from_numpy(vol).unsqueeze(0).float()
        label = torch.tensor(self.samples[idx].label, dtype=torch.long)
        return tensor, label


def build_manifest_from_dirs(class_dirs: dict, extensions=(".nii", ".nii.gz")) -> List[Sample]:
    """label 0/1 assigned by sorted class-name order, e.g. {'AD':.., 'NC':..} -> AD=0, NC=1."""
    samples = []
    for label_idx, (_, folder) in enumerate(sorted(class_dirs.items())):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Data folder not found: {folder}")
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(extensions):
                samples.append(Sample(os.path.join(folder, fname), label_idx))
    return samples


def split_manifest(samples: List[Sample], val_ratio: float, seed: int):
    """Stratified split so each class is represented in both train and val."""
    random.seed(seed)
    by_label = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    train, val = [], []
    for _, items in by_label.items():
        items = items[:]
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


def build_loaders(data_root: dict, target_shape, val_ratio, seed, batch_size):
    manifest = build_manifest_from_dirs(data_root)
    label_names = sorted(data_root.keys())
    print(f"  Label mapping: {label_names[0]}=0, {label_names[1]}=1")
    for i, name in enumerate(label_names):
        n = sum(1 for s in manifest if s.label == i)
        print(f"  {name}: {n} samples")
    if len(manifest) < 10:
        print("  WARNING: very few samples -- results will be noisy.")

    train_samples, val_samples = split_manifest(manifest, val_ratio, seed)
    print(f"  Train: {len(train_samples)}   Val: {len(val_samples)}")

    train_ds = MRIDataset(train_samples, augment=True, target_shape=target_shape)
    val_ds = MRIDataset(val_samples, augment=False, target_shape=target_shape)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


# ==============================================================================
# MODEL: Inception-based Convolutional Autoencoder (ICAE) + classifier head
# ==============================================================================

class GaussianDropout(nn.Module):
    def __init__(self, p: float = 0.8):
        super().__init__()
        self.alpha = p / (1.0 - p)

    def forward(self, x):
        if not self.training or self.alpha == 0:
            return x
        noise = torch.randn_like(x) * self.alpha + 1.0
        return x * noise


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.8):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = GaussianDropout(dropout_p)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = torch.relu(x)
        return self.pool(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=True, dropout_p=0.8, final=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False) if upsample else nn.Identity()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = GaussianDropout(dropout_p)
        self.final = final

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        if self.final:
            return torch.sigmoid(x)
        x = self.dropout(x)
        return torch.relu(x)


class InceptionModule3D(nn.Module):
    def __init__(self, in_ch, branch_ch=10):
        super().__init__()
        c = branch_ch
        self.branch1 = nn.Sequential(nn.Conv3d(in_ch, c, 1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_ch, c, 1), nn.ReLU(inplace=True),
            nn.Conv3d(c, c, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_ch, c, 1), nn.ReLU(inplace=True),
            nn.Conv3d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(c, c, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(in_ch, c, 1), nn.ReLU(inplace=True),
        )
        self.out_channels = 4 * c

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class InceptionConvAutoencoder(nn.Module):
    """ICAE: 2 conv stem layers + 1 Inception module (encoder), mirrored decoder."""

    def __init__(self, in_channels=1, feat=10, inception_branch=10, dropout_p=0.8):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, feat, dropout_p)
        self.enc2 = ConvBlock(feat, feat, dropout_p)
        self.inception = InceptionModule3D(feat, inception_branch)
        inc_out = self.inception.out_channels

        self.dec_inc = DeconvBlock(inc_out, feat, upsample=False, dropout_p=dropout_p)
        self.dec2 = DeconvBlock(feat, feat, dropout_p=dropout_p)
        self.dec1 = DeconvBlock(feat, in_channels, dropout_p=dropout_p, final=True)

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        return self.inception(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.dec_inc(z)
        x_hat = self.dec2(x_hat)
        return self.dec1(x_hat)


class Classifier(nn.Module):
    """Wraps an ICAE encoder with a classification head (no FC layer, per paper)."""

    def __init__(self, encoder: InceptionConvAutoencoder, num_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(encoder.inception.out_channels, num_classes)

    def forward(self, x):
        z = self.encoder.encode(x)
        z = self.pool(z).flatten(1)
        return self.head(z)


def match_shape(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Crop/pad x_hat's spatial dims to exactly match x (handles pooling rounding)."""
    if x_hat.shape == x.shape:
        return x_hat
    slices = [slice(None), slice(None)]
    for d_hat, d in zip(x_hat.shape[2:], x.shape[2:]):
        if d_hat >= d:
            start = (d_hat - d) // 2
            slices.append(slice(start, start + d))
        else:
            slices.append(slice(0, d_hat))
    cropped = x_hat[tuple(slices)]
    if cropped.shape != x.shape:
        pad = []
        for d_hat, d in zip(reversed(cropped.shape[2:]), reversed(x.shape[2:])):
            diff = max(0, d - d_hat)
            pad.extend([diff // 2, diff - diff // 2])
        cropped = F.pad(cropped, pad)
    return cropped


def l1_penalty(model: nn.Module) -> torch.Tensor:
    return sum(p.abs().sum() for p in model.parameters())


# ==============================================================================
# TRAINING
# ==============================================================================

def pretrain_autoencoder(model: nn.Module, loader: DataLoader) -> nn.Module:
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.9), weight_decay=L2_LAMBDA)
    bce = nn.BCELoss()
    model.train()

    for epoch in range(EPOCHS_PRETRAIN):
        running_loss, n_batches = 0.0, 0
        for x, _ in loader:
            x = x.to(DEVICE)
            x_hat = match_shape(model(x), x)
            loss = bce(x_hat.clamp(1e-6, 1 - 1e-6), x.clamp(0, 1)) + L1_LAMBDA * l1_penalty(model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        if epoch % max(1, EPOCHS_PRETRAIN // 10) == 0 or epoch == EPOCHS_PRETRAIN - 1:
            print(f"  [Pretrain] epoch {epoch+1}/{EPOCHS_PRETRAIN}  loss={running_loss / max(n_batches,1):.4f}")
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    n_correct, n_total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        n_correct += (logits.argmax(1) == y).sum().item()
        n_total += y.size(0)
    return n_correct / max(n_total, 1)


def finetune_classifier(classifier: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    classifier.to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.9), weight_decay=L2_LAMBDA)
    ce = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0.0, copy.deepcopy(classifier.state_dict())

    for epoch in range(EPOCHS_FINETUNE):
        classifier.train()
        running_loss, n_correct, n_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = classifier(x)
            loss = ce(logits, y) + L1_LAMBDA * l1_penalty(classifier)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_correct += (logits.argmax(1) == y).sum().item()
            n_total += y.size(0)

        train_acc = n_correct / max(n_total, 1)
        val_acc = evaluate(classifier, val_loader)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(classifier.state_dict())

        if epoch % max(1, EPOCHS_FINETUNE // 10) == 0 or epoch == EPOCHS_FINETUNE - 1:
            print(f"  [Finetune] epoch {epoch+1}/{EPOCHS_FINETUNE}  "
                  f"train_loss={running_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    classifier.load_state_dict(best_state)
    return classifier


# ==============================================================================
# SALIENCY (gradient-based class saliency, Simonyan et al. 2013 / paper Algorithm 1)
# ==============================================================================

@torch.no_grad()
def smooth_3d(volume: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    k = kernel_size
    kernel = torch.ones((1, 1, k, k, k), device=volume.device) / (k ** 3)
    v = volume.unsqueeze(0).unsqueeze(0)
    return F.conv3d(v, kernel, padding=k // 2).squeeze(0).squeeze(0)


def instance_saliency(model: nn.Module, x: torch.Tensor, target_class: int, smooth_kernel=9) -> torch.Tensor:
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    score = logits[0, target_class]
    model.zero_grad(set_to_none=True)
    score.backward()
    saliency = x.grad.detach()[0, 0].abs()
    if smooth_kernel:
        saliency = smooth_3d(saliency, smooth_kernel)
    return saliency


def group_saliency_map(model: nn.Module, loader: DataLoader, target_class: int, max_subjects=20) -> torch.Tensor:
    model.eval()
    total, n = None, 0
    for x, y in loader:
        for i in range(x.size(0)):
            if y[i].item() != target_class:
                continue
            vol = x[i:i+1].to(DEVICE)
            m = instance_saliency(model, vol, target_class)
            total = m if total is None else total + m
            n += 1
            if n >= max_subjects:
                break
        if n >= max_subjects:
            break
    if n == 0:
        raise ValueError(f"No samples of class {target_class} found for saliency computation.")
    fused = total / n
    fused = (fused - fused.min()) / fused.max().clamp(min=1e-8)
    return fused.detach().cpu()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"Using device: {DEVICE}\n")

    # ---- Data ----
    print("=== Preparing AD vs. NC data ===")
    ad_nc_train_loader, ad_nc_val_loader = build_loaders(
        DATA_ROOT_AD_NC, TARGET_SHAPE, VAL_RATIO, SEED, BATCH_SIZE
    )

    print("\n=== Preparing pMCI vs. sMCI data ===")
    mci_train_loader, mci_val_loader = build_loaders(
        DATA_ROOT_MCI, TARGET_SHAPE, VAL_RATIO, SEED, BATCH_SIZE
    )

    # ---- Stage 1: unsupervised pretraining ----
    print("\n=== Stage 1: ICAE unsupervised pretraining ===")
    autoencoder = InceptionConvAutoencoder()
    autoencoder = pretrain_autoencoder(autoencoder, ad_nc_train_loader)

    # ---- Stage 2: AD vs. NC fine-tuning ----
    print("\n=== Stage 2: AD vs. NC supervised fine-tuning ===")
    ad_nc_classifier = Classifier(autoencoder)
    ad_nc_classifier = finetune_classifier(ad_nc_classifier, ad_nc_train_loader, ad_nc_val_loader)
    ad_nc_val_acc = evaluate(ad_nc_classifier, ad_nc_val_loader)
    print(f"\n>>> AD vs. NC final val accuracy: {ad_nc_val_acc:.4f}")

    torch.save(ad_nc_classifier.state_dict(), AD_NC_MODEL_PATH)
    print(f">>> Saved: {AD_NC_MODEL_PATH}")

    # ---- Stage 3: transfer learning to pMCI vs. sMCI ----
    print("\n=== Stage 3: pMCI vs. sMCI transfer learning ===")
    fresh_autoencoder = InceptionConvAutoencoder()
    mci_classifier = Classifier(fresh_autoencoder)
    mci_classifier.encoder.load_state_dict(ad_nc_classifier.encoder.state_dict())  # transfer weights
    mci_classifier = finetune_classifier(mci_classifier, mci_train_loader, mci_val_loader)
    mci_val_acc = evaluate(mci_classifier, mci_val_loader)
    print(f"\n>>> pMCI vs. sMCI final val accuracy: {mci_val_acc:.4f}")

    torch.save(mci_classifier.state_dict(), MCI_MODEL_PATH)
    print(f">>> Saved: {MCI_MODEL_PATH}")

    # ---- Summary ----
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"AD vs. NC       val accuracy: {ad_nc_val_acc:.4f}")
    print(f"pMCI vs. sMCI   val accuracy: {mci_val_acc:.4f}")

    # ---- Optional: saliency map ----
    if RUN_SALIENCY:
        print("\n=== Generating saliency map (AD vs. NC, target class = AD) ===")
        # label mapping is alphabetical: AD=0, NC=1 (see build_loaders printout above)
        saliency_map = group_saliency_map(ad_nc_classifier, ad_nc_val_loader, target_class=0)
        np.save(SALIENCY_OUTPUT_PATH, saliency_map.numpy())
        print(f">>> Saved saliency map: {SALIENCY_OUTPUT_PATH}  shape={tuple(saliency_map.shape)}")


if __name__ == "__main__":
    main()