"""
Utils para el Proyecto Final #1 - Transfer Learning SAM2
Segmentacion Semantica Multiclase en SuperTuxKart

Contiene:
- SuperTuxKartDataset: Dataset con carga, augmentation y normalizacion
- Funciones de visualizacion (denormalize, show_predictions, mask_to_color)
- Metricas (IoU por clase y mean IoU)
- Helpers para explorar el dataset
"""

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
import torchvision.transforms.functional as TF

# ============================================================
# Constantes
# ============================================================
NUM_CLASSES = 7
CLASS_NAMES = ["Background", "Track", "Kart", "Pickup", "Nitro", "Bomb", "Projectile"]
IMG_SIZE = (512, 512)  # SAM2 trabaja mejor con resoluciones mas altas
BATCH_SIZE = 4  # Mas bajo por el tamanio de SAM2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

COLORS = np.array([
    [0, 0, 0],        # Background - negro
    [128, 128, 128],   # Track - gris
    [255, 0, 0],       # Kart - rojo
    [0, 255, 0],       # Pickup - verde
    [0, 0, 255],       # Nitro - azul
    [255, 255, 0],     # Bomb - amarillo
    [255, 0, 255],     # Projectile - magenta
], dtype=np.uint8)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# Dataset
# ============================================================
class SuperTuxKartDataset(Dataset):
    """Dataset de segmentacion semantica para SuperTuxKart.

    Estructura esperada:
        root/
            images/
                abyss_frame_0000.png
                ...
            masks/
                abyss_mask_combined_0000.png
                ...
    """

    def __init__(self, root, img_size=IMG_SIZE, augment=False):
        self.img_size = img_size
        self.augment = augment
        self.samples = []
        self.sample_classes = []  # clases presentes en cada muestra (para oversampling)

        images_dir = os.path.join(root, "images")
        masks_dir = os.path.join(root, "masks")

        for fname in sorted(os.listdir(images_dir)):
            if not fname.endswith(".png"):
                continue
            # images/abyss_frame_0128.png -> masks/abyss_mask_combined_0128.png
            parts = fname.replace(".png", "").split("_frame_")
            if len(parts) != 2:
                continue
            track_name, idx = parts
            mask_fname = f"{track_name}_mask_combined_{idx}.png"
            mask_path = os.path.join(masks_dir, mask_fname)
            if os.path.exists(mask_path):
                self.samples.append((os.path.join(images_dir, fname), mask_path))
                mask = np.array(Image.open(mask_path))
                self.sample_classes.append(set(np.unique(mask).tolist()))

        print(f"Dataset cargado: {len(self.samples)} muestras desde {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_path, mask_path = self.samples[index]

        image = Image.open(frame_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, fill=0)
                mask = TF.rotate(mask, angle, fill=0)
            if random.random() > 0.3:
                image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
            if random.random() > 0.3:
                image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
            if random.random() > 0.3:
                image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask = torch.from_numpy(np.array(mask)).long()
        mask = mask.clamp(0, NUM_CLASSES - 1)

        return image, mask


# ============================================================
# DataLoaders con Oversampling
# ============================================================
def create_dataloaders(dataset_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                       val_split=0.2, seed=42):
    """Crea train y val dataloaders con oversampling para clases raras."""
    dataset = SuperTuxKartDataset(dataset_path, img_size=img_size, augment=False)
    dataset_aug = SuperTuxKartDataset(dataset_path, img_size=img_size, augment=True)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    train_aug = Subset(dataset_aug, train_set.indices)

    # Oversampling: clases raras (Nitro=4, Bomb=5, Projectile=6) con 5x peso
    rare_classes = {4, 5, 6}
    weights = []
    for idx in train_set.indices:
        classes_in_sample = dataset.sample_classes[idx]
        if classes_in_sample & rare_classes:
            weights.append(5.0)
        else:
            weights.append(1.0)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_aug, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    n_rare = sum(1 for w in weights if w > 1)
    print(f"Train: {len(train_aug)} muestras ({len(train_loader)} batches)")
    print(f"  - Con clases raras: {n_rare} (peso 5x)")
    print(f"  - Sin clases raras: {len(weights) - n_rare} (peso 1x)")
    print(f"Val:   {len(val_set)} muestras ({len(val_loader)} batches)")

    return train_loader, val_loader


# ============================================================
# Metricas
# ============================================================
def compute_iou(pred, target, num_classes=NUM_CLASSES):
    """Mean IoU (promedio sobre clases presentes)."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union == 0:
            continue
        ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0


def compute_iou_per_class(pred, target, num_classes=NUM_CLASSES):
    """IoU desglosado por clase."""
    ious = {}
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union > 0:
            ious[CLASS_NAMES[cls]] = (intersection / union).item()
    return ious


# ============================================================
# Visualizacion
# ============================================================
def denormalize(tensor):
    """Deshace normalizacion ImageNet para visualizar."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def mask_to_color(mask_np):
    """Convierte mascara de indices a imagen RGB con colores por clase."""
    return COLORS[mask_np]


def show_samples(dataset, num_samples=4):
    """Muestra muestras aleatorias del dataset con su mascara."""
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img, mask = dataset[idx]
        img_vis = denormalize(img).permute(1, 2, 0).numpy()
        mask_color = mask_to_color(mask.numpy())
        overlay = (img_vis * 0.5 + mask_color / 255.0 * 0.5)

        axes[row, 0].imshow(img_vis)
        axes[row, 0].set_title(f"Frame #{idx}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_color)
        axes[row, 1].set_title("Mascara")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis("off")

    patches = [mpatches.Patch(color=np.array(COLORS[i]) / 255, label=CLASS_NAMES[i])
               for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc='lower center', ncol=NUM_CLASSES, fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)
    plt.show()


def show_predictions(model, dataset_subset, device=DEVICE, num_samples=6):
    """Visualiza predicciones vs ground truth."""
    model.eval()
    indices = random.sample(range(len(dataset_subset)), min(num_samples, len(dataset_subset)))

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img, mask_gt = dataset_subset[idx]

        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).squeeze(0).cpu()

        img_vis = denormalize(img).permute(1, 2, 0).numpy()
        gt_color = mask_to_color(mask_gt.numpy())
        pred_color = mask_to_color(pred.numpy())
        overlay = (img_vis * 0.5 + pred_color / 255.0 * 0.5)

        axes[row, 0].imshow(img_vis)
        axes[row, 0].set_title("Input")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_color)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_color)
        iou = compute_iou(pred, mask_gt)
        axes[row, 2].set_title(f"Prediccion (IoU: {iou:.3f})")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Overlay")
        axes[row, 3].axis("off")

    patches = [mpatches.Patch(color=np.array(COLORS[i]) / 255, label=CLASS_NAMES[i])
               for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc='lower center', ncol=NUM_CLASSES, fontsize=9)
    plt.suptitle("Resultados de Segmentacion", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)
    plt.show()


def explore_class_distribution(dataset_path):
    """Analiza la distribucion de pixeles por clase en el dataset."""
    masks_dir = os.path.join(dataset_path, "masks")
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    frames_with_class = np.zeros(NUM_CLASSES, dtype=np.int64)

    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".png")])
    for mf in mask_files:
        mask = np.array(Image.open(os.path.join(masks_dir, mf)))
        for c in range(NUM_CLASSES):
            count = (mask == c).sum()
            pixel_counts[c] += count
            if count > 0:
                frames_with_class[c] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars = ax1.bar(CLASS_NAMES, pixel_counts, color=[c / 255 for c in COLORS])
    ax1.set_ylabel("Cantidad de pixeles")
    ax1.set_title("Distribucion de pixeles por clase (log)")
    ax1.set_yscale("log")
    for bar, count in zip(bars, pixel_counts):
        pct = count / pixel_counts.sum() * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{pct:.2f}%', ha='center', va='bottom', fontsize=8)

    ax2.bar(CLASS_NAMES, frames_with_class, color=[c / 255 for c in COLORS])
    ax2.set_ylabel("Cantidad de frames")
    ax2.set_title(f"Frames con cada clase (total: {len(mask_files)})")
    for bar, count in zip(ax2.patches, frames_with_class):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(count), ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()

    print("\nDistribucion de pixeles:")
    for name, count in zip(CLASS_NAMES, pixel_counts):
        print(f"  {name:12s}: {count:>12,} ({count / pixel_counts.sum() * 100:.2f}%)")
    print(f"\nFrames con cada clase:")
    for name, count in zip(CLASS_NAMES, frames_with_class):
        print(f"  {name:12s}: {count}")
