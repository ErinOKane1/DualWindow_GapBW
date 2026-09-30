"""Train a residual U-Net to predict full-band OCT from dual-window B-scans.

This starter script uses every S1/S2 TIFF pair in ``test_data`` by default:

    python train_dual_window_unet.py \
        --data-dir test_data \
        --pixel-map test_data/OCT_pixelMap.mat \
        --epochs 100

It reconstructs three aligned volumes:
    input channel 0: short-wavelength dual-window OCT
    input channel 1: long-wavelength dual-window OCT
    target:          full-band OCT

Every ``--save-every`` epochs, it saves a one-row comparison panel:
full band, short DW, long DW, and DL reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from oct_hsv_python_stage1.dual_window import (
    amplitude_to_db,
    reconstruct_dual_window_pair,
)

from time import perf_counter


class ResidualBlock(nn.Module):
    """Residual block matching the paper-style alpha-scaled skip structure."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.block(x)


class DualWindowUNet(nn.Module):
    """Small residual U-Net: 2-channel DW input -> 1-channel full-band output."""

    def __init__(self, in_channels: int = 2, out_channels: int = 1, base: int = 32):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base, kernel_size=3, padding=1)

        self.enc1 = nn.Sequential(*[ResidualBlock(base) for _ in range(4)])
        self.down1 = nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1)

        self.enc2 = nn.Sequential(*[ResidualBlock(base * 2) for _ in range(3)])
        self.down2 = nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1)

        self.bottleneck = nn.Sequential(*[ResidualBlock(base * 4) for _ in range(4)])

        self.up1 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
        self.dec1 = nn.Sequential(*[ResidualBlock(base * 2) for _ in range(3)])

        self.up2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
        self.dec2 = nn.Sequential(*[ResidualBlock(base) for _ in range(3)])

        self.out_conv = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = (4 - height % 4) % 4
        pad_w = (4 - width % 4) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        e1 = self.enc1(self.in_conv(x))
        e2 = self.enc2(self.down1(e1))
        bottleneck = self.bottleneck(self.down2(e2))

        d1 = self.dec1(self.up1(bottleneck) + e2)
        d2 = self.dec2(self.up2(d1) + e1)
        out = self.out_conv(d2)
        return out[..., :height, :width]

class ForegroundPatchBscanDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        patch_size: tuple[int, int] = (256, 256),
        patches_per_bscan: int = 4,
        random_crop: bool = True,
        min_mean_target: float = 0.08,
        min_fraction_target: float = 0.05,
        target_threshold: float = 0.10,
        max_attempts: int = 25,
    ):
        self.inputs = inputs
        self.targets = targets
        self.indices = indices.astype(np.int64)
        self.patch_h, self.patch_w = patch_size
        self.patches_per_bscan = patches_per_bscan
        self.random_crop = random_crop
        self.min_mean_target = min_mean_target
        self.min_fraction_target = min_fraction_target
        self.target_threshold = target_threshold
        self.max_attempts = max_attempts

    def __len__(self) -> int:
        return len(self.indices) * self.patches_per_bscan

    def _choose_patch_origin(self, height: int, width: int, patch_h: int, patch_w: int):
        if self.random_crop:
            top = np.random.randint(0, height - patch_h + 1) if height > patch_h else 0
            left = np.random.randint(0, width - patch_w + 1) if width > patch_w else 0
        else:
            top = max(0, (height - patch_h) // 2)
            left = max(0, (width - patch_w) // 2)
        return top, left

    def __getitem__(self, item: int):
        bscan_index = self.indices[item // self.patches_per_bscan]

        x = self.inputs[bscan_index]
        y = self.targets[bscan_index]

        _, height, width = x.shape
        patch_h = min(self.patch_h, height)
        patch_w = min(self.patch_w, width)

        best_patch = None
        best_score = -np.inf

        for _ in range(self.max_attempts):
            top, left = self._choose_patch_origin(height, width, patch_h, patch_w)

            x_patch = x[:, top:top + patch_h, left:left + patch_w]
            y_patch = y[:, top:top + patch_h, left:left + patch_w]

            target_patch = y_patch[0]
            mean_target = float(target_patch.mean())
            fraction_target = float((target_patch > self.target_threshold).mean())

            score = mean_target + fraction_target

            if score > best_score:
                best_score = score
                best_patch = (x_patch, y_patch)

            if (
                mean_target >= self.min_mean_target
                and fraction_target >= self.min_fraction_target
            ):
                return torch.from_numpy(x_patch), torch.from_numpy(y_patch)

        # Fallback: if no patch passes the threshold, return the best attempt.
        x_patch, y_patch = best_patch
        return torch.from_numpy(x_patch), torch.from_numpy(y_patch)
    

class BscanDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, indices: np.ndarray):
        self.inputs = inputs
        self.targets = targets
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        x = torch.from_numpy(self.inputs[index])
        y = torch.from_numpy(self.targets[index])
        return x, y


def find_tiff_pairs(data_dir: Path, s1_glob: str, s2_glob: str) -> tuple[list[Path], list[Path]]:
    s1_paths = sorted(data_dir.glob(s1_glob))
    s2_paths = sorted(data_dir.glob(s2_glob))
    if not s1_paths:
        raise FileNotFoundError(f"No S1 files found with {data_dir / s1_glob}")
    if len(s1_paths) != len(s2_paths):
        raise ValueError(f"Found {len(s1_paths)} S1 files and {len(s2_paths)} S2 files.")
    return s1_paths, s2_paths


def reconstruct_or_load_cache(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = Path(args.cache)
    if cache_path.exists() and not args.rebuild_cache:
        print(f"Loading cached training arrays: {cache_path}")
        with np.load(cache_path) as data:
            return data["full"], data["short"], data["long"]

    data_dir = Path(args.data_dir)
    s1_paths, s2_paths = find_tiff_pairs(data_dir, args.s1_glob, args.s2_glob)
    full_slices = []
    short_slices = []
    long_slices = []

    print(f"Reconstructing {len(s1_paths)} B-scans from {data_dir}")
    for index, (s1_path, s2_path) in enumerate(zip(s1_paths, s2_paths), start=1):
        result = reconstruct_dual_window_pair(
            s1_path,
            s2_path,
            args.pixel_map,
            c2=args.c2,
            c3=args.c3,
            broad_width_fraction=args.broad_width,
            narrow_width_fraction=args.narrow_width,
            center_separation_fraction=args.center_separation,
        )
        full_slices.append(result.full_band.astype(np.float32, copy=False))
        short_slices.append(result.short_wavelength.astype(np.float32, copy=False))
        long_slices.append(result.long_wavelength.astype(np.float32, copy=False))
        print(f"  reconstructed {index}/{len(s1_paths)}: {s1_path.name}")

    full = np.stack(full_slices, axis=0)
    short = np.stack(short_slices, axis=0)
    long = np.stack(long_slices, axis=0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        full=full,
        short=short,
        long=long,
        metadata=json.dumps(
            {
                "data_dir": str(data_dir),
                "pixel_map": str(args.pixel_map),
                "broad_width": args.broad_width,
                "narrow_width": args.narrow_width,
                "center_separation": args.center_separation,
            }
        ),
    )
    print(f"Saved cache: {cache_path}")
    return full, short, long


def normalize_volumes(
    full_amp: np.ndarray,
    short_amp: np.ndarray,
    long_amp: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.8,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    full_db = amplitude_to_db(full_amp).astype(np.float32)
    short_db = amplitude_to_db(short_amp).astype(np.float32)
    long_db = amplitude_to_db(long_amp).astype(np.float32)
    low, high = np.percentile(full_db, [low_percentile, high_percentile])

    def scale(x: np.ndarray) -> np.ndarray:
        return np.clip((x - low) / max(high - low, 1e-6), 0.0, 1.0).astype(np.float32)

    inputs = np.stack((scale(short_db), scale(long_db)), axis=1)
    targets = scale(full_db)[:, None, :, :]
    return inputs, targets, float(low), float(high)


def make_train_val_indices(n_slices: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_slices)
    if n_slices < 5 or val_fraction <= 0:
        return indices, indices
    val_count = max(1, int(round(n_slices * val_fraction)))
    val_indices = indices[-val_count:]
    train_indices = indices[:-val_count]
    if train_indices.size == 0:
        train_indices = indices
    return train_indices, val_indices


def save_prediction_panel(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    index: int,
    epoch: int,
    output_dir: Path,
    device: torch.device,
) -> None:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(inputs[index:index + 1]).to(device)
        pred = model(x).cpu().numpy()[0, 0]

    full = targets[index, 0]
    short = inputs[index, 0]
    long = inputs[index, 1]
    images = [
        (full, "Full-band target"),
        (short, "Short DW input"),
        (long, "Long DW input"),
        (pred, "DL reconstruction"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for ax, (image, title) in zip(axes, images):
        im = ax.imshow(image, cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("A-line")
        ax.set_ylabel("Depth pixel")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"epoch_{epoch:04d}_bscan_{index:04d}.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prediction panel: {figure_path}")

def regression_metrics(pred, target):
    error = pred - target
    abs_error = torch.abs(error)
    mse = torch.mean(error**2)
    psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-12))

    return {
        "l1": float(torch.mean(abs_error).detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "psnr_db": float(psnr.detach().cpu()),
        "acc_005": float(torch.mean((abs_error < 0.05).float()).detach().cpu()),
        "acc_010": float(torch.mean((abs_error < 0.10).float()).detach().cpu()),
    }

def train(args: argparse.Namespace) -> None:
    full, short, long = reconstruct_or_load_cache(args)
    inputs, targets, db_low, db_high = normalize_volumes(full, short, long)
    train_indices, val_indices = make_train_val_indices(inputs.shape[0], args.val_fraction)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = DualWindowUNet(base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()

    PATCH_SIZE = (256, 256)
    PATCHES_PER_BSCAN = 4

    train_loader = DataLoader(
        ForegroundPatchBscanDataset(
            inputs,
            targets,
            train_indices,
            patch_size=PATCH_SIZE,
            patches_per_bscan=PATCHES_PER_BSCAN,
            random_crop=True,
            min_mean_target=0.08,
            min_fraction_target=0.05,
            target_threshold=0.10,
            max_attempts=25,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        ForegroundPatchBscanDataset(
            inputs,
            targets,
            val_indices,
            patch_size=PATCH_SIZE,
            patches_per_bscan=1,
            random_crop=False,
            min_mean_target=0.0,
            min_fraction_target=0.0,
            target_threshold=0.10,
            max_attempts=1,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training on {len(train_indices)} B-scans; validating on {len(val_indices)} B-scans")
    print(f"Device: {device}; normalization dB range: {db_low:.2f} to {db_high:.2f}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = perf_counter()
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_metrics = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                val_metrics.append(regression_metrics(pred, y))

        val_summary = {
            key: float(np.mean([metrics[key] for metrics in val_metrics]))
            for key in val_metrics[0]
        }

        train_loss = float(np.mean(train_losses))
        #val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        epoch_seconds = perf_counter() - epoch_start

        print(
            f"Epoch {epoch:04d}/{args.epochs} | "
            f"{epoch_seconds:.1f}s | "
            f"train L1={train_loss:.5f} | "
            f"val L1={val_summary['l1']:.5f} | "
            f"val MSE={val_summary['mse']:.6f} | "
            f"PSNR={val_summary['psnr_db']:.2f} dB | "
            f"acc<0.05={100 * val_summary['acc_005']:.1f}% | "
            f"acc<0.10={100 * val_summary['acc_010']:.1f}%"
        )

        if epoch % args.save_every == 0 or epoch == 1 or epoch == args.epochs:
            panel_index = int(val_indices[len(val_indices) // 2])
            save_prediction_panel(model, inputs, targets, panel_index, epoch, output_dir, device)
            checkpoint_path = output_dir / "latest_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "db_low": db_low,
                    "db_high": db_high,
                    "args": vars(args),
                },
                checkpoint_path,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a U-Net to reconstruct full-band OCT from short/long DW OCT."
    )
    parser.add_argument("--data-dir", default="test_data", help="Folder containing S1/S2 TIFF pairs.")
    parser.add_argument("--pixel-map", default="test_data/OCT_pixelMap.mat", help="MAT file with pixelMap.")
    parser.add_argument("--s1-glob", default="*S1*.tif*", help="Glob for S1 TIFFs inside --data-dir.")
    parser.add_argument("--s2-glob", default="*S2*.tif*", help="Glob for S2 TIFFs inside --data-dir.")
    parser.add_argument("--cache", default="output_training/dual_window_training_cache.npz")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--output-dir", default="output_training")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--c2", type=float, default=38.02)
    parser.add_argument("--c3", type=float, default=-54.63)
    parser.add_argument("--broad-width", type=float, default=0.25)
    parser.add_argument("--narrow-width", type=float, default=0.05)
    parser.add_argument("--center-separation", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
