"""Shared utilities for explainability scripts.

Provides helpers to load models, build dataloaders, format tensors for
visualisation, and create structured output directories so individual XAI
methods stay lightweight and consistent.
"""
from __future__ import annotations

import json
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torchvision.utils import make_grid, save_image

from models.hybrid_vae import HybridVAE, HybridVAEConfig
from data.amsl_quads import AMSLQuadsConfig, build_dataloader

# -----------------------------------------------------------------------------
# Model / device utilities
# -----------------------------------------------------------------------------

def resolve_device(choice: str = "auto") -> str:
    """Resolve a user provided device string to either 'cuda' or 'cpu'."""
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice not in {"cuda", "cpu"}:
        raise ValueError("device must be one of {'auto','cuda','cpu'}")
    if choice == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return choice


def filter_cfg(raw_cfg: Dict) -> HybridVAEConfig:
    """Strip unknown keys from a checkpoint config before dataclass init."""
    allowed = {f.name for f in dataclass_fields(HybridVAEConfig)}
    clean = {}
    for key, value in raw_cfg.items():
        if key in allowed:
            if key == "attn_scales" and isinstance(value, list):
                value = tuple(value)
            clean[key] = value
    return HybridVAEConfig(**clean)


def load_model(
    ckpt_path: str,
    *,
    device: str = "auto",
    img_size_override: Optional[int] = None,
    eval_mode: bool = True,
) -> Tuple[HybridVAE, HybridVAEConfig, Dict]:
    """Load a HybridVAE checkpoint and return (model, cfg, payload)."""
    payload = torch.load(ckpt_path, map_location="cpu")
    raw_cfg = payload["cfg"] if isinstance(payload["cfg"], dict) else payload["cfg"].__dict__
    cfg = filter_cfg(raw_cfg)
    if img_size_override is not None:
        cfg.img_size = img_size_override

    resolved = resolve_device(device)
    model = HybridVAE(cfg).to(resolved)
    model.load_state_dict(payload["model"], strict=False)
    model = model.eval() if eval_mode else model.train()
    return model, cfg, payload


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def build_loader(
    cfg: HybridVAEConfig,
    *,
    root: str,
    split: str,
    batch_size: int,
    device: str,
    shuffle: bool = False,
    max_items: Optional[int] = None,
    include_annotations: bool = False,
):
    """Return (dataset, dataloader) with Windows-safe worker defaults."""
    is_windows = (torch.utils.data.get_worker_info() is not None)  # runtime check
    # If called outside DataLoader workers, fall back to os.name to avoid surprises
    if torch.utils.data.get_worker_info() is None:
        import os
        is_windows = os.name == "nt"

    num_workers = 0 if is_windows else 6
    ds_cfg = AMSLQuadsConfig(
        root=root,
        split=split,
        img_size=cfg.img_size,
        grayscale=(cfg.img_channels == 1),
        include_annotations=include_annotations,
        max_items=max_items,
    )
    ds, dl = build_dataloader(
        ds_cfg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(device == "cuda" and not is_windows and num_workers > 0),
    )
    return ds, dl


# -----------------------------------------------------------------------------
# Visual utilities
# -----------------------------------------------------------------------------

def ensure_rgb(t: Tensor) -> Tensor:
    """Repeat single-channel tensors to 3 channels for saving."""
    if t.size(1) == 1:
        return t.repeat(1, 3, 1, 1)
    return t


def save_grid(
    tensors: Iterable[Tensor],
    path: Path,
    *,
    nrow: int,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Save a tiled grid made from a collection of tensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stacked = torch.cat([ensure_rgb(t) for t in tensors], dim=0)
    grid = make_grid(stacked, nrow=nrow, padding=padding, normalize=normalize, value_range=value_range)
    save_image(grid, str(path))


def timestamped_subdir(base: Path, method: str) -> Path:
    """Create a timestamped directory for a given explainability method."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = base / method / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def dump_metadata(path: Path, payload: Dict) -> None:
    """Write JSON metadata alongside generated artefacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -----------------------------------------------------------------------------
# Heatmap overlay
# -----------------------------------------------------------------------------

def overlay_heatmap(
    base: Tensor,
    heatmap: Tensor,
    *,
    alpha: float = 0.5,
    cmap: str = "magma",
) -> Tensor:
    """Blend a normalised heatmap onto the base image."""
    import matplotlib.pyplot as plt

    base_rgb = ensure_rgb(base).cpu()
    hm = heatmap.squeeze(1).cpu()
    overlays = []
    cmap_fn = plt.get_cmap(cmap)
    for img, hm_i in zip(base_rgb, hm):
        hm_norm = hm_i - hm_i.min()
        denom = hm_norm.max().item() or 1.0
        hm_norm = hm_norm / denom
        colors = torch.tensor(cmap_fn(hm_norm.numpy())[:, :, :3]).permute(2, 0, 1)
        blended = (1 - alpha) * img + alpha * colors
        overlays.append(blended.clamp(0, 1))
    return torch.stack(overlays, dim=0).to(base.device)

