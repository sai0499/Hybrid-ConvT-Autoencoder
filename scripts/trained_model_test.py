# scripts/reconstruct.py
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import torchvision.transforms.functional as TF
from pytorch_msssim import ms_ssim as ms_ssim_fn

from models.hybrid_vae import HybridVAE, HybridVAEConfig

# Speed: safe TF32 paths on NVIDIA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------- helpers -----------------------------
def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3]) + eps
    return 10.0 * torch.log10(1.0 / mse)

def to_vis3(t: torch.Tensor) -> torch.Tensor:
    return t.repeat(1, 3, 1, 1) if t.size(1) == 1 else t

def filter_cfg(raw_cfg: dict) -> HybridVAEConfig:
    """Build a clean HybridVAEConfig from possibly noisy checkpoint cfg."""
    from dataclasses import fields as dataclass_fields
    allowed = {f.name for f in dataclass_fields(HybridVAEConfig)}
    clean = {}
    for k, v in raw_cfg.items():
        if k in allowed:
            if k == "attn_scales" and isinstance(v, list):
                v = tuple(v)
            clean[k] = v
    return HybridVAEConfig(**clean)

def load_model(ckpt_path: str, img_size_override: int | None, device: str) -> tuple[HybridVAE, HybridVAEConfig]:
    payload = torch.load(ckpt_path, map_location="cpu")
    raw_cfg = payload["cfg"] if isinstance(payload["cfg"], dict) else payload["cfg"].__dict__
    cfg = filter_cfg(raw_cfg)
    if img_size_override is not None:
        cfg.img_size = img_size_override
    model = HybridVAE(cfg).to(device).eval()
    model.load_state_dict(payload["model"], strict=False)
    return model, cfg

def load_and_preprocess_image(
    path: str, target_size: int, channels: int, fit: str = "resize", binarize: float | None = None
) -> torch.Tensor:
    """
    Load an image with PIL, convert to 1ch or 3ch, fit to square target_size via:
      - 'resize'    : direct resize to (S,S)
      - 'letterbox' : keep aspect ratio and pad with white
    Return tensor [1, C, S, S] in [0,1].
    """
    assert fit in {"resize", "letterbox"}
    img = Image.open(path).convert("L" if channels == 1 else "RGB")

    if fit == "resize":
        # use NEAREST for crisp rectangles; change to BICUBIC for natural images
        img = img.resize((target_size, target_size), resample=Image.NEAREST)
    else:
        # letterbox to square on white background
        w, h = img.size
        scale = min(target_size / w, target_size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img_resized = img.resize((nw, nh), resample=Image.NEAREST)
        bg = Image.new(img.mode, (target_size, target_size), color=(255 if channels == 1 else (255, 255, 255)))
        left = (target_size - nw) // 2
        top = (target_size - nh) // 2
        bg.paste(img_resized, (left, top))
        img = bg

    # to tensor in [0,1], shape [C,H,W]
    t = TF.to_tensor(img)

    if binarize is not None:
        # threshold after scaling to [0,1]
        t = (t > float(binarize)).float()

    return t.unsqueeze(0)  # [1,C,H,W]

def save_triptych(x: torch.Tensor, recon: torch.Tensor, out_dir: Path, stem: str):
    """
    Save side-by-side (input | recon) and an error heatmap.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    k = 1  # single image
    grid = torch.cat([x, recon], dim=0)
    save_image(make_grid(to_vis3(grid), nrow=k, padding=2), str(out_dir / f"{stem}_side_by_side.png"))

    err = (recon - x).abs()
    err = err / (err.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
    save_image(make_grid(to_vis3(err), nrow=1, padding=2), str(out_dir / f"{stem}_error.png"))

# ----------------------------- main -----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Reconstruct a single image with HybridVAE")
    ap.add_argument("--ckpt", required=True, type=str, help="Path to model checkpoint (.pt)")
    ap.add_argument("--input", required=True, type=str, help="Path to input image (bmp/png/jpg)")
    ap.add_argument("--out_dir", type=str, default="results/reconstruct", help="Output folder")
    ap.add_argument("--img_size", type=int, default=None, help="Override image size (default=from ckpt)")
    ap.add_argument("--fit", choices=["resize", "letterbox"], default="resize", help="How to fit to square")
    ap.add_argument("--binarize", type=float, default=None, help="Optional threshold in [0,1] (e.g., 0.5)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)

    # Load model
    model, cfg = load_model(args.ckpt, args.img_size, device)
    C, S = cfg.img_channels, cfg.img_size

    # Load & preprocess your image
    x = load_and_preprocess_image(args.input, target_size=S, channels=C, fit=args.fit, binarize=args.binarize).to(device)

    # Encode → Decode (using skip features for true reconstruction)
    mu, logvar, feats = model.encode(x)
    recon = model.decode(mu, feats).clamp(0, 1)

    # Metrics
    l1 = torch.mean(torch.abs(recon - x), dim=[1, 2, 3]).item()
    ssim = ms_ssim_fn(recon.float(), x.float(), data_range=1.0, size_average=True).item()
    psnr_v = psnr(recon, x).mean().item()

    print(f"Reconstruction metrics: L1={l1:.6f}  MS-SSIM={ssim:.4f}  PSNR={psnr_v:.2f}dB")

    # Save visuals
    stem = Path(args.input).stem
    save_triptych(x, recon, out_dir, stem)
    save_image(to_vis3(recon), str(out_dir / f"{stem}_recon.png"))
    save_image(to_vis3(x), str(out_dir / f"{stem}_input.png"))

    print(f"Saved outputs to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
