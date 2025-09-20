# scripts/xai_attr_correlation.py
from __future__ import annotations
import sys, os, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.hybrid_vae import HybridVAE, HybridVAEConfig
from data.amsl_quads import AMSLQuadsConfig, build_dataloader

# Speed (safe): TF32 on NVIDIA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def filter_cfg(raw_cfg: dict) -> HybridVAEConfig:
    """Build a clean HybridVAEConfig from a possibly noisy checkpoint cfg."""
    from dataclasses import fields as dataclass_fields
    allowed = {f.name for f in dataclass_fields(HybridVAEConfig)}
    clean = {}
    for k, v in raw_cfg.items():
        if k in allowed:
            if k == "attn_scales" and isinstance(v, list):
                v = tuple(v)
            clean[k] = v
    return HybridVAEConfig(**clean)


def _scalarize(a: np.ndarray) -> np.ndarray:
    """
    Ensure per-sample scalar: if 'a' has extra dims (e.g., [N,1,256]),
    flatten per sample and average to produce shape [N].
    """
    a = np.asarray(a)
    if a.ndim == 1:
        return a
    return a.reshape(a.shape[0], -1).mean(axis=1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Latent–attribute correlations (XAI)")
    ap.add_argument("--ckpt", required=True, type=str, help="Path to model checkpoint (.pt)")
    ap.add_argument("--root", default="./AMSL Dataset", type=str, help="Dataset root")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split")
    ap.add_argument("--img_size", type=int, default=None, help="Override image size (else from ckpt)")
    ap.add_argument("--max_items", type=int, default=5000, help="Max number of samples")
    ap.add_argument("--batch", type=int, default=128, help="Batch size for encoding")
    ap.add_argument("--out_dir", type=str, default="results/xai", help="Output directory")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_windows = (os.name == "nt")
    num_workers = 0 if is_windows else 6

    # ----- load model -----
    payload = torch.load(args.ckpt, map_location="cpu")
    raw_cfg = payload["cfg"] if isinstance(payload["cfg"], dict) else payload["cfg"].__dict__
    cfg = filter_cfg(raw_cfg)
    if args.img_size is not None:
        cfg.img_size = args.img_size

    model = HybridVAE(cfg).to(device).eval()
    model.load_state_dict(payload["model"], strict=False)

    # ----- data -----
    ds_cfg = AMSLQuadsConfig(
        root=args.root, split=args.split, img_size=cfg.img_size,
        grayscale=(cfg.img_channels == 1), include_annotations=False, max_items=None
    )
    _, dl = build_dataloader(
        ds_cfg, batch_size=args.batch, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(device == "cuda" and not is_windows)
    )

    # ----- accumulators -----
    Z = []            # [N, D] latent means
    area = []         # [N] foreground area fraction
    xspread = []      # [N] horizontal spread
    yspread = []      # [N] vertical spread
    edged = []        # [N] edge density

    # ----- attribute helpers -----
    def stats(x: torch.Tensor):
        """
        x: [B,C,H,W] in [0,1]
        Returns per-sample scalars: area_frac, x_spread, y_spread, edge_density
        """
        if x.size(1) != 1:
            xg = x.mean(dim=1, keepdim=True)  # to 1ch for stats
        else:
            xg = x

        # foreground mask (dark on white bg)
        fg = (xg < 0.98).float()

        # area fraction
        area_frac = fg.mean(dim=[1, 2, 3])  # [B]

        # horizontal / vertical spread via second central moment of projections
        B, _, H, W = xg.shape
        # sums over rows/cols
        hx = fg.sum(dim=2).squeeze(1)  # [B, W]
        hy = fg.sum(dim=3).squeeze(1)  # [B, H]
        # normalize to distributions
        hx = hx / (hx.sum(dim=1, keepdim=True) + 1e-6)
        hy = hy / (hy.sum(dim=1, keepdim=True) + 1e-6)
        xs = torch.arange(W, device=x.device).float()
        ys = torch.arange(H, device=x.device).float()
        xs_c = xs - xs.mean()
        ys_c = ys - ys.mean()
        x_spread = (hx * (xs_c ** 2)).sum(dim=1)  # [B]
        y_spread = (hy * (ys_c ** 2)).sum(dim=1)  # [B]

        # edge density (simple Sobel magnitude average)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 4.0
        ky = kx.transpose(2, 3)
        gx = F.conv2d(xg, kx, padding=1)
        gy = F.conv2d(xg, ky, padding=1)
        edge_density = (gx.abs() + gy.abs()).mean(dim=[1, 2, 3])  # [B]

        return area_frac, x_spread, y_spread, edge_density

    # ----- loop -----
    n = 0
    for b in dl:
        x = b["images"].to(device)
        mu, logvar, feats = model.encode(x)

        Z.append(mu.detach().cpu().numpy())
        a, xs, ys, ed = stats(x)
        area.append(a.detach().cpu().numpy())
        xspread.append(xs.detach().cpu().numpy())
        yspread.append(ys.detach().cpu().numpy())
        edged.append(ed.detach().cpu().numpy())

        n += x.size(0)
        if n >= args.max_items:
            break

    # ----- pack arrays -----
    Z = np.concatenate(Z, axis=0)[:args.max_items]  # [N, D]
    area = np.concatenate(area, axis=0)[:len(Z)]
    xspread = np.concatenate(xspread, axis=0)[:len(Z)]
    yspread = np.concatenate(yspread, axis=0)[:len(Z)]
    edged = np.concatenate(edged, axis=0)[:len(Z)]

    attrs = {
        "area_frac": area,
        "x_spread": xspread,
        "y_spread": yspread,
        "edge_density": edged,
    }

    # ----- correlations -----
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "latent_attribute_correlations.csv"

    # normalize latent codes once (per dim)
    Zn = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-8)  # [N, D]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["attribute"] + [f"z{d}" for d in range(Z.shape[1])]
        w.writerow(header)

        for name, vals in attrs.items():
            v = _scalarize(np.asarray(vals))                   # [N]
            v = (v - v.mean()) / (v.std() + 1e-8)              # z-score
            corr = (Zn * v[:, None]).mean(axis=0)              # [D]

            # write row
            w.writerow([name] + [f"{c:.4f}" for c in corr])

            # plot top-10 dims by |corr|
            idx = np.argsort(-np.abs(corr))[:10]
            plt.figure(figsize=(8, 3), dpi=150)
            plt.bar([f"z{int(i)}" for i in idx], corr[idx])
            plt.title(f"Pearson correlation with {name}")
            plt.tight_layout()
            plt.savefig(out / f"corr_{name}.png"); plt.close()

    print(f"Saved correlations CSV → {csv_path}")
    print(f"Saved bar plots in → {out}")

if __name__ == "__main__":
    main()
