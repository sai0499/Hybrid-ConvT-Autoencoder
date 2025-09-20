from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from explainability_AI.common import (
    build_loader,
    dump_metadata,
    load_model,
    resolve_device,
    timestamped_subdir,
)

# Enable TF32 for speed on supported GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RESULTS_ROOT = Path("results/xai")
METHOD_NAME = "attr_correlation"


def _scalarize(a: np.ndarray) -> np.ndarray:
    """Flatten per-sample tensors and average so every entry is 1D."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a
    return a.reshape(a.shape[0], -1).mean(axis=1)


def _attribute_stats(x: torch.Tensor):
    """Return area fraction, horizontal spread, vertical spread, edge density."""
    if x.size(1) != 1:
        xg = x.mean(dim=1, keepdim=True)
    else:
        xg = x

    fg = (xg < 0.98).float()
    area_frac = fg.mean(dim=[1, 2, 3])

    bsz, _, h, w = xg.shape
    hx = fg.sum(dim=2).squeeze(1)
    hy = fg.sum(dim=3).squeeze(1)
    hx = hx / (hx.sum(dim=1, keepdim=True) + 1e-6)
    hy = hy / (hy.sum(dim=1, keepdim=True) + 1e-6)

    xs = torch.arange(w, device=x.device).float()
    ys = torch.arange(h, device=x.device).float()
    xs_c = xs - xs.mean()
    ys_c = ys - ys.mean()
    x_spread = (hx * (xs_c ** 2)).sum(dim=1)
    y_spread = (hy * (ys_c ** 2)).sum(dim=1)

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 4.0
    ky = kx.transpose(2, 3)
    gx = F.conv2d(xg, kx, padding=1)
    gy = F.conv2d(xg, ky, padding=1)
    edge_density = (gx.abs() + gy.abs()).mean(dim=[1, 2, 3])

    return area_frac, x_spread, y_spread, edge_density


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Quantify correlation between latent codes and shape attributes")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--root", default="./AMSL Dataset", help="Dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--img_size", type=int, default=None, help="Override input size (falls back to checkpoint)")
    parser.add_argument("--max_items", type=int, default=5000, help="Limit number of samples encoded")
    parser.add_argument("--batch", type=int, default=128, help="Encoding batch size")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out_root", default=str(RESULTS_ROOT), help="Base directory for artefacts")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device=device, img_size_override=args.img_size)

    _, loader = build_loader(
        cfg,
        root=args.root,
        split=args.split,
        batch_size=args.batch,
        device=device,
        shuffle=False,
        max_items=args.max_items,
        include_annotations=False,
    )

    latents, area, xspread, yspread, edged = [], [], [], [], []
    seen = 0
    for batch in loader:
        x = batch["images"].to(device)
        mu, logvar, feats = model.encode(x)
        latents.append(mu.detach().cpu().numpy())
        a, xs, ys, ed = _attribute_stats(x)
        area.append(a.cpu().numpy())
        xspread.append(xs.cpu().numpy())
        yspread.append(ys.cpu().numpy())
        edged.append(ed.cpu().numpy())
        seen += x.size(0)
        if seen >= args.max_items:
            break

    latents_np = np.concatenate(latents, axis=0)[: args.max_items]
    attr_arrays = {
        "area_frac": np.concatenate(area, axis=0)[: len(latents_np)],
        "x_spread": np.concatenate(xspread, axis=0)[: len(latents_np)],
        "y_spread": np.concatenate(yspread, axis=0)[: len(latents_np)],
        "edge_density": np.concatenate(edged, axis=0)[: len(latents_np)],
    }

    out_root = Path(args.out_root)
    run_dir = timestamped_subdir(out_root, METHOD_NAME)
    csv_path = run_dir / "latent_attribute_correlations.csv"

    zn = (latents_np - latents_np.mean(axis=0)) / (latents_np.std(axis=0) + 1e-8)

    top_corr = {}
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["attribute"] + [f"z{d}" for d in range(latents_np.shape[1])])

        for name, vals in attr_arrays.items():
            v = _scalarize(vals)
            v = (v - v.mean()) / (v.std() + 1e-8)
            corr = (zn * v[:, None]).mean(axis=0)
            writer.writerow([name] + [f"{c:.4f}" for c in corr])

            order = np.argsort(-np.abs(corr))[:10]
            top_corr[name] = [{"dim": int(idx), "corr": float(corr[idx])} for idx in order]

            plt.figure(figsize=(9, 4), dpi=150)
            bars = [f"z{int(i)}" for i in order]
            values = corr[order]
            plt.barh(bars[::-1], values[::-1], color="#3366cc")
            plt.xlabel("Pearson correlation")
            plt.title(f"Latent dimensions most correlated with {name}")
            plt.grid(axis="x", linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(run_dir / f"{name}_top10.png")
            plt.close()

    metadata = {
        "method": METHOD_NAME,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "split": args.split,
        "img_size": cfg.img_size,
        "latent_dim": cfg.latent_dim,
        "samples_used": int(len(latents_np)),
        "batch": args.batch,
        "top_correlations": top_corr,
    }
    dump_metadata(run_dir / "meta.json", metadata)

    print(f"Correlation table saved to {csv_path}")
    print(f"Individual bar charts and metadata available in {run_dir}")


if __name__ == "__main__":
    main()
