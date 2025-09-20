from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from explainability_AI.common import (
    build_loader,
    dump_metadata,
    ensure_rgb,
    load_model,
    overlay_heatmap,
    resolve_device,
    timestamped_subdir,
)

# Accelerators
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RESULTS_ROOT = Path("results/xai")
METHOD_NAME = "decoder_influence"


@torch.no_grad()
def fetch_samples(cfg, *, root: str, device: str, batch: int, count: int):
    _, loader = build_loader(
        cfg,
        root=root,
        split="val",
        batch_size=max(batch, count),
        device=device,
        shuffle=False,
        max_items=None,
        include_annotations=False,
    )
    imgs = next(iter(loader))["images"].to(device)
    return imgs[:count]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Finite-difference sensitivity of decoder outputs to latent dims")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default="./AMSL Dataset")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--dims", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device=device, img_size_override=args.img_size)
    x = fetch_samples(cfg, root=args.root, device=device, batch=8, count=args.samples)

    mu, logvar, feats = model.encode(x)
    base = model.decode(mu, feats).clamp(0, 1)

    run_dir = timestamped_subdir(Path(args.out_root), METHOD_NAME)

    dim_summaries = []
    for dim in args.dims:
        z_plus = mu.clone()
        z_minus = mu.clone()
        z_plus[:, dim] += args.delta
        z_minus[:, dim] -= args.delta
        xp = model.decode(z_plus, feats).clamp(0, 1)
        xm = model.decode(z_minus, feats).clamp(0, 1)
        diff = (xp - xm).abs() / (2.0 * args.delta)
        diff = diff / (diff.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)

        overlays = overlay_heatmap(base, diff, alpha=0.55, cmap="inferno")
        stats = diff.mean(dim=[1, 2, 3]).cpu().numpy()
        dim_summaries.append({
            "dim": int(dim),
            "mean_influence": [float(s) for s in stats],
        })

        fig, axes = plt.subplots(args.samples, 3, figsize=(9, 3 * args.samples), dpi=140)
        if args.samples == 1:
            axes = np.expand_dims(axes, axis=0)
        for row in range(args.samples):
            axes[row, 0].imshow(ensure_rgb(x[row:row+1]).squeeze(0).permute(1, 2, 0).cpu(), cmap="gray")
            axes[row, 0].set_title(f"Input #{row}")
            axes[row, 0].axis("off")

            im = axes[row, 1].imshow(diff[row, 0].cpu(), cmap="inferno", vmin=0.0, vmax=1.0)
            axes[row, 1].set_title("Sensitivity")
            axes[row, 1].axis("off")
            fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

            axes[row, 2].imshow(overlays[row].permute(1, 2, 0).cpu())
            axes[row, 2].set_title("Overlay")
            axes[row, 2].axis("off")

        fig.suptitle(f"Latent dimension z{dim}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = run_dir / f"latent_dim_{dim:02d}.png"
        plt.savefig(out_path)
        plt.close(fig)

    metadata = {
        "method": METHOD_NAME,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "img_size": cfg.img_size,
        "device": device,
        "delta": args.delta,
        "samples": args.samples,
        "dimensions": dim_summaries,
    }
    dump_metadata(run_dir / "meta.json", metadata)

    print(f"Decoder influence figures saved in {run_dir}")


if __name__ == "__main__":
    main()
