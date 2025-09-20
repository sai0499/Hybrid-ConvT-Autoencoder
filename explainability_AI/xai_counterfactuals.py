from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from typing import List

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms.functional as TF

from explainability_AI.common import (
    build_loader,
    dump_metadata,
    ensure_rgb,
    load_model,
    overlay_heatmap,
    resolve_device,
    save_grid,
    timestamped_subdir,
)

# Speed toggles on NVIDIA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RESULTS_ROOT = Path("results/xai")
NUDGE_METHOD = "counterfactual_nudge"
OPT_METHOD = "counterfactual_optimize"


def load_single_image(path: str, target_size: int, channels: int, fit: str = "resize") -> torch.Tensor:
    img = Image.open(path).convert("L" if channels == 1 else "RGB")
    if fit == "resize":
        img = img.resize((target_size, target_size), resample=Image.NEAREST)
    else:
        w, h = img.size
        scale = min(target_size / w, target_size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img_resized = img.resize((nw, nh), resample=Image.NEAREST)
        bg = Image.new(img.mode, (target_size, target_size), (255 if channels == 1 else (255, 255, 255)))
        bg.paste(img_resized, ((target_size - nw) // 2, (target_size - nh) // 2))
        img = bg
    return TF.to_tensor(img).unsqueeze(0)


@torch.no_grad()
def fetch_val_sample(cfg, *, root: str, device: str) -> torch.Tensor:
    _, loader = build_loader(
        cfg,
        root=root,
        split="val",
        batch_size=8,
        device=device,
        shuffle=False,
        max_items=None,
        include_annotations=False,
    )
    return next(iter(loader))["images"].to(device)[:1]


# -------------------- differentiable attributes --------------------
def attr_area(x: torch.Tensor) -> torch.Tensor:
    xg = x.mean(dim=1, keepdim=True) if x.size(1) != 1 else x
    return 1.0 - xg.mean(dim=[1, 2, 3])


def attr_spread(x: torch.Tensor):
    xg = x.mean(dim=1, keepdim=True) if x.size(1) != 1 else x
    fg = 1.0 - xg
    _, _, h, w = xg.shape
    hx = fg.sum(dim=2).squeeze(1)
    hy = fg.sum(dim=3).squeeze(1)
    hx = hx / (hx.sum(dim=1, keepdim=True) + 1e-6)
    hy = hy / (hy.sum(dim=1, keepdim=True) + 1e-6)
    xs = torch.arange(w, device=x.device, dtype=x.dtype)
    ys = torch.arange(h, device=x.device, dtype=x.dtype)
    xs_c = xs - xs.mean()
    ys_c = ys - ys.mean()
    x_spread = (hx * (xs_c ** 2)).sum(dim=1)
    y_spread = (hy * (ys_c ** 2)).sum(dim=1)
    return x_spread, y_spread


# -------------------- counterfactual helpers --------------------
@torch.no_grad()
def latent_traversal(model, x: torch.Tensor, dims: List[int], delta: float):
    mu, logvar, feats = model.encode(x)
    base = model.decode(mu, feats).clamp(0, 1)
    results = []
    for dim in dims:
        zp = mu.clone()
        zm = mu.clone()
        zp[:, dim] += delta
        zm[:, dim] -= delta
        xp = model.decode(zp, feats).clamp(0, 1)
        xm = model.decode(zm, feats).clamp(0, 1)
        influence = (xp - xm).abs() / (2.0 * delta)
        influence = influence / (influence.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
        overlay = overlay_heatmap(base, influence, alpha=0.55, cmap="inferno")
        results.append({
            "dim": int(dim),
            "minus": xm,
            "plus": xp,
            "overlay": overlay,
            "influence": influence,
        })
    return base, results


def optimize_dims(model: torch.nn.Module, x: torch.Tensor, dims: List[int], attr: str, target: float, *, steps: int, lr: float, lam: float, grad_clip: float | None = None):
    mu, logvar, feats = model.encode(x)
    z0 = mu.detach()
    z = nn.Parameter(z0.clone())
    mask = torch.zeros(z0.size(1), device=z.device)
    for d in dims:
        if 0 <= d < z0.size(1):
            mask[d] = 1.0

    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        z_eff = z0 + (z - z0) * mask
        xr = model.decode(z_eff, feats).clamp(0, 1)
        if attr == "area":
            attr_loss = (attr_area(xr) - target) ** 2
        elif attr == "xspread":
            xs, _ = attr_spread(xr)
            attr_loss = (xs - target) ** 2
        elif attr == "yspread":
            _, ys = attr_spread(xr)
            attr_loss = (ys - target) ** 2
        else:
            raise ValueError("attr must be one of {'area','xspread','yspread'}")
        reg = ((z - z0) ** 2 * mask).mean()
        loss = attr_loss.mean() + lam * reg
        opt.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([z], grad_clip)
        opt.step()

    with torch.no_grad():
        z_eff = z0 + (z - z0) * mask
        xr = model.decode(z_eff, feats).clamp(0, 1)
        if attr == "area":
            value = float(attr_area(xr).mean().item())
        elif attr == "xspread":
            value = float(attr_spread(xr)[0].mean().item())
        else:
            value = float(attr_spread(xr)[1].mean().item())
    return xr, value


def main():
    parser = argparse.ArgumentParser(description="Counterfactual explainability for the HybridVAE")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default="./AMSL Dataset")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--input", type=str, default=None, help="Optional external image")
    parser.add_argument("--fit", choices=["resize", "letterbox"], default="resize")

    sub = parser.add_subparsers(dest="mode", required=True)

    nud = sub.add_parser("nudge", help="Traverse latent dims with ±delta")
    nud.add_argument("--dims", type=int, nargs="+", default=[0, 1, 2])
    nud.add_argument("--delta", type=float, default=1.0)
    nud.add_argument("--out_root", default=str(RESULTS_ROOT))

    optp = sub.add_parser("optimize", help="Optimise latent dims towards a target attribute")
    optp.add_argument("--dims", type=int, nargs="+", default=[0])
    optp.add_argument("--attr", choices=["area", "xspread", "yspread"], default="area")
    optp.add_argument("--target", type=float, required=True)
    optp.add_argument("--steps", type=int, default=200)
    optp.add_argument("--lr", type=float, default=0.05)
    optp.add_argument("--lam", type=float, default=0.01)
    optp.add_argument("--out_root", default=str(RESULTS_ROOT))

    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device=device, img_size_override=args.img_size)

    if args.input:
        sample = load_single_image(args.input, cfg.img_size, cfg.img_channels, fit=args.fit).to(device)
    else:
        sample = fetch_val_sample(cfg, root=args.root, device=device)

    if args.mode == "nudge":
        base, traversals = latent_traversal(model, sample, args.dims, args.delta)
        run_dir = timestamped_subdir(Path(args.out_root), NUDGE_METHOD)

        for entry in traversals:
            dim = entry["dim"]
            fig, axes = plt.subplots(1, 5, figsize=(15, 3), dpi=140)
            axes[0].imshow(ensure_rgb(sample).squeeze(0).permute(1, 2, 0).cpu(), cmap="gray")
            axes[0].set_title("Input")
            axes[0].axis("off")

            axes[1].imshow(ensure_rgb(base).squeeze(0).permute(1, 2, 0).cpu())
            axes[1].set_title("Base recon")
            axes[1].axis("off")

            axes[2].imshow(ensure_rgb(entry["minus"]).squeeze(0).permute(1, 2, 0).cpu())
            axes[2].set_title(f"z{dim} - Δ")
            axes[2].axis("off")

            axes[3].imshow(ensure_rgb(entry["plus"]).squeeze(0).permute(1, 2, 0).cpu())
            axes[3].set_title(f"z{dim} + Δ")
            axes[3].axis("off")

            im = axes[4].imshow(entry["influence"][0, 0].cpu(), cmap="inferno", vmin=0.0, vmax=1.0)
            axes[4].set_title("Sensitivity")
            axes[4].axis("off")
            plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

            plt.suptitle(f"Latent traversal for dimension z{dim}")
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            plt.savefig(run_dir / f"nudge_z{dim:02d}.png")
            plt.close(fig)

        dump_metadata(run_dir / "meta.json", {
            "method": NUDGE_METHOD,
            "checkpoint": str(Path(args.ckpt).resolve()),
            "img_size": cfg.img_size,
            "delta": args.delta,
            "dims": args.dims,
            "input_path": str(Path(args.input).resolve()) if args.input else None,
        })
        print(f"Nudge visualisations saved to {run_dir}")

    else:
        with torch.no_grad():
            mu0, logvar0, feats0 = model.encode(sample)
            base = model.decode(mu0, feats0).clamp(0, 1)
            base_area = float(attr_area(base).item())
            base_xs, base_ys = attr_spread(base)
            base_stats = {
                "area": base_area,
                "xspread": float(base_xs.item()),
                "yspread": float(base_ys.item()),
            }

        edited, achieved = optimize_dims(
            model,
            sample,
            args.dims,
            args.attr,
            args.target,
            steps=args.steps,
            lr=args.lr,
            lam=args.lam,
        )

        diff = (edited - base).abs()
        diff_norm = diff / (diff.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)

        run_dir = timestamped_subdir(Path(args.out_root), OPT_METHOD)
        save_grid(
            [ensure_rgb(sample), ensure_rgb(base), ensure_rgb(edited), ensure_rgb(diff_norm)],
            run_dir / "counterfactual_grid.png",
            nrow=1,
        )

        dump_metadata(run_dir / "meta.json", {
            "method": OPT_METHOD,
            "checkpoint": str(Path(args.ckpt).resolve()),
            "img_size": cfg.img_size,
            "dims": args.dims,
            "attr": args.attr,
            "target": args.target,
            "achieved": achieved,
            "base_stats": base_stats,
            "steps": args.steps,
            "lr": args.lr,
            "lam": args.lam,
            "input_path": str(Path(args.input).resolve()) if args.input else None,
        })
        print(f"Optimisation results saved to {run_dir}")


if __name__ == "__main__":
    main()
