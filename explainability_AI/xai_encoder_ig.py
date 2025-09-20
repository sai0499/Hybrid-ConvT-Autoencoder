from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
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

# Speed toggles
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RESULTS_ROOT = Path("results/xai")
METHOD_NAME = "encoder_integrated_gradients"


def run_ig(model: torch.nn.Module, x: torch.Tensor, target_dim: int, n_steps: int, internal_bs: int, baseline: str) -> torch.Tensor:
    from captum.attr import IntegratedGradients

    x = x.detach().clone().requires_grad_(True)
    if baseline == "white":
        ref = torch.ones_like(x)
    elif baseline == "black":
        ref = torch.zeros_like(x)
    else:
        ref = x.mean(dim=[2, 3], keepdim=True).expand_as(x)

    def forward_latent(inp: torch.Tensor) -> torch.Tensor:
        mu, logvar, feats = model.encode(inp)
        return mu[:, target_dim]

    ig = IntegratedGradients(forward_latent)
    return ig.attribute(x, baselines=ref, n_steps=n_steps, internal_batch_size=internal_bs)


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
    images = next(iter(loader))["images"].to(device)
    return images[:count]


def main():
    parser = argparse.ArgumentParser(description="Integrated Gradients for encoder latent units")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default="./AMSL Dataset")
    parser.add_argument("--img_size", type=int, default=256, help="Smaller size reduces IG cost")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--target_dim", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--internal_bs", type=int, default=1)
    parser.add_argument("--baseline", choices=["white", "black", "mean"], default="white")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device=device, img_size_override=args.img_size)
    inputs = fetch_samples(cfg, root=args.root, device=device, batch=8, count=args.samples)

    run_dir = timestamped_subdir(Path(args.out_root), METHOD_NAME)

    try:
        attributions = run_ig(model, inputs, args.target_dim, args.n_steps, args.internal_bs, args.baseline)
    except RuntimeError as exc:
        if "CUDA out of memory" in str(exc) and device == "cuda":
            print("[IG] CUDA OOM – retrying on CPU...")
            torch.cuda.empty_cache()
            device = "cpu"
            model, cfg, _ = load_model(args.ckpt, device="cpu", img_size_override=args.img_size)
            inputs = inputs.cpu()
            attributions = run_ig(model, inputs, args.target_dim, args.n_steps, args.internal_bs, args.baseline)
        else:
            raise

    with torch.no_grad():
        mag = attributions.abs()
        mag = mag / (mag.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
        overlays = overlay_heatmap(inputs, mag, alpha=0.5, cmap="magma")

    sample_files = []
    for idx in range(inputs.size(0)):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=160)
        axes[0].imshow(ensure_rgb(inputs[idx:idx+1]).squeeze(0).permute(1, 2, 0).cpu(), cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")

        im = axes[1].imshow(mag[idx, 0].cpu(), cmap="magma", vmin=0.0, vmax=1.0)
        axes[1].set_title("Attribution")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlays[idx].permute(1, 2, 0).cpu())
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = run_dir / f"sample_{idx:02d}.png"
        plt.savefig(out_path)
        plt.close(fig)
        sample_files.append(out_path.name)

    summary = {
        "method": METHOD_NAME,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "img_size": cfg.img_size,
        "device": device,
        "target_dim": args.target_dim,
        "samples": inputs.size(0),
        "n_steps": args.n_steps,
        "baseline": args.baseline,
        "sample_images": sample_files,
    }
    dump_metadata(run_dir / "meta.json", summary)

    print(f"Integrated Gradients visualisations saved in {run_dir}")


if __name__ == "__main__":
    main()
