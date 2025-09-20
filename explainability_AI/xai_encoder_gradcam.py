from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import torch
from captum.attr import LayerGradCam, LayerAttribution
from torchvision.utils import make_grid, save_image

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
METHOD_NAME = "encoder_gradcam"


def build_forward_fn(model: torch.nn.Module, target_dim: int):
    def forward_mu(inp: torch.Tensor) -> torch.Tensor:
        mu, logvar, feats = model.encode(inp)
        return mu[:, target_dim]
    return forward_mu


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
    parser = argparse.ArgumentParser(description="Layer-wise Grad-CAM on encoder latent units")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default="./AMSL Dataset")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--target_dim", type=int, default=0, help="Latent dimension to analyse")
    parser.add_argument("--layer", type=str, default="encoder.res_skip", help="Module path for Grad-CAM")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device=device, img_size_override=args.img_size)

    try:
        layer = model
        for attr in args.layer.split('.'):
            layer = getattr(layer, attr)
    except AttributeError as exc:
        raise ValueError(f"Layer path '{args.layer}' not found on model") from exc

    inputs = fetch_samples(cfg, root=args.root, device=device, batch=8, count=args.samples)
    inputs.requires_grad_(True)

    forward_fn = build_forward_fn(model, args.target_dim)
    gradcam = LayerGradCam(forward_fn, layer)

    attributions = gradcam.attribute(inputs, relu_attributions=True)
    upsampled = LayerAttribution.interpolate(attributions, inputs.shape[2:])
    cam = upsampled.mean(dim=1, keepdim=True)
    cam = cam / (cam.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
    cam = cam.detach()
    overlays = overlay_heatmap(inputs.detach(), cam, alpha=0.55, cmap="inferno")

    run_dir = timestamped_subdir(Path(args.out_root), METHOD_NAME)

    for idx in range(inputs.size(0)):
        panels = torch.stack([
            ensure_rgb(inputs[idx:idx+1]).squeeze(0).detach().cpu(),
            ensure_rgb(cam[idx:idx+1]).squeeze(0).detach().cpu(),
            overlays[idx].detach().cpu(),
        ], dim=0)
        grid = make_grid(panels, nrow=3, padding=4)
        save_image(grid, str(run_dir / f"sample_{idx:02d}.png"))

    metadata = {
        "method": METHOD_NAME,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "img_size": cfg.img_size,
        "device": device,
        "target_dim": args.target_dim,
        "layer": args.layer,
        "samples": inputs.size(0),
        "files": [f"sample_{i:02d}.png" for i in range(inputs.size(0))],
    }
    dump_metadata(run_dir / "meta.json", metadata)

    print(f"Grad-CAM visualisations saved in {run_dir}")


if __name__ == "__main__":
    main()
