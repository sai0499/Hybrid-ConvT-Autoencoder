from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import torch
from torchvision.utils import save_image, make_grid

from models.hybrid_vae import HybridVAE, HybridVAEConfig
from data.amsl_quads import AMSLQuadsConfig, build_dataloader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def filter_cfg(raw_cfg: dict) -> HybridVAEConfig:
    from dataclasses import fields as dataclass_fields
    allowed = {f.name for f in dataclass_fields(HybridVAEConfig)}
    clean = {}
    for k, v in raw_cfg.items():
        if k in allowed:
            if k == "attn_scales" and isinstance(v, list): v = tuple(v)
            clean[k] = v
    return HybridVAEConfig(**clean)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="./AMSL Dataset")
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--dims", type=int, nargs="*", default=[0,1,2,3], help="latent dims to visualize")
    ap.add_argument("--delta", type=float, default=1.0, help="finite diff step on z")
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.ckpt, map_location="cpu")
    raw_cfg = payload["cfg"] if isinstance(payload["cfg"], dict) else payload["cfg"].__dict__
    cfg = filter_cfg(raw_cfg)
    if args.img_size is not None: cfg.img_size = args.img_size

    model = HybridVAE(cfg).to(device).eval()
    model.load_state_dict(payload["model"], strict=False)

    # one small val batch
    is_windows = (os.name == "nt")
    ds_cfg = AMSLQuadsConfig(root=args.root, split="val", img_size=cfg.img_size,
                             grayscale=(cfg.img_channels==1), include_annotations=False)
    _, dl = build_dataloader(ds_cfg, batch_size=max(8, args.samples), shuffle=False,
                             num_workers=0 if is_windows else 6,
                             pin_memory=(device=="cuda"), persistent_workers=False)
    x = next(iter(dl))["images"].to(device)[:args.samples]
    mu, logvar, feats = model.encode(x)
    z0 = mu

    # base recon (for reference)
    base = model.decode(z0, feats).clamp(0,1)

    maps = []
    for d in args.dims:
        z_plus = z0.clone();  z_minus = z0.clone()
        z_plus[:, d]  += args.delta
        z_minus[:, d] -= args.delta
        x_plus  = model.decode(z_plus,  feats).clamp(0,1)
        x_minus = model.decode(z_minus, feats).clamp(0,1)
        # central diff magnitude per pixel
        infl = (x_plus - x_minus).abs() / (2.0 * args.delta)
        # normalize per sample for visibility
        infl = infl / (infl.amax(dim=[1,2,3], keepdim=True) + 1e-8)
        maps.append(infl)

    maps = torch.cat(maps, dim=0)
    # stack ref inputs, base recon, and influence maps
    vis = torch.cat([x, base, maps], dim=0)

    out = Path("results/xai"); out.mkdir(parents=True, exist_ok=True)
    save_image(make_grid(vis.repeat(1,3,1,1) if vis.size(1)==1 else vis, nrow=args.samples, padding=2),
               str(out / f"decoder_influence_dims_{'_'.join(map(str,args.dims))}.png"))
    print("Saved:", out / f"decoder_influence_dims_{'_'.join(map(str,args.dims))}.png")

if __name__ == "__main__":
    main()
