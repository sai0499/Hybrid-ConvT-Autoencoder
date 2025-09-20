# scripts/xai_counterfactuals.py
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image, make_grid
import torchvision.transforms.functional as TF

from models.hybrid_vae import HybridVAE, HybridVAEConfig
from data.amsl_quads import AMSLQuadsConfig, build_dataloader

# speed: safe TF32 paths
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -------------------- utils / I/O --------------------
def filter_cfg(raw_cfg: dict) -> HybridVAEConfig:
    from dataclasses import fields as dataclass_fields
    allowed = {f.name for f in dataclass_fields(HybridVAEConfig)}
    clean = {}
    for k, v in raw_cfg.items():
        if k in allowed:
            if k == "attn_scales" and isinstance(v, list):
                v = tuple(v)
            clean[k] = v
    return HybridVAEConfig(**clean)

def load_model(ckpt: str, img_size_override: int | None, device: str) -> tuple[HybridVAE, HybridVAEConfig, dict]:
    payload = torch.load(ckpt, map_location="cpu")
    raw_cfg = payload["cfg"] if isinstance(payload["cfg"], dict) else payload["cfg"].__dict__
    cfg = filter_cfg(raw_cfg)
    if img_size_override is not None:
        cfg.img_size = img_size_override
    model = HybridVAE(cfg).to(device).eval()
    model.load_state_dict(payload["model"], strict=False)
    return model, cfg, payload

@torch.no_grad()
def build_val_sample(cfg: HybridVAEConfig, root: str, device: str) -> torch.Tensor:
    is_windows = (os.name == "nt")
    ds_cfg = AMSLQuadsConfig(root=root, split="val", img_size=cfg.img_size,
                             grayscale=(cfg.img_channels==1), include_annotations=False)
    _, dl = build_dataloader(ds_cfg, batch_size=8, shuffle=False,
                             num_workers=0 if is_windows else 6,
                             pin_memory=(device=="cuda"), persistent_workers=False)
    x = next(iter(dl))["images"].to(device)[:1]  # single image
    return x

def load_single_image(path: str, target_size: int, channels: int, fit: str="resize") -> torch.Tensor:
    """
    Load image → [1,C,S,S] in [0,1]. 'fit' can be 'resize' or 'letterbox'.
    """
    img = Image.open(path).convert("L" if channels==1 else "RGB")
    if fit == "resize":
        img = img.resize((target_size, target_size), resample=Image.NEAREST)
    else:
        w, h = img.size
        s = min(target_size/w, target_size/h)
        nw, nh = max(1, int(round(w*s))), max(1, int(round(h*s)))
        img_r = img.resize((nw, nh), resample=Image.NEAREST)
        bg = Image.new(img.mode, (target_size, target_size), (255 if channels==1 else (255,255,255)))
        bg.paste(img_r, ((target_size-nw)//2, (target_size-nh)//2))
        img = bg
    t = TF.to_tensor(img).unsqueeze(0)
    return t

def to_vis3(t: torch.Tensor) -> torch.Tensor:
    return t.repeat(1,3,1,1) if t.size(1)==1 else t

def save_grid(imgs: list[torch.Tensor], path: Path, nrow: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(make_grid(torch.cat(imgs, dim=0), nrow=nrow, padding=2), str(path))


# -------------------- differentiable attributes --------------------
def attr_area(x: torch.Tensor) -> torch.Tensor:
    """
    Foreground area fraction for black-on-white shapes.
    x: [B,C,H,W] in [0,1]. Return [B].
    """
    xg = x.mean(dim=1, keepdim=True) if x.size(1) != 1 else x
    return 1.0 - xg.mean(dim=[1,2,3])

def attr_spread(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Horizontal and vertical spread via 2nd central moments of a soft foreground.
    Return (x_spread [B], y_spread [B]).
    """
    xg = x.mean(dim=1, keepdim=True) if x.size(1) != 1 else x
    fg = 1.0 - xg  # soft foreground in [0,1]
    B, _, H, W = xg.shape
    hx = fg.sum(dim=2).squeeze(1)  # [B,W]
    hy = fg.sum(dim=3).squeeze(1)  # [B,H]
    hx = hx / (hx.sum(dim=1, keepdim=True) + 1e-6)
    hy = hy / (hy.sum(dim=1, keepdim=True) + 1e-6)
    xs = torch.arange(W, device=x.device, dtype=x.dtype); xs_c = xs - xs.mean()
    ys = torch.arange(H, device=x.device, dtype=x.dtype); ys_c = ys - ys.mean()
    x_spread = (hx * (xs_c**2)).sum(dim=1)
    y_spread = (hy * (ys_c**2)).sum(dim=1)
    return x_spread, y_spread


# -------------------- counterfactuals --------------------
@torch.no_grad()
def nudge_dims(model: HybridVAE, x: torch.Tensor, dims: list[int], delta: float) -> list[torch.Tensor]:
    """
    Encode x, nudge z=mu by ±delta on listed dims (one panel per dim).
    Returns list of tensors [B,C,H,W] for concatenation/visualization.
    """
    mu, logvar, feats = model.encode(x)
    base = model.decode(mu, feats).clamp(0,1)

    panels = [to_vis3(x), to_vis3(base)]
    for d in dims:
        zp = mu.clone(); zm = mu.clone()
        zp[:, d] += delta
        zm[:, d] -= delta
        xp = model.decode(zp, feats).clamp(0,1)
        xm = model.decode(zm, feats).clamp(0,1)
        panels.extend([to_vis3(xm), to_vis3(xp)])
    return panels

def optimize_dims(model: HybridVAE, x: torch.Tensor, dims: list[int],
                  attr: str, target: float, steps: int=200, lr: float=0.05, lam: float=0.01,
                  grad_clip: float = 1.0) -> tuple[torch.Tensor, float]:
    """
    Keep skip features fixed; optimize only selected dims in z to hit a target attribute.
    attr ∈ {'area','xspread','yspread'}
    Returns (x_cf, achieved_attr_value)
    """
    # --- Get encoder outputs without building a graph; detach feats! ---
    with torch.no_grad():
        mu, logvar, feats = model.encode(x)
    z0 = mu.detach()                              # [1, D]
    feats = {k: v.detach() for k, v in feats.items()}  # detach skips

    # Optimize z as a learnable parameter (only selected dims will be used)
    z = nn.Parameter(z0.clone())                  # leaf Param

    # mask to update only chosen dims
    D = z0.size(1)
    mask = torch.zeros(D, device=z.device)
    for d in dims:
        if 0 <= d < D:
            mask[d] = 1.0

    opt = torch.optim.Adam([z], lr=lr)

    for t in range(steps):
        # Build z_eff that only differs on selected dims
        z_eff = z0 + (z - z0) * mask  # [1, D]

        xr = model.decode(z_eff, feats).clamp(0,1)  # decode depends on z only

        # Attribute loss
        if attr == "area":
            a = attr_area(xr)                       # [1]
            loss_attr = (a - target)**2
        elif attr == "xspread":
            xs, ys = attr_spread(xr)
            loss_attr = (xs - target)**2
        elif attr == "yspread":
            xs, ys = attr_spread(xr)
            loss_attr = (ys - target)**2
        else:
            raise ValueError("attr must be one of: area, xspread, yspread")

        # Regularize z to stay near z0 on selected dims
        loss_reg = ((z - z0)**2 * mask).mean()

        loss = loss_attr.mean() + lam * loss_reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([z], grad_clip)
        opt.step()

    # Final decode & measurement
    with torch.no_grad():
        z_eff = z0 + (z - z0) * mask
        xr = model.decode(z_eff, feats).clamp(0,1)
        if attr == "area":
            value = float(attr_area(xr).mean().item())
        elif attr == "xspread":
            value = float(attr_spread(xr)[0].mean().item())
        else:
            value = float(attr_spread(xr)[1].mean().item())

    return xr, value


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Counterfactual editing for HybridVAE")
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--root", default="./AMSL Dataset", type=str)
    ap.add_argument("--img_size", type=int, default=None, help="override size if desired")
    ap.add_argument("--input", type=str, default=None, help="optional path to a single image")
    ap.add_argument("--fit", choices=["resize","letterbox"], default="resize")

    sub = ap.add_subparsers(dest="mode", required=True)

    # nudge mode
    nud = sub.add_parser("nudge", help="Nudge specific latent dims by ±delta")
    nud.add_argument("--dims", type=int, nargs="+", default=[0,1,2])
    nud.add_argument("--delta", type=float, default=1.0)
    nud.add_argument("--out", type=str, default="results/xai/cf_nudge.png")

    # optimize mode
    optp = sub.add_parser("optimize", help="Optimize dims to hit a target attribute")
    optp.add_argument("--dims", type=int, nargs="+", default=[0])
    optp.add_argument("--attr", choices=["area","xspread","yspread"], default="area")
    optp.add_argument("--target", type=float, required=True, help="Target value for attribute")
    optp.add_argument("--steps", type=int, default=200)
    optp.add_argument("--lr", type=float, default=0.05)
    optp.add_argument("--lam", type=float, default=0.01)
    optp.add_argument("--out", type=str, default="results/xai/cf_opt.png")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_model(args.ckpt, args.img_size, device)

    # pick input image
    if args.input:
        x = load_single_image(args.input, cfg.img_size, cfg.img_channels, fit=args.fit).to(device)
    else:
        x = build_val_sample(cfg, args.root, device)

    if args.mode == "nudge":
        panels = nudge_dims(model, x, args.dims, args.delta)
        # layout: [input, recon, (-Δ for z_d, +Δ for z_d) ...]
        nrow = 2 + 2*len(args.dims)
        save_grid(panels, Path(args.out), nrow=nrow)
        print(f"Saved counterfactual nudge grid → {args.out}")

    else:  # optimize
        # base recon and stats (for reference)
        with torch.no_grad():
            mu, logvar, feats = model.encode(x)
            base = model.decode(mu, feats).clamp(0,1)
            base_area = float(attr_area(base).mean().item())
            base_xs, base_ys = attr_spread(base)
            base_xs = float(base_xs.mean().item())
            base_ys = float(base_ys.mean().item())

        xr, val = optimize_dims(model, x, args.dims, args.attr, args.target,
                                steps=args.steps, lr=args.lr, lam=args.lam)

        # report
        print(f"Base: area={base_area:.4f}  xspread={base_xs:.4f}  yspread={base_ys:.4f}")
        print(f"Edited ({args.attr}→{args.target}): {args.attr}={val:.4f}")

        # visuals: input | base | edited | diff
        diff = (xr - base).abs()
        grid = [to_vis3(x), to_vis3(base), to_vis3(xr), to_vis3(diff / (diff.amax(dim=[1,2,3], keepdim=True)+1e-8))]
        save_grid(grid, Path(args.out), nrow=4)
        print(f"Saved counterfactual optimize grid → {args.out}")

if __name__ == "__main__":
    main()
