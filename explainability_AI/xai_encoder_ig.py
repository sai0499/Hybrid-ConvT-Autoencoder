from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

import argparse
import torch
from torchvision.utils import save_image, make_grid

from models.hybrid_vae import HybridVAE, HybridVAEConfig
from data.amsl_quads import AMSLQuadsConfig, build_dataloader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def overlay_heatmap(x: torch.Tensor, hm: torch.Tensor, alpha: float = 0.45, cmap: str = "jet"):
    """
    x: [B,1,H,W] or [B,3,H,W] in [0,1]
    hm: [B,1,H,W] in [0,1]  (normalized heatmap)
    returns [B,3,H,W] RGB overlays
    """
    if x.size(1) == 1:
        x_rgb = x.repeat(1,3,1,1)
    else:
        x_rgb = x
    overlays = []
    for i in range(x.size(0)):
        base = (x_rgb[i].permute(1,2,0).cpu().numpy())  # H,W,3
        h = hm[i,0].cpu().numpy()                       # H,W
        cm = plt.get_cmap(cmap)(h)[..., :3]             # H,W,3 in [0,1]
        out = (1-alpha)*base + alpha*cm
        overlays.append(torch.from_numpy(out).permute(2,0,1))
    return torch.stack(overlays, 0).clamp(0, 1).to(x.device)

def filter_cfg(raw_cfg: dict) -> HybridVAEConfig:
    from dataclasses import fields as dataclass_fields
    allowed = {f.name for f in dataclass_fields(HybridVAEConfig)}
    clean = {}
    for k, v in raw_cfg.items():
        if k in allowed:
            if k == "attn_scales" and isinstance(v, list): v = tuple(v)
            clean[k] = v
    return HybridVAEConfig(**clean)

def to_vis3(t: torch.Tensor) -> torch.Tensor:
    return t.repeat(1,3,1,1) if t.size(1)==1 else t

def load_model(ckpt: str, img_size_override: int | None, device: str) -> tuple[HybridVAE, HybridVAEConfig, dict]:
    payload = torch.load(ckpt, map_location="cpu")
    raw_cfg = payload["cfg"] if isinstance(payload["cfg"], dict) else payload["cfg"].__dict__
    cfg = filter_cfg(raw_cfg)
    if img_size_override is not None:
        cfg.img_size = img_size_override  # smaller size = cheaper attention
    model = HybridVAE(cfg).to(device).eval()
    model.load_state_dict(payload["model"], strict=False)
    return model, cfg, payload

@torch.no_grad()
def get_val_batch(cfg: HybridVAEConfig, root: str, batch: int, device: str) -> torch.Tensor:
    is_windows = (os.name == "nt")
    ds_cfg = AMSLQuadsConfig(root=root, split="val", img_size=cfg.img_size,
                             grayscale=(cfg.img_channels==1), include_annotations=False)
    _, dl = build_dataloader(ds_cfg, batch_size=batch, shuffle=False,
                             num_workers=0 if is_windows else 6,
                             pin_memory=(device=="cuda"), persistent_workers=False)
    return next(iter(dl))["images"].to(device)

def run_ig(model: HybridVAE, x: torch.Tensor, target_dim: int,
           n_steps: int, internal_bs: int, baseline_kind: str) -> torch.Tensor:
    from captum.attr import IntegratedGradients

    x = x.detach().clone().requires_grad_(True)

    if baseline_kind == "white":
        baseline = torch.ones_like(x)
    elif baseline_kind == "black":
        baseline = torch.zeros_like(x)
    else:
        baseline = x.mean(dim=[2,3], keepdim=True).expand_as(x)

    def forward_mu(inp: torch.Tensor) -> torch.Tensor:
        mu, logvar, feats = model.encode(inp)
        return mu[:, target_dim]  # [B]

    ig = IntegratedGradients(forward_mu)
    attributions = ig.attribute(x, baselines=baseline, n_steps=n_steps,
                                internal_batch_size=internal_bs)  # [B,C,H,W]
    return attributions

def main():
    ap = argparse.ArgumentParser(description="Integrated Gradients on encoder μ[d] wrt input")
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--root", default="./AMSL Dataset", type=str)
    ap.add_argument("--img_size", type=int, default=256, help="use a SMALLER size for IG (e.g., 256)")
    ap.add_argument("--samples", type=int, default=2, help="fewer samples reduce VRAM")

    ap.add_argument("--target_dim", type=int, default=0, help="μ dimension to attribute")
    ap.add_argument("--n_steps", type=int, default=32, help="integration steps (32 is plenty)")
    ap.add_argument("--internal_bs", type=int, default=1, help="chunk IG path for memory")
    ap.add_argument("--baseline", choices=["white","black","mean"], default="white")
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--out_dir", type=str, default="results/xai")
    args = ap.parse_args()

    # device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # load model at downsized resolution for cheaper attention
    model, cfg, _ = load_model(args.ckpt, args.img_size, device)

    # small batch
    with torch.no_grad():
        x = get_val_batch(cfg, args.root, max(4, args.samples), device)[:args.samples]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # try IG on selected device; if CUDA OOM, fall back to CPU automatically
    try:
        attributions = run_ig(model, x, args.target_dim, args.n_steps, args.internal_bs, args.baseline)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and device == "cuda":
            print("[IG] CUDA OOM → falling back to CPU for attribution...")
            torch.cuda.empty_cache()
            device = "cpu"
            model_cpu, cfg_cpu, _ = load_model(args.ckpt, args.img_size, device="cpu")
            x_cpu = x.detach().cpu()
            attributions = run_ig(model_cpu, x_cpu, args.target_dim, args.n_steps, args.internal_bs, args.baseline)
            x = x_cpu
            model = model_cpu
        else:
            raise

    # normalize for visualization
    with torch.no_grad():
        att = attributions.abs()
        att = att / (att.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
        overlay = overlay_heatmap(x, att, alpha=0.45, cmap="jet")
        grid = torch.cat([to_vis3(x), to_vis3(att), overlay], dim=0)
        save_image(make_grid(grid, nrow=x.size(0), padding=2),
                   str(out_dir / f"ig_encoder_mu_dim{args.target_dim}.png"))
        print("Saved:", out_dir / f"ig_encoder_mu_dim{args.target_dim}.png")

if __name__ == "__main__":
    main()
