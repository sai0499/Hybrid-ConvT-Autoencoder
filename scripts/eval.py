from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.utils import save_image, make_grid
from pytorch_msssim import ms_ssim as ms_ssim_fn

from data.amsl_quads import AMSLQuadsConfig, build_dataloader
from models.hybrid_vae import HybridVAE, HybridVAEConfig

# speed (safe): TF32 for conv/matmul
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def psnr(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3]) + eps
    return 10.0 * torch.log10(1.0 / mse)

def make_weight_map(x: Tensor, alpha: float = 3.0, beta: float = 2.0) -> Tensor:
    """
    Boost foreground (dark) + edges to fight background dominance.
    x in [0,1], [B,1,H,W] preferred (will average channels if C!=1).
    """
    if x.shape[1] != 1:
        x = x.mean(dim=1, keepdim=True)

    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                      device=x.device, dtype=x.dtype).view(1,1,3,3)/4.0
    ky = kx.transpose(2,3)
    gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
    edges = (gx.abs() + gy.abs() > 0.05).float()
    fg = (x < 0.98).float()

    w = 1.0 + alpha*fg + beta*edges
    w = w / (w.mean(dim=[1,2,3], keepdim=True) + 1e-6)
    return w

def kl_per_dim(mu: Tensor, logvar: Tensor) -> Tensor:
    # compute in fp32; assume model clamps logvar internally on its own
    mu32, lv32 = mu.float(), logvar.float()
    return 0.5 * torch.mean(torch.exp(lv32) + mu32**2 - 1.0 - lv32, dim=0)

def count_active_units(kld: Tensor, tau: float = 0.01) -> int:
    return int((kld > tau).sum().item())

def load_cfg_from_ckpt(ckpt_path: str) -> HybridVAEConfig:
    from dataclasses import fields as dataclass_fields
    d = torch.load(ckpt_path, map_location="cpu")
    raw = d["cfg"] if isinstance(d["cfg"], dict) else d["cfg"].__dict__
    allowed = {f.name for f in dataclass_fields(HybridVAEConfig)}
    clean = {}
    for k, v in raw.items():
        if k in allowed:
            if k == "attn_scales" and isinstance(v, list):
                v = tuple(v)
            clean[k] = v
    return HybridVAEConfig(**clean), d

def _to_vis3(t: torch.Tensor) -> torch.Tensor:
    # expand grayscale to 3ch for nicer visualization
    return t.repeat(1, 3, 1, 1) if t.size(1) == 1 else t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--root", default="./AMSL Dataset", type=str)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--img_size", type=int, default=None, help="override resolution if desired")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lpips", action="store_true", help="compute LPIPS (requires pip install lpips)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_windows = (os.name == "nt")
    num_workers = 0 if is_windows else 6

    # Load config + model
    cfg, payload = load_cfg_from_ckpt(args.ckpt)
    if args.img_size is not None:
        cfg.img_size = args.img_size  # allow override
    model = HybridVAE(cfg).to(device).eval()
    model.load_state_dict(payload["model"], strict=False)

    # Data
    grayscale = (cfg.img_channels == 1)
    ds_cfg = AMSLQuadsConfig(root=args.root, split=args.split, img_size=cfg.img_size,
                             grayscale=grayscale, include_annotations=False, max_items=800)
    _, dl = build_dataloader(ds_cfg, batch_size=args.batch, shuffle=False,
                             num_workers=num_workers, pin_memory=(device=="cuda"),
                             persistent_workers=(device=="cuda" and not is_windows))

    # Optional LPIPS
    if args.lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
            for p in lpips_fn.parameters(): p.requires_grad_(False)
            def to_lpips(x):  # [0,1]→[-1,1], 3ch
                if x.size(1)==1: x = x.repeat(1,3,1,1)
                return x*2.0 - 1.0
        except Exception as e:
            print("LPIPS not available, proceeding without it. Error:", e)
            args.lpips = False
            lpips_fn = None

    # Eval loop
    n = 0
    sum_l1 = sum_l1w = sum_msssim = sum_psnr = 0.0
    sum_lpips = 0.0

    with torch.no_grad():
        for b in dl:
            x = b["images"].to(device)
            mu, logvar, feats = model.encode(x)
            recon = model.decode(mu, feats).clamp(0,1)

            x32, r32 = x.float(), recon.float()
            # metrics
            l1v = torch.mean(torch.abs(r32 - x32), dim=[1,2,3])
            wmap = make_weight_map(x32)
            l1wv = torch.mean(wmap * torch.abs(r32 - x32), dim=[1,2,3])
            mssv = ms_ssim_fn(r32, x32, data_range=1.0, size_average=False)
            psnrv = psnr(r32, x32)

            sum_l1   += float(l1v.sum());   sum_l1w  += float(l1wv.sum())
            sum_msssim += float(mssv.sum()); sum_psnr += float(psnrv.sum())
            if args.lpips:
                sum_lpips += float(lpips_fn(to_lpips(r32), to_lpips(x32)).mean()) * x.size(0)
            n += x.size(0)

        avg_l1 = sum_l1 / n
        avg_l1w = sum_l1w / n
        avg_msssim = sum_msssim / n
        avg_psnr = sum_psnr / n
        avg_lpips = (sum_lpips / n) if args.lpips else None

        # Latent diagnostics (on one batch)
        x = next(iter(dl))["images"].to(device)
        mu, logvar, feats = model.encode(x)
        kld_dim = kl_per_dim(mu, logvar)
        active = count_active_units(kld_dim, tau=0.01)
        mean_kld = float(kld_dim.mean().detach().cpu())

    # Save a visual grid
    out_dir = Path("results/eval"); out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        x = next(iter(dl))["images"].to(device)[:8]
        mu, logvar, feats = model.encode(x)
        recon = model.decode(mu, feats).clamp(0,1)
        # 1) error map in 1ch
        err = (recon - x).abs()

        # 2) normalize per-image for visibility
        err = err / (err.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)

        # 3) make all three tensors 3-channel for a pretty grid
        x_vis = _to_vis3(x)
        recon_vis = _to_vis3(recon)
        err_vis = _to_vis3(err)

        grid = torch.cat([x_vis, recon_vis, err_vis], dim=0)
        save_image(make_grid(grid, nrow=8, padding=2), str(out_dir / f"{args.split}_recon_grid.png"))

    # Print summary
    print(f"{args.split}: "
          f"L1={avg_l1:.5f}  L1w={avg_l1w:.5f}  MS-SSIM={avg_msssim:.4f}  PSNR={avg_psnr:.2f}dB"
          + (f"  LPIPS={avg_lpips:.4f}" if avg_lpips is not None else ""))
    print(f"latent: active={active}/{cfg.latent_dim}  meanKL={mean_kld:.3f}")
    print(f"Saved grid → {out_dir / (args.split + '_recon_grid.png')}")

if __name__ == "__main__":
    main()
