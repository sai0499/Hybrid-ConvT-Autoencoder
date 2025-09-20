# scripts/sample.py
from __future__ import annotations
import sys, os, math, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import torch
from torchvision.utils import save_image, make_grid

from models.hybrid_vae import HybridVAE, HybridVAEConfig
from data.amsl_quads import AMSLQuadsConfig, build_dataloader

# Speed: enable TF32 on supported NVIDIA GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------- utils -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_grid(t: torch.Tensor, path: Path, nrow: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(make_grid(t, nrow=nrow, padding=2), str(path))

def _filter_cfg(raw_cfg: dict) -> HybridVAEConfig:
    """
    Build a clean HybridVAEConfig from a (possibly noisy) ckpt cfg dict.
    It ignores runtime keys like 'skip_hw' that aren't dataclass fields.
    """
    from dataclasses import fields as dataclass_fields
    allowed = {f.name for f in dataclass_fields(HybridVAEConfig)}
    clean = {}
    for k, v in raw_cfg.items():
        if k in allowed:
            if k == "attn_scales" and isinstance(v, list):
                v = tuple(v)
            clean[k] = v
    return HybridVAEConfig(**clean)

def load_model(ckpt_path: str, img_size_override: int | None, device: str) -> tuple[HybridVAE, HybridVAEConfig, dict]:
    payload = torch.load(ckpt_path, map_location="cpu")
    raw_cfg = payload["cfg"] if isinstance(payload["cfg"], dict) else payload["cfg"].__dict__
    cfg = _filter_cfg(raw_cfg)
    if img_size_override is not None:
        cfg.img_size = img_size_override
    model = HybridVAE(cfg).to(device).eval()
    model.load_state_dict(payload["model"], strict=False)
    return model, cfg, payload

def build_val_batch(cfg: HybridVAEConfig, root: str, batch_size: int, device: str):
    is_windows = (os.name == "nt")
    ds_cfg = AMSLQuadsConfig(
        root=root, split="val", img_size=cfg.img_size,
        grayscale=(cfg.img_channels == 1), include_annotations=False, max_items=None
    )
    _, dl = build_dataloader(
        ds_cfg, batch_size=batch_size, shuffle=False,
        num_workers=0 if is_windows else 6,
        pin_memory=(device == "cuda"),
        persistent_workers=False
    )
    batch = next(iter(dl))["images"].to(device)
    return batch, dl

def tile_feats_to(feats: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    """Repeat/truncate each skip tensor along batch dimension to get exactly n samples."""
    out = {}
    for k, v in feats.items():
        b = v.size(0)
        if b >= n:
            out[k] = v[:n]
        else:
            reps = (n + b - 1) // b
            out[k] = v.repeat(reps, 1, 1, 1)[:n]
    return out


# ----------------------------- main -----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Sampling & visualization for HybridVAE")
    ap.add_argument("--ckpt", required=True, type=str, help="Path to .pt checkpoint")
    ap.add_argument("--root", default="./AMSL Dataset", type=str, help="Dataset root")
    ap.add_argument("--mode", choices=["prior", "cond", "recon"], default="prior",
                    help="prior=latent-only, cond=borrow val skip feats, recon=side-by-side")
    ap.add_argument("--img_size", type=int, default=None, help="Override image size if desired")
    ap.add_argument("--batch", type=int, default=16, help="Batch size for val loader (recon/cond)")
    ap.add_argument("--n_prior", type=int, default=32, help="Number of prior/cond samples to save")
    ap.add_argument("--trav_dims", type=int, nargs="*", default=[0,1,2,3], help="Latent dims to traverse")
    ap.add_argument("--trav_steps", type=int, default=11, help="Steps in each traversal")
    ap.add_argument("--trav_range", type=float, default=2.5, help="Range per traversal dim (±range)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg, _ = load_model(args.ckpt, args.img_size, device)
    out_dir = Path("results/samples"); out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- recon mode ----------------
    if args.mode == "recon":
        val_batch, _ = build_val_batch(cfg, args.root, max(8, args.batch), device)
        k = min(8, val_batch.size(0))
        x = val_batch[:k]
        mu, logvar, feats = model.encode(x)
        recon = model.decode(mu, feats).clamp(0, 1)
        save_grid(torch.cat([x, recon], dim=0), out_dir / "val_recon_grid.png", nrow=k)
        print("Saved:", out_dir / "val_recon_grid.png")
        return

    # ---------------- prior mode (latent-only) ----------------
    if args.mode == "prior":
        # unconditional samples
        n = args.n_prior
        z = torch.randn(n, cfg.latent_dim, device=device)
        x_prior = model.decode(z, feats=None).clamp(0, 1)
        save_grid(x_prior, out_dir / "prior_samples.png", nrow=min(8, n))
        print("Saved:", out_dir / "prior_samples.png")

        # traversals
        steps = args.trav_steps
        vals = torch.linspace(-args.trav_range, args.trav_range, steps, device=device)
        for d in args.trav_dims:
            z0 = torch.randn(1, cfg.latent_dim, device=device).repeat(steps, 1)
            z0[:, d] = vals
            x_trav = model.decode(z0, feats=None).clamp(0, 1)
            save_grid(x_trav, out_dir / f"traversal_dim{d}.png", nrow=steps)
            print("Saved:", out_dir / f"traversal_dim{d}.png")
        return

    # ---------------- cond mode (borrow skip feats) ----------------
    # Get a val batch and reuse its skip-features while sampling z.
    if args.mode == "cond":
        val_batch, _ = build_val_batch(cfg, args.root, max(args.n_prior, args.batch), device)
        mu_v, logvar_v, feats_v = model.encode(val_batch)

        # conditional "prior" samples (latent varies, skip feats borrowed)
        n = args.n_prior
        z = torch.randn(n, cfg.latent_dim, device=device)
        feats_cond = tile_feats_to(feats_v, n)
        x_cond = model.decode(z, feats_cond).clamp(0, 1)
        save_grid(x_cond, out_dir / "cond_samples.png", nrow=min(8, n))
        print("Saved:", out_dir / "cond_samples.png")

        # conditional traversals
        steps = args.trav_steps
        vals = torch.linspace(-args.trav_range, args.trav_range, steps, device=device)
        for d in args.trav_dims:
            z0 = torch.randn(1, cfg.latent_dim, device=device).repeat(steps, 1)
            z0[:, d] = vals
            feats_steps = tile_feats_to(feats_v, steps)
            x_trav = model.decode(z0, feats_steps).clamp(0, 1)
            save_grid(x_trav, out_dir / f"cond_traversal_dim{d}.png", nrow=steps)
            print("Saved:", out_dir / f"cond_traversal_dim{d}.png")
        return


if __name__ == "__main__":
    main()
