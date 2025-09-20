# scripts/xai_tsne.py
from __future__ import annotations
import sys, os, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
from torchvision.utils import save_image, make_grid

from models.hybrid_vae import HybridVAE, HybridVAEConfig
from data.amsl_quads import AMSLQuadsConfig, build_dataloader

# optional: sklearn for TSNE; otherwise fall back to PCA
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except Exception:
    HAS_TSNE = False
    from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--root", default="./AMSL Dataset", type=str)
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--max_items", type=int, default=2000, help="number of points for t-SNE")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_windows = (os.name == "nt")
    num_workers = 0 if is_windows else 6

    payload = torch.load(args.ckpt, map_location="cpu")
    raw_cfg = payload["cfg"] if isinstance(payload["cfg"], dict) else payload["cfg"].__dict__
    cfg = filter_cfg(raw_cfg)
    if args.img_size is not None:
        cfg.img_size = args.img_size

    model = HybridVAE(cfg).to(device).eval()
    model.load_state_dict(payload["model"], strict=False)

    ds_cfg = AMSLQuadsConfig(root=args.root, split=args.split, img_size=cfg.img_size,
                             grayscale=(cfg.img_channels==1), include_annotations=False, max_items=None)
    _, dl = build_dataloader(ds_cfg, batch_size=args.batch, shuffle=False,
                             num_workers=num_workers, pin_memory=(device=="cuda"),
                             persistent_workers=(device=="cuda" and not is_windows))

    Z, area, l1 = [], [], []
    n = 0
    for b in dl:
        x = b["images"].to(device)
        mu, logvar, feats = model.encode(x)
        recon = model.decode(mu, feats).clamp(0,1)

        z = mu.detach().cpu().numpy()
        Z.append(z)

        # simple interpretable stats
        x_np = x.detach().cpu().numpy()
        area.append((1.0 - x_np.mean(axis=(1,2,3))))        # fraction of black pixels
        l1.append(np.mean(np.abs(recon.detach().cpu().numpy() - x_np), axis=(1,2,3)))

        n += x.size(0)
        if n >= args.max_items:
            break

    Z = np.concatenate(Z, axis=0)[:args.max_items]
    area = np.concatenate(area, axis=0)[:len(Z)]
    l1 = np.concatenate(l1, axis=0)[:len(Z)]

    # embed
    if HAS_TSNE:
        emb = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42).fit_transform(Z)
        method = "tsne"
    else:
        emb = PCA(n_components=2).fit_transform(Z)
        method = "pca"

    out_dir = Path("results/xai"); out_dir.mkdir(parents=True, exist_ok=True)

    def scatter(vals, name, cmap="viridis"):
        plt.figure(figsize=(6,5), dpi=150)
        sc = plt.scatter(emb[:,0], emb[:,1], c=vals, cmap=cmap, s=8, edgecolors="none")
        plt.colorbar(sc)
        plt.title(f"Latent space ({method}) colored by {name}")
        plt.tight_layout()
        p = out_dir / f"latent_{method}_{name}.png"
        plt.savefig(p); plt.close()
        print("Saved:", p)

    scatter(area, "area_frac")
    scatter(l1, "recon_L1", cmap="magma")

    # also save the raw arrays for report use
    np.save(out_dir/"latent_Z.npy", Z)
    np.save(out_dir/"latent_embed.npy", emb)
    np.save(out_dir/"latent_area.npy", area)
    np.save(out_dir/"latent_l1.npy", l1)

if __name__ == "__main__":
    main()
