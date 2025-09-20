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
    resolve_device,
    save_grid,
    timestamped_subdir,
)

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except Exception:
    HAS_TSNE = False
    from sklearn.decomposition import PCA

# Speed-up hints
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RESULTS_ROOT = Path("results/xai")
METHOD_NAME = "latent_embedding"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Embed latent codes with t-SNE/PCA and visualise attribute trends")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default="./AMSL Dataset")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--max_items", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out_root", default=str(RESULTS_ROOT))
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity (ignored for PCA fallback)")
    parser.add_argument("--preview", type=int, default=8, help="Number of recon pairs saved for context")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device=device, img_size_override=args.img_size)

    _, loader = build_loader(
        cfg,
        root=args.root,
        split=args.split,
        batch_size=args.batch,
        device=device,
        shuffle=False,
        max_items=args.max_items,
        include_annotations=False,
    )

    latents, area_frac, l1_scores = [], [], []
    preview_pairs = None
    seen = 0

    for batch in loader:
        x = batch["images"].to(device)
        mu, logvar, feats = model.encode(x)
        recon = model.decode(mu, feats).clamp(0, 1)

        latents.append(mu.detach().cpu().numpy())
        x_np = x.detach().cpu().numpy()
        recon_np = recon.detach().cpu().numpy()
        area_frac.append(1.0 - x_np.mean(axis=(1, 2, 3)))
        l1_scores.append(np.mean(np.abs(recon_np - x_np), axis=(1, 2, 3)))

        if preview_pairs is None:
            k = min(args.preview, x.size(0))
            preview_pairs = (x[:k].detach().cpu(), recon[:k].detach().cpu())

        seen += x.size(0)
        if seen >= args.max_items:
            break

    latents_np = np.concatenate(latents, axis=0)[: args.max_items]
    area_np = np.concatenate(area_frac, axis=0)[: len(latents_np)]
    l1_np = np.concatenate(l1_scores, axis=0)[: len(latents_np)]

    if HAS_TSNE:
        embed = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            learning_rate="auto",
            init="pca",
            random_state=42,
        ).fit_transform(latents_np)
        embed_method = "tsne"
    else:
        embed = PCA(n_components=2).fit_transform(latents_np)
        embed_method = "pca"

    run_dir = timestamped_subdir(Path(args.out_root), METHOD_NAME)

    def scatter(values, label, cmap="viridis"):
        plt.figure(figsize=(7, 5), dpi=150)
        sc = plt.scatter(embed[:, 0], embed[:, 1], c=values, cmap=cmap, s=12, edgecolors="none")
        plt.title(f"Latent embedding coloured by {label}")
        cbar = plt.colorbar(sc)
        cbar.set_label(label)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        path = run_dir / f"{embed_method}_{label}.png"
        plt.savefig(path)
        plt.close()
        return path

    area_path = scatter(area_np, "area_fraction", cmap="viridis")
    l1_path = scatter(l1_np, "recon_L1", cmap="magma")

    np.save(run_dir / "latents.npy", latents_np)
    np.save(run_dir / "embedding.npy", embed)
    np.save(run_dir / "area_fraction.npy", area_np)
    np.save(run_dir / "recon_l1.npy", l1_np)

    if preview_pairs is not None:
        src, rec = preview_pairs
        tensors = [ensure_rgb(src), ensure_rgb(rec), ensure_rgb((rec - src).abs() / ((rec - src).abs().amax(dim=[1,2,3], keepdim=True) + 1e-8))]
        save_grid(tensors, run_dir / "reconstruction_preview.png", nrow=src.size(0))

    metadata = {
        "method": METHOD_NAME,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "split": args.split,
        "img_size": cfg.img_size,
        "latent_dim": cfg.latent_dim,
        "samples_used": int(len(latents_np)),
        "embedding_method": embed_method,
        "perplexity": args.perplexity if HAS_TSNE else None,
        "scatter_plots": {
            "area_fraction": str(area_path.name),
            "recon_L1": str(l1_path.name),
        },
    }
    dump_metadata(run_dir / "meta.json", metadata)

    print(f"Saved embedding visuals and arrays to {run_dir}")


if __name__ == "__main__":
    main()
