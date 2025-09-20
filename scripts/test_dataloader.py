# scripts/test_dataloader.py
import os
from pathlib import Path
import torch
from torchvision.utils import make_grid, save_image

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add repo root
from data.amsl_quads import AMSLQuadsConfig, build_dataloader, SUBFOLDERS

def save_grid(tensor_bchw: torch.Tensor, out_path: str, nrow: int = 8) -> None:
    grid = make_grid(tensor_bchw, nrow=nrow, padding=2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)

def main():
    root = "./AMSL Dataset"   # your chosen relative path
    img_size = 256            # quick test at 256; we’ll move to 512 after
    grayscale = True

    # Count & preview for each split
    for split in ("train", "val", "test"):
        cfg = AMSLQuadsConfig(
            root=root,
            split=split,
            img_size=img_size,
            grayscale=grayscale,
            include_annotations=True,   # proves XML mapping works
        )
        ds, dl = build_dataloader(cfg, batch_size=32, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False)

        print(f"\n[{split}] total images: {len(ds):,}")
        # Per-family breakdown
        for fam in SUBFOLDERS:
            # count by path prefix
            n_fam = sum(1 for p, f in ds.files if f == fam)
            print(f"  {fam:28s}: {n_fam:,}")

        # Grab one batch and save a grid
        batch = next(iter(dl))
        imgs = batch["images"]       # [B,C,H,W] in [0,1]
        out = f"results/sanity_{split}_{img_size}px.png"
        save_grid(imgs, out, nrow=8)
        print(f"Saved sample grid → {out}")

        # If annotations exist, print a quick peek
        if any(a is not None for a in batch["annotations"]):
            a0 = next(a for a in batch["annotations"] if a is not None)
            print("Annotation example:", {k: a0[k] for k in a0 if k in ("width","height","objects")})

if __name__ == "__main__":
    main()
