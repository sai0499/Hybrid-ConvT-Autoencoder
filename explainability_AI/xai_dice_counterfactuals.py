from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torchvision.utils import make_grid, save_image

import dice_ml
from raiutils.exceptions import UserConfigValidationException

from explainability_AI.common import (
    build_loader,
    dump_metadata,
    ensure_rgb,
    load_model,
    resolve_device,
    timestamped_subdir,
)

# Speed toggles
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RESULTS_ROOT = Path("results/xai")
METHOD_NAME = "dice_counterfactuals"


@torch.no_grad()
def collect_latent_dataset(
    model: torch.nn.Module,
    cfg,
    *,
    root: str,
    device: str,
    dims: List[int],
    max_items: int,
    batch_size: int,
) -> Tuple[pd.DataFrame, List[float]]:
    feature_names = [f"z{d}" for d in dims]
    records = []
    areas: List[float] = []

    _, loader = build_loader(
        cfg,
        root=root,
        split="val",
        batch_size=min(batch_size, max_items),
        device=device,
        shuffle=False,
        max_items=max_items,
        include_annotations=False,
    )

    for batch in loader:
        imgs = batch["images"].to(device)
        mu, logvar, feats = model.encode(imgs)
        mu_np = mu.detach().cpu().numpy()
        x_np = imgs.detach().cpu().numpy()
        area = 1.0 - x_np.mean(axis=(1, 2, 3))
        areas.extend(area.tolist())

        for i in range(mu_np.shape[0]):
            if len(records) >= max_items:
                break
            rec = {f: float(mu_np[i, d]) for f, d in zip(feature_names, dims)}
            rec["area"] = float(area[i])
            records.append(rec)
        if len(records) >= max_items:
            break

    df = pd.DataFrame(records)
    if "area" not in df.columns:
        raise RuntimeError("Area column missing in records")
    if df["area"].nunique() < 2:
        threshold = float(df["area"].iloc[0])
    else:
        threshold = float(df["area"].quantile(0.6))
    df["label"] = (df["area"] >= threshold).astype(int)
    df = df[[*feature_names, "label"]]
    return df, areas


def build_surrogate(df: pd.DataFrame, dims: List[int]) -> Tuple[dice_ml.Data, dice_ml.Model, LogisticRegression]:
    feature_names = [f"z{d}" for d in dims]
    data = dice_ml.Data(dataframe=df, continuous_features=feature_names, outcome_name="label")
    clf = LogisticRegression(max_iter=800)
    clf.fit(df[feature_names], df["label"])
    model = dice_ml.Model(model=clf, backend="sklearn")
    return data, model, clf


@torch.no_grad()
def generate_counterfactuals(
    model: torch.nn.Module,
    cfg,
    *,
    root: str,
    device: str,
    dims: List[int],
    dice_data,
    dice_model,
    n_samples: int,
    total_cfs: int,
    batch_size: int,
) -> Tuple[List[dict], torch.Tensor, torch.Tensor, List[torch.Tensor | None]]:
    model.eval()
    feature_names = [f"z{d}" for d in dims]

    _, loader = build_loader(
        cfg,
        root=root,
        split="val",
        batch_size=max(batch_size, n_samples),
        device=device,
        shuffle=False,
        max_items=None,
        include_annotations=False,
    )
    batch = next(iter(loader))
    imgs = batch["images"].to(device)[:n_samples]
    mu, logvar, feats = model.encode(imgs)
    feats_anchor = {k: v.detach() for k, v in (feats or {}).items()}
    base = model.decode(mu, feats_anchor if feats_anchor else None).clamp(0, 1)

    x_np = imgs.detach().cpu().numpy()
    area = 1.0 - x_np.mean(axis=(1, 2, 3))

    dice = dice_ml.Dice(dice_data, dice_model, method="random")
    cf_infos: List[dict] = []
    cf_recons: List[torch.Tensor | None] = []

    for idx in range(n_samples):
        query = {f: float(mu[idx, d].detach().cpu()) for f, d in zip(feature_names, dims)}
        try:
            cf = dice.generate_counterfactuals(pd.DataFrame([query]), total_CFs=total_cfs, desired_class="opposite")
        except UserConfigValidationException:
            cf = None
        record = {
            "index": int(idx),
            "original_area": float(area[idx]),
            "cf_area": None,
            "cf_features": None,
        }
        cf_tensor: torch.Tensor | None = None

        if cf is not None and cf.cf_examples_list and not cf.cf_examples_list[0].final_cfs_df.empty:
            cf_row = cf.cf_examples_list[0].final_cfs_df.iloc[0]
            mu_cf = mu[idx].detach().clone()
            for d in dims:
                mu_cf[d] = float(cf_row[f"z{d}"])
            cf_tensor = model.decode(mu_cf.unsqueeze(0), feats_anchor if feats_anchor else None).clamp(0, 1)
            cf_area = float((1.0 - cf_tensor.mean(dim=[1, 2, 3])).item())
            record["cf_area"] = cf_area
            record["cf_features"] = {f"z{d}": float(cf_row[f"z{d}"]) for d in dims}

        cf_infos.append(record)
        cf_recons.append(cf_tensor)

    return cf_infos, imgs, base, cf_recons


def main():
    parser = argparse.ArgumentParser(description="Generate counterfactual latents via DiCE")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default="./AMSL Dataset")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dims", type=int, nargs="*", default=[17, 19, 57, 29, 7, 10])
    parser.add_argument("--dataset_items", type=int, default=4000)
    parser.add_argument("--dice_batch", type=int, default=64, help="Batch size while collecting latents")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--sample_batch", type=int, default=8)
    parser.add_argument("--total_cfs", type=int, default=1)
    parser.add_argument("--out_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device=device, img_size_override=args.img_size)
    model.eval()

    df, areas = collect_latent_dataset(
        model,
        cfg,
        root=args.root,
        device=device,
        dims=args.dims,
        max_items=args.dataset_items,
        batch_size=args.dice_batch,
    )
    dice_data, dice_model, clf = build_surrogate(df, args.dims)

    cf_infos, imgs, base, cf_recons = generate_counterfactuals(
        model,
        cfg,
        root=args.root,
        device=device,
        dims=args.dims,
        dice_data=dice_data,
        dice_model=dice_model,
        n_samples=args.samples,
        total_cfs=args.total_cfs,
        batch_size=args.sample_batch,
    )

    run_dir = timestamped_subdir(Path(args.out_root), METHOD_NAME)

    for idx in range(args.samples):
        panels = [
            ensure_rgb(imgs[idx:idx+1].detach().cpu()).squeeze(0),
            ensure_rgb(base[idx:idx+1].detach().cpu()).squeeze(0),
        ]
        if cf_recons[idx] is not None:
            cf = ensure_rgb(cf_recons[idx].detach().cpu()).squeeze(0)
            diff = (cf_recons[idx] - base[idx:idx+1]).abs()
            diff = diff / (diff.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
            panels.extend([cf, ensure_rgb(diff.detach().cpu()).squeeze(0)])
        stack = torch.stack(panels)
        grid = make_grid(stack, nrow=len(panels), padding=4)
        save_image(grid, str(run_dir / f"sample_{idx:02d}.png"))

    metadata = {
        "method": METHOD_NAME,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "img_size": cfg.img_size,
        "device": device,
        "dims": args.dims,
        "dataset_items": args.dataset_items,
        "dice_batch": args.dice_batch,
        "samples": args.samples,
        "total_cfs": args.total_cfs,
        "surrogate_accuracy": float(clf.score(df[[f"z{d}" for d in args.dims]], df["label"])),
        "results": cf_infos,
    }
    dump_metadata(run_dir / "meta.json", metadata)

    print(f"DiCE counterfactuals saved in {run_dir}")


if __name__ == "__main__":
    main()
