from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPORT_ROOT_DEFAULT = Path("results/xai")
SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIRS = {
    "attr_correlation": "attr_correlation",
    "latent_embedding": "latent_embedding",
    "encoder_ig": "encoder_integrated_gradients",
    "encoder_gradcam": "encoder_gradcam",
    "decoder_influence": "decoder_influence",
    "nudge": "counterfactual_nudge",
    "optimize": "counterfactual_optimize",
    "dice_counterfactuals": "dice_counterfactuals",
}


def run_step(label: str, cmd: List[str], method_key: str, run_root: Path, log_path: Path) -> List[Path]:
    method_base = run_root / METHOD_DIRS[method_key]
    before = {p for p in method_base.glob("*") if p.is_dir()} if method_base.exists() else set()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n==== {label} ====" + "\n")
        log_f.write("Command: " + " ".join(cmd) + "\n")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                log_f.write("STDOUT:\n" + exc.stdout + "\n")
            if exc.stderr:
                log_f.write("STDERR:\n" + exc.stderr + "\n")
            raise
        else:
            if result.stdout:
                log_f.write("STDOUT:\n" + result.stdout + "\n")
            if result.stderr:
                log_f.write("STDERR:\n" + result.stderr + "\n")

    new_dirs = []
    if method_base.exists():
        candidates = [p for p in method_base.glob("*") if p.is_dir()]
        new_dirs = sorted([p for p in candidates if p not in before], key=lambda p: p.stat().st_mtime)
    return new_dirs

def run_step(label: str, cmd: List[str], method_key: str, run_root: Path, log_path: Path) -> List[Path]:
    method_base = run_root / METHOD_DIRS[method_key]
    before = {p for p in method_base.glob("*") if p.is_dir()} if method_base.exists() else set()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n==== {label} ====" + "\n")
        log_f.write("Command: " + " ".join(cmd) + "\n")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                log_f.write("STDOUT:\n" + exc.stdout + "\n")
            if exc.stderr:
                log_f.write("STDERR:\n" + exc.stderr + "\n")
            raise
        else:
            if result.stdout:
                log_f.write("STDOUT:\n" + result.stdout + "\n")
            if result.stderr:
                log_f.write("STDERR:\n" + result.stderr + "\n")

    new_dirs = []
    if method_base.exists():
        candidates = [p for p in method_base.glob("*") if p.is_dir()]
        new_dirs = sorted([p for p in candidates if p not in before], key=lambda p: p.stat().st_mtime)
    return new_dirs


def load_meta(paths: List[Path]) -> List[Dict]:
    metas = []
    for p in paths:
        meta_path = p / "meta.json"
        if meta_path.exists():
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
    return metas


def summarise(meta: Dict) -> List[str]:
    lines = []
    method = meta.get("method", "unknown")
    lines.append(f"Method: {method}")
    if method == "attr_correlation":
        lines.append(f"  Samples used: {meta.get('samples_used')} | latent_dim: {meta.get('latent_dim')} ")
        top = meta.get("top_correlations", {})
        for attr, vals in top.items():
            if vals:
                first = vals[0]
                lines.append(f"  {attr}: z{first['dim']} corr={first['corr']:.3f}")
    elif method == "latent_embedding":
        lines.append(f"  Samples used: {meta.get('samples_used')} | method: {meta.get('embedding_method')} ")
    elif method == "encoder_integrated_gradients":
        lines.append(f"  Target dim: z{meta.get('target_dim')} | samples: {meta.get('samples')} | baseline: {meta.get('baseline')} ")
    elif method == "decoder_influence":
        dims = ", ".join(f"z{c['dim']}" for c in meta.get('dimensions', []))
        lines.append(f"  Dimensions analysed: {dims}")
    elif method == "encoder_gradcam":
        lines.append(f"  Layer: {meta.get('layer')} | target dim: z{meta.get('target_dim')} | samples: {meta.get('samples')}")
    elif method == "counterfactual_nudge":
        lines.append(f"  Dims traversed: {meta.get('dims')} | delta: {meta.get('delta')}")
    elif method == "counterfactual_optimize":
        lines.append(f"  Target {meta.get('attr')} -> {meta.get('target')} | achieved {meta.get('achieved'):.4f}")
    elif method == "dice_counterfactuals":
        lines.append(f"  Dims: {meta.get('dims')} | surrogate acc: {meta.get('surrogate_accuracy', 0):.3f} | samples: {meta.get('samples')}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Run the full explainability suite and compile a summary report")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default="./AMSL Dataset")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out_root", default=str(REPORT_ROOT_DEFAULT))

    parser.add_argument("--max_items", type=int, default=4000, help="Samples for encoder statistics")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--tsne_items", type=int, default=2000)
    parser.add_argument("--tsne_batch", type=int, default=64)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--ig_dims", type=int, nargs="*", default=[0])
    parser.add_argument("--ig_samples", type=int, default=4)
    parser.add_argument("--ig_steps", type=int, default=32)
    parser.add_argument("--gradcam_dims", type=int, nargs="*", default=[0])
    parser.add_argument("--gradcam_samples", type=int, default=4)
    parser.add_argument("--gradcam_layer", type=str, default="encoder.res_skip")
    parser.add_argument("--decoder_dims", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--decoder_samples", type=int, default=4)
    parser.add_argument("--decoder_delta", type=float, default=1.0)
    parser.add_argument("--nudge_delta", type=float, default=1.0)
    parser.add_argument("--nudge_dims", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--opt_dims", type=int, nargs="*", default=[0])
    parser.add_argument("--opt_attr", choices=["area", "xspread", "yspread"], default="area")
    parser.add_argument("--opt_target", type=float, default=0.35)
    parser.add_argument("--opt_steps", type=int, default=200)
    parser.add_argument("--opt_lr", type=float, default=0.05)
    parser.add_argument("--opt_lam", type=float, default=0.01)
    parser.add_argument("--dice_dims", type=int, nargs="*", default=[17, 19, 57, 29, 7, 10])
    parser.add_argument("--dice_dataset_items", type=int, default=4000)
    parser.add_argument("--dice_batch", type=int, default=64)
    parser.add_argument("--dice_samples", type=int, default=4)
    parser.add_argument("--dice_total_cfs", type=int, default=1)
    parser.add_argument("--dice_device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--skip_nudge", action="store_true")
    parser.add_argument("--skip_optimize", action="store_true")
    parser.add_argument("--skip_dice", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now().strftime("run_%Y%m%d-%H%M%S")
    run_root = Path(args.out_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "execution.log"

    python_exe = sys.executable
    summaries: List[Dict] = []

    # Attr correlations
    cmd = [
        python_exe,
        str(SCRIPT_DIR / "xai_attr_correlation.py"),
        "--ckpt", args.ckpt,
        "--root", args.root,
        "--out_root", str(run_root),
        "--device", args.device,
        "--max_items", str(args.max_items),
        "--batch", str(args.batch),
    ]
    if args.img_size is not None:
        cmd += ["--img_size", str(args.img_size)]
    new_dirs = run_step("Latent vs attribute correlations", cmd, "attr_correlation", run_root, log_path)
    summaries.extend(load_meta(new_dirs))

    # Latent embedding
    cmd = [
        python_exe,
        str(SCRIPT_DIR / "xai_tsne.py"),
        "--ckpt", args.ckpt,
        "--root", args.root,
        "--out_root", str(run_root),
        "--device", args.device,
        "--max_items", str(args.tsne_items),
        "--batch", str(args.tsne_batch),
        "--perplexity", str(args.tsne_perplexity),
    ]
    if args.img_size is not None:
        cmd += ["--img_size", str(args.img_size)]
    new_dirs = run_step("Latent embedding visualisation", cmd, "latent_embedding", run_root, log_path)
    summaries.extend(load_meta(new_dirs))

    # Encoder IG per requested dim
    for dim in args.ig_dims:
        cmd = [
            python_exe,
            str(SCRIPT_DIR / "xai_encoder_ig.py"),
            "--ckpt", args.ckpt,
            "--root", args.root,
            "--out_root", str(run_root),
            "--device", args.device,
            "--img_size", str(args.img_size) if args.img_size is not None else str(256),
            "--samples", str(args.ig_samples),
            "--target_dim", str(dim),
            "--n_steps", str(args.ig_steps),
        ]
        new_dirs = run_step(f"Encoder IG for z{dim}", cmd, "encoder_ig", run_root, log_path)
        summaries.extend(load_meta(new_dirs))

    for gc_dim in args.gradcam_dims:
        cmd = [
            python_exe,
            str(SCRIPT_DIR / "xai_encoder_gradcam.py"),
            "--ckpt", args.ckpt,
            "--root", args.root,
            "--out_root", str(run_root),
            "--device", args.device,
            "--img_size", str(args.img_size) if args.img_size is not None else str(256),
            "--samples", str(args.gradcam_samples),
            "--target_dim", str(gc_dim),
            "--layer", args.gradcam_layer,
        ]
        new_dirs = run_step(f"Encoder Grad-CAM for z{gc_dim}", cmd, "encoder_gradcam", run_root, log_path)
        summaries.extend(load_meta(new_dirs))

    # Decoder influence
    cmd = [
        python_exe,
        str(SCRIPT_DIR / "xai_decoder_influence.py"),
        "--ckpt", args.ckpt,
        "--root", args.root,
        "--out_root", str(run_root),
        "--device", args.device,
        "--samples", str(args.decoder_samples),
        "--delta", str(args.decoder_delta),
        "--dims",
    ] + [str(d) for d in args.decoder_dims]
    if args.img_size is not None:
        cmd += ["--img_size", str(args.img_size)]
    new_dirs = run_step("Decoder latent influence", cmd, "decoder_influence", run_root, log_path)
    summaries.extend(load_meta(new_dirs))

    # Counterfactual nudge
    if not args.skip_nudge:
        cmd = [
            python_exe,
            str(SCRIPT_DIR / "xai_counterfactuals.py"),
            "--ckpt", args.ckpt,
            "--root", args.root,
            "--device", args.device,
        ]
        if args.img_size is not None:
            cmd += ["--img_size", str(args.img_size)]
        cmd += ["nudge", "--out_root", str(run_root), "--delta", str(args.nudge_delta), "--dims"] + [str(d) for d in args.nudge_dims]
        new_dirs = run_step("Counterfactual nudges", cmd, "nudge", run_root, log_path)
        summaries.extend(load_meta(new_dirs))

    # Counterfactual optimisation
    if not args.skip_optimize:
        cmd = [
            python_exe,
            str(SCRIPT_DIR / "xai_counterfactuals.py"),
            "--ckpt", args.ckpt,
            "--root", args.root,
            "--device", args.device,
        ]
        if args.img_size is not None:
            cmd += ["--img_size", str(args.img_size)]
        cmd += ["optimize", "--out_root", str(run_root), "--attr", args.opt_attr, "--target", str(args.opt_target), "--steps", str(args.opt_steps), "--lr", str(args.opt_lr), "--lam", str(args.opt_lam), "--dims"] + [str(d) for d in args.opt_dims]
        new_dirs = run_step("Counterfactual optimisation", cmd, "optimize", run_root, log_path)
        summaries.extend(load_meta(new_dirs))

    if not args.skip_dice:
        dice_device = args.dice_device if args.dice_device is not None else args.device
        cmd = [
            python_exe,
            str(SCRIPT_DIR / "xai_dice_counterfactuals.py"),
            "--ckpt", args.ckpt,
            "--root", args.root,
            "--out_root", str(run_root),
            "--device", dice_device,
            "--samples", str(args.dice_samples),
            "--dataset_items", str(args.dice_dataset_items),
            "--dice_batch", str(args.dice_batch),
            "--total_cfs", str(args.dice_total_cfs),
            "--dims",
        ] + [str(d) for d in args.dice_dims]
        if args.img_size is not None:
            cmd += ["--img_size", str(args.img_size)]
        new_dirs = run_step("DiCE counterfactuals", cmd, "dice_counterfactuals", run_root, log_path)
        summaries.extend(load_meta(new_dirs))

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    md_lines = ["# Explainability Report", "", f"Run directory: `{run_root}`", ""]
    for meta in summaries:
        md_lines.extend(summarise(meta))
        md_lines.append("")
    (run_root / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Explainability artefacts stored in {run_root}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
