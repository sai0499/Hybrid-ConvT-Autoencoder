#!/usr/bin/env python
"""One-button pipeline for Stage-A (autoencoder pre-training) and Stage-B (adversarial finetune).

Stage-A runs the reconstruction-focused trainer (train_debug.py).
Stage-B automatically consumes the best checkpoint from Stage-A and launches train_stageB.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGE_A_SCRIPT = Path("scripts/train_debug.py")
STAGE_B_SCRIPT = Path("scripts/train_stageB.py")
DEFAULT_STAGE_A_CKPT = Path("checkpoints/hybridvae_debug256_best.pt")


def run_command(label: str, cmd: list[str]):
    print(f"\n=== Running {label} ===")
    print("Command:", " ".join(cmd))
    completed = subprocess.run(cmd, check=True)
    print(f"=== {label} finished (return code {completed.returncode}) ===\n")


def main():
    parser = argparse.ArgumentParser(description="Train Stage-A VAE and Stage-B adversarial finetune in one go")
    parser.add_argument("--stage-a-script", type=Path, default=STAGE_A_SCRIPT,
                        help="Path to Stage-A training script (default: scripts/train_debug.py)")
    parser.add_argument("--stage-b-script", type=Path, default=STAGE_B_SCRIPT,
                        help="Path to Stage-B training script (default: scripts/train_stageB.py)")
    parser.add_argument("--stage-a-ckpt", type=Path, default=DEFAULT_STAGE_A_CKPT,
                        help="Where Stage-A stores its best checkpoint (fed into Stage-B)")
    parser.add_argument("--skip-stage-a", action="store_true",
                        help="Only run Stage-B and assume the Stage-A checkpoint already exists")

    # Stage-B specific options (mirrors train_stageB.py)
    parser.add_argument("--root", default="./AMSL Dataset", help="Dataset root directory")
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution for Stage-B")
    parser.add_argument("--base", type=int, default=48, help="Base channel width for Stage-B VAE")
    parser.add_argument("--batch", type=int, default=12, help="Batch size for Stage-B")
    parser.add_argument("--epochs", type=int, default=8, help="Number of Stage-B epochs")
    parser.add_argument("--lr_g", type=float, default=2e-4, help="Generator/autoencoder learning rate")
    parser.add_argument("--lr_d", type=float, default=2e-4, help="Discriminator learning rate")
    parser.add_argument("--freeze_enc_epochs", type=int, default=10,
                        help="Number of epochs to freeze the encoder in Stage-B")
    parser.add_argument("--prior_prob", type=float, default=0.25,
                        help="Probability of sampling prior latents for the discriminator")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable AMP in Stage-B (passes --amp only when AMP is requested)")

    args = parser.parse_args()

    stage_a_script = args.stage_a_script
    stage_b_script = args.stage_b_script
    stage_a_ckpt = args.stage_a_ckpt

    python_exe = sys.executable

    if not args.skip_stage_a:
        if not stage_a_script.exists():
            raise FileNotFoundError(f"Stage-A script not found: {stage_a_script}")
        run_command("Stage-A (train_debug)", [python_exe, str(stage_a_script)])
    else:
        print("Skipping Stage-A as requested.")

    if not stage_a_ckpt.exists():
        raise FileNotFoundError(
            f"Stage-A checkpoint not found at {stage_a_ckpt}."
            " Provide --stage-a-ckpt pointing to the correct file or rerun Stage-A."
        )

    if not stage_b_script.exists():
        raise FileNotFoundError(f"Stage-B script not found: {stage_b_script}")

    stage_b_cmd: list[str] = [
        python_exe,
        str(stage_b_script),
        "--ckpt", str(stage_a_ckpt),
        "--root", args.root,
        "--img_size", str(args.img_size),
        "--base", str(args.base),
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--lr_g", str(args.lr_g),
        "--lr_d", str(args.lr_d),
        "--freeze_enc_epochs", str(args.freeze_enc_epochs),
        "--prior_prob", str(args.prior_prob),
    ]
    if not args.no_amp:
        stage_b_cmd.append("--amp")

    run_command("Stage-B (train_stageB)", stage_b_cmd)

    print("Training pipeline complete.")


if __name__ == "__main__":
    main()
