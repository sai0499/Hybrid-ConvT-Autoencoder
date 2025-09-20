# scripts/train_debug.py
from __future__ import annotations
from pathlib import Path
import sys, time, random
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.amp import autocast, GradScaler
from torchvision.utils import save_image, make_grid
from pytorch_msssim import ms_ssim

# Speed + stability on NVIDIA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- toggles ---
USE_AMP = True         # mixed precision: model forward in AMP, losses in fp32
USE_ATTENTION = True   # attention math must be stabilized in models/blocks.py

# --- data + model ---
from data.amsl_quads import AMSLQuadsConfig, build_dataloader
from models.hybrid_vae import HybridVAE, HybridVAEConfig


# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def psnr(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3]) + eps
    return 10.0 * torch.log10(1.0 / mse)

@torch.no_grad()
def save_grid_img(t: Tensor, path: Path, nrow: int = 8):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(make_grid(t, nrow=nrow, padding=2), str(path))

class EMA:
    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}
    @torch.no_grad()
    def update(self, module: nn.Module):
        for k, v in module.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def copy_to(self, module: nn.Module):
        module.load_state_dict(self.shadow, strict=True)

def kl_capacity(global_step: int, total_steps: int, C_max: float = 6.0, warmup_steps: int = 500) -> float:
    if global_step <= warmup_steps: return 0.0
    t = min(1.0, (global_step - warmup_steps) / max(1, total_steps - warmup_steps))
    return C_max * t

def skip_schedule(epoch: int) -> tuple[float, float]:
    # (drop_skip_p, skip_gate)
    if epoch <= 2:  return 0.0, 1.0
    if epoch <= 4:  return 0.3, 1.0
    if epoch <= 6:  return 0.5, 0.9
    return 0.3, 1.0

def kl_per_dim(mu: Tensor, logvar: Tensor) -> Tensor:
    return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=0)

def count_active_units(kld: Tensor, tau: float = 0.01) -> int:
    return int((kld > tau).sum().item())

def make_weight_map(x: Tensor, alpha: float = 3.0, beta: float = 2.0) -> Tensor:
    """
    x∈[0,1], [B,1,H,W] preferred. Boost foreground (dark) + edges to fight bg dominance.
    """
    if x.shape[1] != 1:
        x = x.mean(dim=1, keepdim=True)

    fg = (x < 0.98).float()

    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)/4.0
    ky = kx.transpose(2,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    edges = torch.sqrt(gx*gx + gy*gy)
    edges = (edges > 0.05).float()

    w = 1.0 + alpha*fg + beta*edges
    w = w / (w.mean(dim=[1,2,3], keepdim=True) + 1e-6)
    return w  # [B,1,H,W]


# -----------------------
# Training
# -----------------------
def main():
    set_seed(42)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    channels_last = True

    # --- data cfg (256² debug; move to 512² on the 24GB server) ---
    data_root = "./AMSL Dataset"
    img_size = 256
    grayscale = True
    batch_size = 12 if use_cuda else 4
    num_workers = 0  # Windows-safe
    attn_scales = (img_size // 8, img_size // 32) if USE_ATTENTION else ()

    # --- model cfg ---
    mcfg = HybridVAEConfig(
        img_channels=1 if grayscale else 3,
        img_size=img_size,
        base_channels=48,          # use 64 at 512²
        latent_dim=64,
        use_coordconv=True,
        use_unet_skips=True,
        drop_skip_p=0.0,
        skip_gate=1.0,
        attn_scales=attn_scales,
        down_method="conv",
        up_method="nearest+conv",
    )

    # --- datasets & loaders ---
    train_cfg = AMSLQuadsConfig(root=data_root, split="train", img_size=img_size,
                                grayscale=grayscale, include_annotations=False, max_items=4000)
    val_cfg   = AMSLQuadsConfig(root=data_root, split="val",   img_size=img_size,
                                grayscale=grayscale, include_annotations=False, max_items=800)

    train_ds, train_dl = build_dataloader(train_cfg, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=use_cuda,
                                          persistent_workers=False)
    val_ds, val_dl = build_dataloader(val_cfg, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=use_cuda,
                                      persistent_workers=False)

    # --- model ---
    model = HybridVAE(mcfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Optional: per-channel dropout on skip (if field isn't in dataclass)
    if not hasattr(model.decoder.cfg, "skip_channel_dropout_p"):
        setattr(model.decoder.cfg, "skip_channel_dropout_p", 0.2)
    else:
        model.decoder.cfg.skip_channel_dropout_p = 0.2

    opt = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
    # prefer BF16 if supported (no scaler needed for BF16)
    AMP_DTYPE = torch.bfloat16 if (USE_AMP and use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    USE_SCALER = (USE_AMP and use_cuda and AMP_DTYPE is torch.float16)
    scaler = GradScaler('cuda', enabled=USE_SCALER)
    ema = EMA(model.decoder, decay=0.999)

    results_dir = Path("results/debug256"); results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- training knobs ---
    epochs = 8
    w_l1 = 1.0
    w_ssim = 0.5
    gamma = 15.0    # capacity strength (raise to 20–50 if KL undershoots)
    C_max = 3.5     # target KL at 256²; raise at 512²
    C_warmup = 400  # steps before capacity ramps up

    global_step = 0
    best_val_ssim = -1.0
    total_train_steps = epochs * max(1, len(train_dl))

    for epoch in range(1, epochs + 1):
        model.train()
        # Apply SkipWarmup schedule
        drop_p, gate = skip_schedule(epoch)
        model.decoder.cfg.drop_skip_p = float(drop_p)
        model.decoder.cfg.skip_gate = float(gate)

        t0 = time.time()
        run_l1 = run_ssim = run_kl = run_total = 0.0
        n_samples = 0

        for batch in train_dl:
            x = batch["images"].to(device, non_blocking=True)
            if channels_last: x = x.to(memory_format=torch.channels_last)

            # ---- encode (AMP) ----
            with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(USE_AMP and use_cuda)):
                mu, logvar, feats = model.encode(x)

            # ---- reparameterize in fp32 (inside model) ----
            z = model.reparameterize(mu, logvar)

            # ---- decode (AMP) ----
            with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(USE_AMP and use_cuda)):
                recon = model.decode(z, feats)

            # ---- convert to fp32 for all losses ----
            x32, recon32 = x.float(), recon.float()
            mu32, logvar32 = mu.float(), logvar.float()

            # NaN guard: if AMP decode produced NaNs, redo decode in fp32 for this batch
            if not torch.isfinite(recon32).all():
                with autocast(device_type="cuda", enabled=False):
                    recon = model.decode(z, feats)
                recon32 = recon.float()

            # Weighted L1 + MS-SSIM (both fp32)
            wmap = make_weight_map(x32)                            # [B,1,H,W]
            l1_w = (wmap * (recon32 - x32).abs()).mean()
            ssim_val = ms_ssim(recon32, x32, data_range=1.0, size_average=True)

            # KL in fp32 (model clamps logvar internally)
            kl = model.kl_divergence(mu32, logvar32).mean()

            # recon + KL capacity
            recon_loss = w_l1 * l1_w + w_ssim * (1.0 - ssim_val)
            C_t = kl_capacity(global_step, total_train_steps, C_max=C_max, warmup_steps=C_warmup)
            loss = recon_loss + gamma * torch.abs(kl - C_t)

            opt.zero_grad(set_to_none=True)
            if USE_SCALER:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if USE_SCALER:
                scaler.step(opt); scaler.update()
            else:
                opt.step()

            ema.update(model.decoder)

            # stats
            bsz = x.size(0)
            n_samples += bsz
            run_l1 += float(l1_w.detach()) * bsz
            run_ssim += float(ssim_val.detach()) * bsz
            run_kl += float(kl.detach()) * bsz
            run_total += float(loss.detach()) * bsz

            if (global_step % 500) == 0:
                with torch.no_grad():
                    tmp_dec = type(model.decoder)(model.cfg).to(device)
                    tmp_dec.load_state_dict(model.decoder.state_dict(), strict=True)
                    ema.copy_to(tmp_dec)
                    mu_, logvar_, feats_ = model.encode(x[:8])
                    recon_ema = tmp_dec(mu_, feats_)
                    grid = torch.cat([x[:8], recon_ema[:8]], dim=0)
                    save_grid_img(grid, results_dir / f"train_recon_step{global_step}.png", nrow=8)

            global_step += 1

        # epoch aggregates
        n = max(1, n_samples)
        tr_l1 = run_l1 / n
        tr_ssim = run_ssim / n
        tr_kl = run_kl / n
        tr_total = run_total / n

        # ---- validation (EMA decoder) ----
        model.eval()
        val_l1 = val_ssim = val_psnr = 0.0
        v_count = 0

        # swap in EMA weights for decoder
        dec_backup = type(model.decoder)(model.cfg).to(device)
        dec_backup.load_state_dict(model.decoder.state_dict(), strict=True)
        ema.copy_to(model.decoder)

        with torch.no_grad():
            for batch in val_dl:
                x = batch["images"].to(device, non_blocking=True)
                if channels_last: x = x.to(memory_format=torch.channels_last)

                mu, logvar, feats = model.encode(x)
                recon = model.decode(mu, feats)

                x32, recon32 = x.float(), recon.float()
                wmap = make_weight_map(x32)
                l1v_w = (wmap * (recon32 - x32).abs()).mean(dim=[1,2,3])
                ssimv = ms_ssim(recon32, x32, data_range=1.0, size_average=False)
                psnrv = psnr(recon32, x32)

                val_l1 += float(l1v_w.sum())
                val_ssim += float(ssimv.sum())
                val_psnr += float(psnrv.sum())
                v_count += x.size(0)

            val_l1 /= v_count
            val_ssim /= v_count
            val_psnr /= v_count

            # preview grid
            x = next(iter(val_dl))["images"].to(device, non_blocking=True)
            if channels_last: x = x.to(memory_format=torch.channels_last)
            mu, logvar, feats = model.encode(x[:8])
            recon = model.decode(mu, feats)
            save_grid_img(torch.cat([x[:8], recon[:8]], dim=0), results_dir / f"val_recon_e{epoch}.png", nrow=8)

            # latent diagnostics
            mu_dbg, logvar_dbg, _ = model.encode(x[:batch_size])
            kld_dim = kl_per_dim(mu_dbg.float(), logvar_dbg.float())
            active = count_active_units(kld_dim, tau=0.01)
            mean_kld = float(kld_dim.mean().detach().cpu())

        # restore non-EMA decoder
        model.decoder.load_state_dict(dec_backup.state_dict(), strict=True)

        dt = time.time() - t0
        C_now = kl_capacity(global_step, total_train_steps, C_max=C_max, warmup_steps=C_warmup)
        print(
            f"[E{epoch:02d}] {dt:5.1f}s  "
            f"train: L1={tr_l1:.5f} SSIM={tr_ssim:.4f} KL={tr_kl:.2f}  "
            f"val: L1={val_l1:.5f} SSIM={val_ssim:.4f} PSNR={val_psnr:.2f}dB  "
            f"| latent: active={active}/{mcfg.latent_dim} meanKL={mean_kld:.3f}  "
            f"| C_now={C_now:.2f}  | drop_skip_p={model.decoder.cfg.drop_skip_p:.2f} gate={model.decoder.cfg.skip_gate:.2f}"
        )

        # save best by val SSIM
        ckpt_path = ckpt_dir / "hybridvae_debug256_best.pt"
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            torch.save({"cfg": mcfg.__dict__, "model": model.state_dict(), "best_val_ssim": best_val_ssim}, ckpt_path)
            print(f"  ↳ saved {ckpt_path} (best SSIM={best_val_ssim:.4f})")

    print("Done.")

if __name__ == "__main__":
    main()
