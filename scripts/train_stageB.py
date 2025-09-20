# scripts/train_stageB.py
from __future__ import annotations
import sys, time, random, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.amp import autocast, GradScaler
from torchvision.utils import save_image, make_grid
from pytorch_msssim import ms_ssim
import argparse

# repo modules
from data.amsl_quads import AMSLQuadsConfig, build_dataloader
from models.hybrid_vae import HybridVAE, HybridVAEConfig
from models.patch_discriminator import PatchDiscriminator
from losses.perceptual_lpips import LPIPSLoss
from losses.ssim import MSSSIMLoss

# speed / stability
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------- utils ----------------
def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_grid(t: Tensor, path: Path, nrow=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(make_grid(t, nrow=nrow, padding=2), str(path))

def psnr(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3]) + eps
    return 10.0 * torch.log10(1.0 / mse)

def make_weight_map(x: Tensor, alpha: float = 2.0, beta: float = 1.5) -> Tensor:
    """
    Boost foreground (dark pixels) and edges to fight background dominance.
    Expects x in [0,1], shape [B,1,H,W] preferred (will average channels if C!=1).
    """
    if x.shape[1] != 1:
        x = x.mean(dim=1, keepdim=True)

    # Sobel edges
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                      device=x.device, dtype=x.dtype).view(1,1,3,3)/4.0
    ky = kx.transpose(2,3)
    gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
    edges = (gx.abs() + gy.abs() > 0.05).float()

    fg = (x < 0.98).float()

    w = 1.0 + alpha*fg + beta*edges
    w = w / (w.mean(dim=[1,2,3], keepdim=True) + 1e-6)
    return w

class HingeGANLoss:
    @staticmethod
    def d_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    @staticmethod
    def g_loss(fake_logits: Tensor) -> Tensor:
        return -fake_logits.mean()

class EMA:
    def __init__(self, module: nn.Module, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}
    @torch.no_grad()
    def update(self, module: nn.Module):
        for k, v in module.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def copy_to(self, module: nn.Module):
        module.load_state_dict(self.shadow, strict=True)

def ramp(global_step: int, total_steps: int, v0: float, v1: float, frac: float) -> float:
    """Linear ramp from v0→v1 over (frac*total_steps)."""
    t = min(1.0, global_step / max(1, int(total_steps * frac)))
    return v0 + (v1 - v0) * t

# --------------- trainer ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./AMSL Dataset", type=str)
    ap.add_argument("--img_size", default=256, type=int)
    ap.add_argument("--base", default=48, type=int, help="base channels for VAE at this resolution")
    ap.add_argument("--batch", default=12, type=int)
    ap.add_argument("--epochs", default=8, type=int)
    ap.add_argument("--ckpt", required=True, type=str, help="Stage-A checkpoint (*.pt)")
    ap.add_argument("--lr_g", default=2e-4, type=float)
    ap.add_argument("--lr_d", default=2e-4, type=float)
    ap.add_argument("--freeze_enc_epochs", default=10, type=int)
    ap.add_argument("--prior_prob", default=0.25, type=float, help="probability to include prior samples for D")
    ap.add_argument("--amp", action="store_true", default=True)
    args = ap.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp and torch.cuda.is_available())
    AMP_DTYPE = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    USE_SCALER = (use_amp and AMP_DTYPE is torch.float16)

    # -------- data --------
    is_windows = (os.name == "nt")
    num_workers = 0 if is_windows else 6

    train_cfg = AMSLQuadsConfig(root=args.root, split="train", img_size=args.img_size,
                                grayscale=True, include_annotations=False, max_items=4000)
    val_cfg   = AMSLQuadsConfig(root=args.root, split="val",   img_size=args.img_size,
                                grayscale=True, include_annotations=False, max_items=800)
    train_ds, train_dl = build_dataloader(train_cfg, batch_size=args.batch, shuffle=True,
                                          num_workers=num_workers, pin_memory=(device=="cuda"),
                                          persistent_workers=(device=="cuda" and not is_windows))
    _, val_dl = build_dataloader(val_cfg, batch_size=args.batch, shuffle=False,
                                 num_workers=num_workers, pin_memory=(device=="cuda"),
                                 persistent_workers=(device=="cuda" and not is_windows))

    # -------- generator (VAE) --------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # checkpoint saved cfg may be a dict
    from dataclasses import fields as dataclass_fields
    raw_cfg = ckpt["cfg"] if isinstance(ckpt["cfg"], dict) else ckpt["cfg"].__dict__
    allowed = {f.name for f in dataclass_fields(HybridVAEConfig)}

    cfg_clean = {}
    for k, v in raw_cfg.items():
        if k in allowed:
            # type fix-ups for common serialized types
            if k == "attn_scales" and isinstance(v, list):
                v = tuple(v)
            cfg_clean[k] = v
    # construct config with only allowed keys
    gcfg = HybridVAEConfig(**cfg_clean)

    # override resolution-specific bits for Stage-B
    gcfg.img_size = args.img_size
    gcfg.base_channels = args.base

    vae = HybridVAE(gcfg).to(device).train()
    vae.load_state_dict(ckpt["model"], strict=False)  # allow shape changes if 256→512

    # keep skip active but not dominant
    if hasattr(vae.decoder.cfg, "drop_skip_p"):
        vae.decoder.cfg.drop_skip_p = 0.3
    if hasattr(vae.decoder.cfg, "skip_gate"):
        vae.decoder.cfg.skip_gate = 1.0
    if not hasattr(vae.decoder.cfg, "skip_channel_dropout_p"):
        setattr(vae.decoder.cfg, "skip_channel_dropout_p", 0.15)
    else:
        vae.decoder.cfg.skip_channel_dropout_p = 0.15

    # -------- discriminator --------
    disc = PatchDiscriminator(in_channels=gcfg.img_channels, base_channels=64,
                              n_layers=4, use_spectral_norm=True).to(device).train()

    # -------- losses --------
    lpips_loss = LPIPSLoss(net="vgg").to(device)
    ms_ssim_loss = MSSSIMLoss(data_range=1.0)  # returns 1 - MS-SSIM
    gan = HingeGANLoss()

    # -------- optimizers / scaler / EMA --------
    opt_g = torch.optim.Adam(vae.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    scaler = GradScaler("cuda", enabled=USE_SCALER)
    ema = EMA(vae.decoder, decay=0.999)

    out_dir = Path("results/stageB"); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(parents=True, exist_ok=True)

    iters_per_epoch = max(1, len(train_dl))
    total_iters = args.epochs * iters_per_epoch
    global_step = 0
    best_score = -1e9

    for epoch in range(1, args.epochs + 1):
        # freeze encoder early for stability
        freeze_enc = (epoch <= args.freeze_enc_epochs)
        for p in vae.encoder.parameters():
            p.requires_grad_(not freeze_enc)

        t0 = time.time()
        vae.train(); disc.train()
        agg = {"l1":0.0, "ssim":0.0, "lpips":0.0, "gan_d":0.0, "gan_g":0.0, "fm":0.0, "n":0}

        for b in train_dl:
            x = b["images"].to(device, non_blocking=True)

            # ========== D step ==========
            with torch.no_grad():
                # recon
                with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp):
                    mu, logvar, feats = vae.encode(x)
                z = vae.reparameterize(mu, logvar)            # latent math is fp32 inside model
                with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp):
                    x_rec = vae.decode(z, feats).clamp(0,1)

                # optional prior samples (unconditional)
                if random.random() < args.prior_prob:
                    z_prior = torch.randn(x.size(0), gcfg.latent_dim, device=device)
                    with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp):
                        x_prior = vae.decode(z_prior, feats=None).clamp(0,1)
                    x_fake = torch.cat([x_rec, x_prior], dim=0)
                else:
                    x_fake = x_rec

            for p in disc.parameters(): p.requires_grad_(True)
            opt_d.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp):
                real_logits, _ = disc(x)
                fake_logits, _ = disc(x_fake.detach())
                d_loss = gan.d_loss(real_logits, fake_logits)

            if USE_SCALER:
                scaler.scale(d_loss).backward()
                scaler.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
                scaler.step(opt_d)
            else:
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
                opt_d.step()

            # ========== G step ==========
            for p in disc.parameters(): p.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)

            # forward fresh for G
            with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp):
                mu, logvar, feats = vae.encode(x)
            z = vae.reparameterize(mu, logvar)
            with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp):
                x_rec = vae.decode(z, feats).clamp(0,1)

            # recon/perceptual in fp32
            x32, xr32 = x.float(), x_rec.float()
            wmap = make_weight_map(x32)
            l1 = (wmap * (xr32 - x32).abs()).mean()
            ssim_loss = ms_ssim_loss(xr32, x32)               # 1 - MS-SSIM
            lp = lpips_loss(xr32, x32)

            # GAN + feature matching
            with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp):
                fake_logits, fake_feats = disc(x_rec)
                real_logits, real_feats = disc(x)
            g_gan = gan.g_loss(fake_logits)
            fm = 0.0
            for rf, ff in zip(real_feats, fake_feats):
                fm = fm + (rf.float() - ff.float()).abs().mean()

            # ramps
            lam_gan = ramp(global_step, total_iters, 0.0, 0.25, frac=0.30)   # 0→0.25 over 30% iters
            lam_lp  = ramp(global_step, total_iters, 0.0, 0.20, frac=0.50)   # 0→0.20 over 50% iters
            lam_fm  = 10.0
            lam_l1  = 1.0
            lam_ss  = 0.3

            g_loss = lam_l1*l1 + lam_ss*ssim_loss + lam_lp*lp + lam_gan*g_gan + lam_fm*fm

            if USE_SCALER:
                scaler.scale(g_loss).backward()
                scaler.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                scaler.step(opt_g); scaler.update()
            else:
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                opt_g.step()

            ema.update(vae.decoder)

            # logs
            bsz = x.size(0); agg["n"] += bsz
            agg["l1"]   += float(l1.detach()) * bsz
            agg["ssim"] += float((1.0 - ssim_loss).detach()) * bsz
            agg["lpips"]+= float(lp.detach()) * bsz
            agg["gan_d"]+= float(d_loss.detach()) * bsz
            agg["gan_g"]+= float(g_gan.detach()) * bsz
            agg["fm"]   += float(fm.detach()) * bsz

            global_step += 1

        # epoch summary
        n = max(1, agg["n"])
        tr_l1, tr_ssim = agg["l1"]/n, agg["ssim"]/n
        tr_lp, tr_d, tr_g = agg["lpips"]/n, agg["gan_d"]/n, agg["gan_g"]/n

        # -------- validation with EMA decoder --------
        vae.eval()
        dec_backup = type(vae.decoder)(vae.cfg).to(device)
        dec_backup.load_state_dict(vae.decoder.state_dict(), strict=True)
        ema.copy_to(vae.decoder)

        val_l1 = val_ssim = val_lp = val_psnr = 0.0
        v_count = 0
        with torch.no_grad():
            for b in val_dl:
                x = b["images"].to(device)
                mu, logvar, feats = vae.encode(x)
                xr = vae.decode(mu, feats).clamp(0,1)

                x32, xr32 = x.float(), xr.float()
                wmap = make_weight_map(x32)
                l1v = (wmap * (xr32 - x32).abs()).mean(dim=[1,2,3])
                ssimv = ms_ssim(xr32, x32, data_range=1.0, size_average=False)
                psnrv = psnr(xr32, x32)
                lpv = lpips_loss(xr32, x32)                    # scalar mean over batch

                val_l1 += float(l1v.sum())
                val_ssim += float(ssimv.sum())
                val_psnr += float(psnrv.sum())
                val_lp += float(lpv) * x.size(0)
                v_count += x.size(0)

            val_l1 /= v_count
            val_ssim /= v_count
            val_psnr /= v_count
            val_lp /= v_count

            # preview grid
            x = next(iter(val_dl))["images"].to(device)[:8]
            mu, logvar, feats = vae.encode(x)
            xr = vae.decode(mu, feats).clamp(0,1)
            save_grid(torch.cat([x, xr], dim=0), out_dir / f"val_recon_e{epoch:02d}.png", nrow=8)

        # restore non-EMA decoder
        vae.decoder.load_state_dict(dec_backup.state_dict(), strict=True)

        dt = time.time() - t0
        print(f"[E{epoch:02d}] {dt:5.1f}s  "
              f"train: L1={tr_l1:.4f} SSIM={tr_ssim:.4f} LPIPS={tr_lp:.3f} D={tr_d:.3f} G={tr_g:.3f}  "
              f"val: L1={val_l1:.4f} SSIM={val_ssim:.4f} LPIPS={val_lp:.3f} PSNR={val_psnr:.2f}dB")

        # save best by SSIM - 0.5*LPIPS
        score = val_ssim - 0.5 * val_lp
        best_path = Path("checkpoints/hybridvae_stageB_best.pt")
        torch.save({
            "cfg": vae.cfg.__dict__,
            "model": vae.state_dict(),
            "disc": disc.state_dict(),
            "epoch": epoch,
            "score": score
        }, best_path)
        print(f"  ↳ saved {best_path} (score={score:.4f})")

    print("Done.")

if __name__ == "__main__":
    main()
