from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from models.hybrid_vae import HybridVAE, HybridVAEConfig

def main():
    cfg = HybridVAEConfig(
        img_channels=1, img_size=512, base_channels=64, latent_dim=64,
        use_coordconv=True, use_unet_skips=True, attn_scales=(64,16),
        down_method="conv", up_method="nearest+conv",
    )
    model = HybridVAE(cfg).cuda() if torch.cuda.is_available() else HybridVAE(cfg)
    model = model.to(memory_format=torch.channels_last)  # perf hint

    x = torch.randn(2, 1, 512, 512, device=next(model.parameters()).device).to(memory_format=torch.channels_last)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        recon, mu, logvar, z = model(x)

    print("Input :", tuple(x.shape))
    print("Recon :", tuple(recon.shape))
    print("mu/logvar/z:", mu.shape, logvar.shape, z.shape)
    # KL per-sample
    kl = model.kl_divergence(mu, logvar)
    print("KL shape:", kl.shape, "mean:", float(kl.mean().detach().cpu()))

if __name__ == "__main__":
    main()
