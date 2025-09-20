from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Tuple

import torch
from torch import nn, Tensor

from .blocks import (
    CoordConv2d, ConvNormAct, ResidualBlock, SelfAttention2d,
    Downsample2d, Upsample2d, get_norm, get_act
)


@dataclass
class HybridVAEConfig:
    img_channels: int = 1
    img_size: int = 512  # must be divisible by 32
    base_channels: int = 64
    latent_dim: int = 64
    norm: str = "gn"
    act: str = "silu"

    use_coordconv: bool = True
    use_unet_skips: bool = True
    drop_skip_p: float = 0.5  # probability to IGNORE skip during training
    skip_gate: float = 1.0  # multiplicative gate (we can ramp this up)
    skip_channel_dropout_p: float = 0.2
    # Interpret these as ABSOLUTE spatial sizes where to place attention.
    # To make it resolution-agnostic, you can pass (img_size//8, img_size//32)
    attn_scales: Tuple[int, ...] = (64, 16)
    down_method: str = "conv"
    up_method: str = "nearest+conv"

    def __post_init__(self):
        assert self.img_size % 32 == 0, "img_size must be divisible by 32"
        # Derived scales
        self.skip_hw = self.img_size // 8  # 64 for 512, 32 for 256
        self.bot_hw = self.img_size // 32  # 16 for 512, 8 for 256


class Encoder(nn.Module):
    """
    512→256→128→64→32→16 (or 256→128→64→32→16→8)
    Saves a skip feature at cfg.skip_hw.
    """

    def __init__(self, cfg: HybridVAEConfig):
        super().__init__()
        C = cfg.base_channels
        self.cfg = cfg

        # stem (S -> S/2)
        if cfg.use_coordconv:
            self.stem = nn.Sequential(
                CoordConv2d(cfg.img_channels, C, kernel_size=7, stride=2, padding=3),
                get_norm(cfg.norm, C),
                get_act(cfg.act),
            )
        else:
            self.stem = ConvNormAct(cfg.img_channels, C, k=7, s=2, norm=cfg.norm, act=cfg.act)

        #  S/2 -> S/4
        self.down1 = nn.Sequential(
            ResidualBlock(C, C, norm=cfg.norm, act=cfg.act),
            Downsample2d(C, C * 2, method=cfg.down_method, norm=cfg.norm, act=cfg.act),
        )  # out: 2C

        #  S/4 -> S/8   (this is the skip scale)
        self.down2 = nn.Sequential(
            ResidualBlock(C * 2, C * 2, norm=cfg.norm, act=cfg.act),
            Downsample2d(C * 2, C * 4, method=cfg.down_method, norm=cfg.norm, act=cfg.act),
        )  # out: 4C @ S/8

        self.res_skip = ResidualBlock(C * 4, C * 4, norm=cfg.norm, act=cfg.act)
        self.attn_skip = SelfAttention2d(C * 4, num_heads=8, pos_enc="sine") \
            if (self.cfg.skip_hw in self.cfg.attn_scales) else nn.Identity()

        #  S/8 -> S/16
        self.down3 = Downsample2d(C * 4, C * 4, method=cfg.down_method, norm=cfg.norm, act=cfg.act)

        #  S/16 -> S/16 (widen channels)
        self.res_bot = ResidualBlock(C * 4, C * 8, norm=cfg.norm, act=cfg.act)

        #  S/16 -> S/32 (bottleneck spatial)
        self.down4 = Downsample2d(C * 8, C * 8, method=cfg.down_method, norm=cfg.norm, act=cfg.act)

        self.attn_bot = SelfAttention2d(C * 8, num_heads=8, pos_enc="sine") \
            if (self.cfg.bot_hw in self.cfg.attn_scales) else nn.Identity()

        # latent heads
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.to_mu = nn.Linear(C * 8, cfg.latent_dim)
        self.to_logvar = nn.Linear(C * 8, cfg.latent_dim)
        for m in (self.to_mu, self.to_logvar):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

        with torch.no_grad():
            self.to_logvar.weight.mul_(0.1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, dict]:
        feats = {}
        x = self.stem(x)  # S/2
        x = self.down1(x)  # S/4
        x = self.down2(x)  # S/8

        x = self.res_skip(x)  # S/8
        x = self.attn_skip(x)  # S/8
        if self.cfg.use_unet_skips:
            feats["skip"] = x  # save skip at S/8

        x = self.down3(x)  # S/16
        x = self.res_bot(x)  # S/16 (8C)
        x = self.down4(x)  # S/32
        x = self.attn_bot(x)  # S/32

        pooled = self.gap(x).flatten(1)  # [B,8C]
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        feats["enc_out"] = x
        return mu, logvar, feats


class Decoder(nn.Module):
    """
    Start from cfg.bot_hw × cfg.bot_hw, upsample back to img_size.
    Concatenate skip at cfg.skip_hw right after the second up.
    """

    def __init__(self, cfg: HybridVAEConfig):
        super().__init__()
        C = cfg.base_channels
        self.cfg = cfg

        # from z to spatial bottleneck
        self.from_z = nn.Linear(cfg.latent_dim, C * 8 * cfg.bot_hw * cfg.bot_hw)
        nn.init.kaiming_uniform_(self.from_z.weight, nonlinearity="relu")
        nn.init.zeros_(self.from_z.bias)

        #  S/32 -> S/16
        self.block_bot = ResidualBlock(C * 8, C * 8, norm=cfg.norm, act=cfg.act)
        self.attn_bot = SelfAttention2d(C * 8, num_heads=8, pos_enc="sine") \
            if (self.cfg.bot_hw in self.cfg.attn_scales) else nn.Identity()
        self.up_bot = Upsample2d(C * 8, C * 4, method=cfg.up_method, norm=cfg.norm, act=cfg.act)  # S/16, 4C

        #  S/16 -> S/8
        self.block_mid = ResidualBlock(C * 4, C * 4, norm=cfg.norm, act=cfg.act)
        self.up_mid = Upsample2d(C * 4, C * 4, method=cfg.up_method, norm=cfg.norm, act=cfg.act)  # S/8, 4C

        # fuse skip at S/8
        in_ch_skip = C * 4 * (2 if cfg.use_unet_skips else 1)
        self.fuse_skip = ConvNormAct(in_ch_skip, C * 4, k=3, s=1, norm=cfg.norm, act=cfg.act)
        self.attn_skip = SelfAttention2d(C * 4, num_heads=8, pos_enc="sine") \
            if (self.cfg.skip_hw in self.cfg.attn_scales) else nn.Identity()

        # S/8 -> S/4 -> S/2 -> S
        self.up_s8_to_s4 = Upsample2d(C * 4, C * 2, method=cfg.up_method, norm=cfg.norm, act=cfg.act)  # S/4
        self.block_s4 = ResidualBlock(C * 2, C * 2, norm=cfg.norm, act=cfg.act)
        self.up_s4_to_s2 = Upsample2d(C * 2, C, method=cfg.up_method, norm=cfg.norm, act=cfg.act)  # S/2
        self.block_s2 = ResidualBlock(C, C, norm=cfg.norm, act=cfg.act)
        self.up_s2_to_s = Upsample2d(C, C // 2, method=cfg.up_method, norm=cfg.norm, act=cfg.act)  # S

        self.out = nn.Sequential(
            ConvNormAct(C // 2, C // 2, k=3, s=1, norm=cfg.norm, act=cfg.act),
            nn.Conv2d(C // 2, cfg.img_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor, feats: Optional[dict] = None) -> Tensor:
        B = z.shape[0]
        x = self.from_z(z).view(B, -1, self.cfg.bot_hw, self.cfg.bot_hw)  # S/32

        x = self.block_bot(x)
        x = self.attn_bot(x)
        x = self.up_bot(x)  # S/16

        x = self.block_mid(x)
        x = self.up_mid(x)  # S/8

        use_skip = self.cfg.use_unet_skips and (feats is not None) and ("skip" in feats)

        # default: zeros placeholder (same shape as x) to keep channels constant
        placeholder = torch.zeros_like(x)

        if use_skip:
            keep = True
            if self.training and self.cfg.drop_skip_p > 0.0:
                keep = torch.rand((), device=x.device) > self.cfg.drop_skip_p

            if keep:
                skip = feats["skip"]
                # optional channel dropout on skip to weaken its dominance
                if self.cfg.skip_channel_dropout_p > 0 and self.training:
                    skip = torch.nn.functional.dropout2d(
                        skip, p=self.cfg.skip_channel_dropout_p, training=True
                    )
                if self.cfg.skip_gate != 1.0:
                    skip = skip * self.cfg.skip_gate
                x = torch.cat([x, skip], dim=1)           # [B, 8C, H, W]
            else:
                x = torch.cat([x, placeholder], dim=1)    # [B, 8C, H, W]
        else:
            x = torch.cat([x, placeholder], dim=1)        # [B, 8C, H, W]

        x = self.fuse_skip(x)
        x = self.attn_skip(x)

        x = self.up_s8_to_s4(x)  # S/4
        x = self.block_s4(x)
        x = self.up_s4_to_s2(x)  # S/2
        x = self.block_s2(x)
        x = self.up_s2_to_s(x)  # S

        return self.out(x)


class HybridVAE(nn.Module):
    def __init__(self, cfg: HybridVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # do latent math in fp32 for stability, then cast back
        mu32 = mu.float()
        logvar32 = logvar.float().clamp(-10.0, 10.0)
        std32 = torch.exp(0.5 * logvar32)
        eps32 = torch.randn_like(std32)
        z32 = mu32 + eps32 * std32
        return z32.to(mu.dtype)

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # compute KL in fp32; caller can .mean() or .sum()
        mu32 = mu.float()
        logvar32 = logvar.float().clamp(-10.0, 10.0)
        return 0.5 * torch.sum(torch.exp(logvar32) + mu32 ** 2 - 1.0 - logvar32, dim=1)

    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, z: Tensor, feats: Optional[dict] = None) -> Tensor:
        return self.decoder(z, feats)

    def forward(self, x: Tensor):
        mu, logvar, feats = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, feats)
        return recon, mu, logvar, z

    @staticmethod
    def kl_per_dim(mu: Tensor, logvar: Tensor) -> Tensor:
        # [B,D] -> [D], mean over batch (nats)
        return 0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=0)

    @staticmethod
    def count_active_units(kl_per_dim: Tensor, tau: float = 0.01) -> int:
        # dims with mean KL above tiny threshold
        return int((kl_per_dim > tau).sum().item())
