from __future__ import annotations
from typing import List, Tuple
import torch
from torch import nn, Tensor

def maybe_sn(module: nn.Module, use: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if use else module

def conv_sn(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, use_sn: bool = True) -> nn.Conv2d:
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
    return maybe_sn(conv, use_sn)

class DBlock(nn.Module):
    """
    Basic discriminator block:
      Conv(k3,s=down) -> LeakyReLU -> Conv(k3,s=1) -> LeakyReLU
    """
    def __init__(self, in_ch: int, out_ch: int, down: bool = True, use_sn: bool = True):
        super().__init__()
        self.conv1 = conv_sn(in_ch, out_ch, k=3, s=(2 if down else 1), p=1, use_sn=use_sn)
        self.conv2 = conv_sn(out_ch, out_ch, k=3, s=1, p=1, use_sn=use_sn)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x

class PatchDiscriminator(nn.Module):
    """
    Spectral-norm PatchGAN that returns (logits, feature_list).
    logits shape ~ [B, 1, H', W'] for hinge/least-squares GAN objectives.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        n_layers: int = 4,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        ch = in_channels
        c = base_channels
        blocks: List[nn.Module] = []
        for i in range(n_layers):
            # Downsample on all but the last block to reach ~70x70 receptive field
            down = (i < n_layers - 1)
            blocks.append(DBlock(ch, c, down=down, use_sn=use_spectral_norm))
            ch = c
            c = min(c * 2, 512)
        self.blocks = nn.ModuleList(blocks)
        self.head = conv_sn(ch, 1, k=3, s=1, p=1, use_sn=use_spectral_norm)

        # lightweight init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        feats: List[Tensor] = []
        h = x
        for blk in self.blocks:
            h = blk(h)
            feats.append(h)
        logits = self.head(h)
        return logits, feats

if __name__ == "__main__":
    # quick sanity test
    d = PatchDiscriminator(in_channels=1, base_channels=64, n_layers=4, use_spectral_norm=True)
    x = torch.randn(2, 1, 256, 256)
    logits, feats = d(x)
    print("logits", logits.shape, "feats", [f.shape for f in feats])
