from pathlib import Path
import sys
# add repo root to import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from models.blocks import (
    CoordConv2d, ResidualBlock, SelfAttention2d,
    Downsample2d, Upsample2d
)

def main():
    x = torch.randn(2, 64, 64, 64)

    cc = CoordConv2d(64, 64, 3)
    y = cc(x); print("CoordConv:", y.shape)

    rb = ResidualBlock(64, 128, dropout=0.1, use_se=True)
    y = rb(y); print("ResBlock:", y.shape)

    down = Downsample2d(128, 256, method="conv")
    y = down(y); print("Down:", y.shape)

    attn = SelfAttention2d(256, num_heads=8, pos_enc="sine")
    y = attn(y); print("Attn:", y.shape)

    up = Upsample2d(256, 128, method="nearest+conv")
    y = up(y); print("Up:", y.shape)

    print("OK")

if __name__ == "__main__":
    main()
