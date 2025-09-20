from __future__ import annotations
from typing import Literal
import torch
from torch import nn, Tensor

def _to_lpips_space(x: Tensor) -> Tensor:
    """
    Convert images in [0,1] (B,C,H,W) to 3ch in [-1,1] for LPIPS.
    """
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    return x * 2.0 - 1.0

class LPIPSLoss(nn.Module):
    """
    Wrapper around https://github.com/richzhang/PerceptualSimilarity (pip install lpips)
    - Expects inputs in [0,1]; internally maps to 3ch [-1,1].
    - Returns mean LPIPS over the batch.
    """
    def __init__(self, net: Literal["alex","vgg","squeeze"] = "vgg"):
        super().__init__()
        try:
            import lpips  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "LPIPS is not installed. Install with: pip install lpips"
            ) from e

        self.lpips = lpips.LPIPS(net=net)
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        x, y: images in [0,1], shapes [B,1/3,H,W]. Returns scalar loss (mean over batch).
        """
        x_ = _to_lpips_space(x.float())
        y_ = _to_lpips_space(y.float())
        val: Tensor = self.lpips(x_, y_)  # [B,1,1,1] or [B]
        return val.mean()
