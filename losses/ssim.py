from __future__ import annotations
from typing import Literal, Optional
import torch
from torch import nn, Tensor

try:
    from pytorch_msssim import ssim as _ssim_fn, ms_ssim as _ms_ssim_fn
except Exception as e:  # pragma: no cover
    _ssim_fn = _ms_ssim_fn = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

def _ensure_available():
    if _ssim_fn is None or _ms_ssim_fn is None:
        raise ImportError(
            "pytorch_msssim is required for SSIM/MS-SSIM. "
            "Install with: pip install pytorch-msssim"
        ) from _IMPORT_ERROR

class SSIMLoss(nn.Module):
    """
    1 - SSIM as a loss. Works with images in [0,1].
    reduction: 'mean' | 'none'
    """
    def __init__(self, data_range: float = 1.0, reduction: Literal["mean","none"] = "mean"):
        super().__init__()
        _ensure_available()
        self.data_range = float(data_range)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # compute per-sample SSIM then convert to loss
        ssim_vals: Tensor = _ssim_fn(
            x.float(), y.float(), data_range=self.data_range, size_average=False
        )  # [B]
        loss = 1.0 - ssim_vals
        if self.reduction == "mean":
            return loss.mean()
        return loss

class MSSSIMLoss(nn.Module):
    """
    1 - MS-SSIM as a loss. Works with images in [0,1].
    reduction: 'mean' | 'none'
    """
    def __init__(self, data_range: float = 1.0, reduction: Literal["mean","none"] = "mean"):
        super().__init__()
        _ensure_available()
        self.data_range = float(data_range)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        ms_vals: Tensor = _ms_ssim_fn(
            x.float(), y.float(), data_range=self.data_range, size_average=False
        )  # [B]
        loss = 1.0 - ms_vals
        if self.reduction == "mean":
            return loss.mean()
        return loss

# Optional convenience functions (values, not losses)
@torch.no_grad()
def ssim_value(x: Tensor, y: Tensor, data_range: float = 1.0, size_average: bool = True) -> Tensor:
    _ensure_available()
    return _ssim_fn(x.float(), y.float(), data_range=data_range, size_average=size_average)

@torch.no_grad()
def ms_ssim_value(x: Tensor, y: Tensor, data_range: float = 1.0, size_average: bool = True) -> Tensor:
    _ensure_available()
    return _ms_ssim_fn(x.float(), y.float(), data_range=data_range, size_average=size_average)
