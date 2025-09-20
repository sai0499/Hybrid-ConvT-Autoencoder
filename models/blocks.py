# Reusable building blocks for the hybrid convolutional–transformer VAE
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


# ----------------------------
# helpers: norm / act / init
# ----------------------------
def get_norm(norm: Literal["identity", "bn", "in", "gn"] = "gn", num_channels: int = 32) -> nn.Module:
    if norm == "identity":
        return nn.Identity()
    if norm == "bn":
        return nn.BatchNorm2d(num_channels, eps=1e-5, momentum=0.1)
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, eps=1e-5, affine=True)
    if norm == "gn":
        # use 32 groups if divisible, else fallback to 16 / 8 / 4 / 1
        for g in (32, 16, 8, 4, 1):
            if num_channels % g == 0:
                return nn.GroupNorm(g, num_channels, eps=1e-5, affine=True)
        return nn.GroupNorm(1, num_channels)  # safe fallback
    raise ValueError(f"Unknown norm: {norm}")


def get_act(act: Literal["relu", "gelu", "silu", "identity"] = "silu") -> nn.Module:
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {act}")


def kaiming_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------
# coordinate utilities / CoordConv
# ---------------------------------
@torch.no_grad()
def _make_coord_grid(h: int, w: int, device, dtype, add_radius: bool = True) -> Tensor:
    """
    Returns [3,H,W] or [2,H,W] (x, y, r) normalized to [-1, 1].
    x increases to the right, y increases downward (image convention).
    """
    y = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    if add_radius:
        rr = torch.sqrt(xx**2 + yy**2).clamp(max=1.0)
        return torch.stack([xx, yy, rr], dim=0)  # [3,H,W]
    return torch.stack([xx, yy], dim=0)  # [2,H,W]


class CoordConv2d(nn.Module):
    """
    Conv2d with coordinate channels concatenated (x, y, optional radius).
    Adds 2 or 3 channels before the conv. Great for absolute position awareness.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int = 1,
        padding: int | Tuple[int, int] | None = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        add_radius: bool = True,
    ):
        super().__init__()
        self.add_radius = add_radius
        extra = 3 if add_radius else 2
        if padding is None:
            # SAME padding heuristic
            if isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + extra,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.apply(kaiming_init)

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        grid = _make_coord_grid(h, w, x.device, x.dtype, add_radius=self.add_radius)
        grid = grid.expand(b, -1, -1, -1)  # [B,2/3,H,W]
        x = torch.cat([x, grid], dim=1)
        return self.conv(x)


# -----------------
# ConvNormAct block
# -----------------
class ConvNormAct(nn.Module):
    """
    Convenience: Conv2d -> Norm -> Activation
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        norm: str = "gn",
        act: str = "silu",
        bias: Optional[bool] = None,
    ):
        super().__init__()
        if p is None:
            p = k // 2
        if bias is None:
            # if followed by norm, bias not essential; keep it False for BN/GN stability
            bias = (norm == "identity")
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=bias)
        self.norm = get_norm(norm, out_ch)
        self.act = get_act(act)
        self.apply(kaiming_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


# -------------------------
# Squeeze-and-Excitation
# -------------------------
class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.fc(self.avg(x))
        return x * w


# -----------------
# Residual block
# -----------------
class ResidualBlock(nn.Module):
    """
    ResBlock with GN + SiLU, optional Squeeze-Excitation and dropout.
    Uses 1x1 projection skip when in/out channels differ.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: str = "gn",
        act: str = "silu",
        dropout: float = 0.0,
        use_se: bool = False,
    ):
        super().__init__()
        self.use_proj = in_ch != out_ch
        self.block1 = ConvNormAct(in_ch, out_ch, k=3, s=1, norm=norm, act=act)
        self.block2 = ConvNormAct(out_ch, out_ch, k=3, s=1, norm=norm, act=act)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.se = SELayer(out_ch) if use_se else nn.Identity()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1) if self.use_proj else nn.Identity()
        self.apply(kaiming_init)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.se(x)
        if self.use_proj:
            identity = self.proj(identity)
        return x + identity


# --------------------------
# 2D Multi-Head Self-Attn
# --------------------------
def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device, dtype) -> Tensor:
    """
    Standard 2-D sine-cos positional encoding (ViT-style), shape [H*W, dim].
    """
    def get_1d_pos_embed(length, d):
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)  # [L,1]
        div_term = torch.exp(torch.arange(0, d, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d))
        pe = torch.zeros(length, d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [L,d]

    d_h = dim // 2
    d_w = dim - d_h
    pe_h = get_1d_pos_embed(h, d_h)  # [H, d_h]
    pe_w = get_1d_pos_embed(w, d_w)  # [W, d_w]
    pe = (
        pe_h[:, None, :].expand(h, w, d_h),
        pe_w[None, :, :].expand(h, w, d_w),
    )
    pe = torch.cat(pe, dim=-1).reshape(h * w, dim)  # [H*W, dim]
    return pe


# models/blocks.py (replace the class SelfAttention2d with this)
class SelfAttention2d(nn.Module):
    """
    Multi-Head Self-Attention over 2D maps (B,C,H,W).
    Numerically stable: compute in float32, clip logits before softmax, cast back.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pos_enc: Literal["none", "sine"] = "sine",
    ):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.c = channels
        self.h = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_enc = pos_enc

        self.apply(kaiming_init)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)         # [B, HW, C]
        dtype_in = x_flat.dtype

        # do attention math in float32 for stability
        x32 = x_flat.to(torch.float32)

        if self.pos_enc == "sine":
            pe = _build_2d_sincos_pos_embed(h, w, self.c, x.device, torch.float32)  # [HW, C]
            x32 = x32 + pe.unsqueeze(0)

        qkv = self.qkv(x32)                            # [B, HW, 3C] (fp32)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to heads
        def split_heads(t: Tensor) -> Tensor:
            return t.view(b, -1, self.h, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # scaled dot-product (fp32) with logit stabilization
        scores = (q @ k.transpose(-2, -1)) * self.scale         # [B, H, N, N]
        scores = scores - scores.amax(dim=-1, keepdim=True)     # subtract max over last dim
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        y = attn @ v                                            # [B, H, N, D]

        y = y.transpose(1, 2).contiguous().view(b, -1, self.c)  # [B, N, C]
        y = self.proj(y)
        y = self.proj_drop(y)

        # cast back to original dtype and shape
        y = y.to(dtype_in).transpose(1, 2).reshape(b, c, h, w)
        return y

# -------------------
# Downsample / Blur
# -------------------
class BlurPool2d(nn.Module):
    """
    Anti-aliased downsampling: blur with fixed kernel then stride-2 subsample.
    """
    def __init__(self, channels: int, filt_size: int = 3, stride: int = 2):
        super().__init__()
        assert stride in (2, 3), "BlurPool stride typically 2 or 3"
        if filt_size == 3:
            kernel_1d = torch.tensor([1., 2., 1.])
        elif filt_size == 5:
            kernel_1d = torch.tensor([1., 4., 6., 4., 1.])
        else:
            raise ValueError("filt_size must be 3 or 5")
        filt = kernel_1d[:, None] * kernel_1d[None, :]
        filt = filt / filt.sum()
        self.register_buffer("filt", filt[None, None, :, :].repeat(channels, 1, 1, 1))
        self.stride = stride
        self.pad = (filt_size // 2, filt_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.filt, stride=self.stride, padding=self.pad, groups=x.shape[1])


class Downsample2d(nn.Module):
    """
    Downsample with choices:
      - method='conv': 3x3 stride-2 conv (default)
      - method='avg' : AvgPool2d kernel=2, stride=2
      - method='max' : MaxPool2d kernel=2, stride=2
      - method='blurpool': anti-aliased blur + subsample
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        method: Literal["conv", "avg", "max", "blurpool"] = "conv",
        norm: str = "gn",
        act: str = "silu",
    ):
        super().__init__()
        self.method = method
        if method == "conv":
            self.op = ConvNormAct(in_ch, out_ch, k=3, s=2, norm=norm, act=act)
        elif method in ("avg", "max"):
            pool = nn.AvgPool2d(2) if method == "avg" else nn.MaxPool2d(2)
            self.op = nn.Sequential(
                pool,
                ConvNormAct(in_ch, out_ch, k=1, s=1, norm=norm, act=act),
            )
        elif method == "blurpool":
            self.op = nn.Sequential(
                BlurPool2d(in_ch, filt_size=5, stride=2),
                ConvNormAct(in_ch, out_ch, k=1, s=1, norm=norm, act=act),
            )
        else:
            raise ValueError(f"Unknown downsample method: {method}")

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


# -------------------
# Upsample
# -------------------
class Upsample2d(nn.Module):
    """
    Upsample with choices:
      - method='nearest+conv' (default): NN upsample ×2 then 3x3 conv
      - method='conv_transpose' : 4x4 stride-2 transposed conv
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        method: Literal["nearest+conv", "conv_transpose"] = "nearest+conv",
        norm: str = "gn",
        act: str = "silu",
    ):
        super().__init__()
        self.method = method
        if method == "nearest+conv":
            self.op = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest"),
                ConvNormAct(in_ch, out_ch, k=3, s=1, norm=norm, act=act),
            )
        elif method == "conv_transpose":
            # Careful with checkerboard artifacts; recommended only at lower scales.
            self.op = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                get_norm(norm, out_ch),
                get_act(act),
            )
            self.apply(kaiming_init)
        else:
            raise ValueError(f"Unknown upsample method: {method}")

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)
