# data/amsl_quads.py
# Robust loader for AMSL quadrilateral dataset
# Layout:
# ./AMSL Dataset/
#   sized_rectangles_filled/
#     annotations/*.xml
#     train/*.bmp  val/*.bmp  test/*.bmp
#   sized_rectangles_unfilled/
#   sized_squares_filled/
#   sized_squares_unfilled/

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import torch
from torchvision.transforms import InterpolationMode
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

__all__ = [
    "AMSLQuads",
    "AMSLQuadsConfig",
    "build_dataloader",
    "SUBFOLDERS",
    "AE_collate",
]

# Four families present in your dataset
SUBFOLDERS = [
    "sized_rectangles_filled",
    "sized_rectangles_unfilled",
    "sized_squares_filled",
    "sized_squares_unfilled",
]

IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg")  # .bmp is primary; others allowed just in case


@dataclass(frozen=True)
class AMSLQuadsConfig:
    root: Union[str, Path] = "./AMSL Dataset"
    split: str = "train"               # "train" | "val" | "test"
    img_size: int = 512
    grayscale: bool = True             # for AE, grayscale is fine; set False if you want RGB
    include_annotations: bool = False  # if True, parse XML and return bbox/meta
    include_meta: bool = True          # always handy for debugging
    # Optional filters
    families: Optional[List[str]] = None  # subset of SUBFOLDERS; None = all
    max_items: Optional[int] = None       # cap for quick tests


def _parse_voc_xml(xml_path: Path) -> Dict:
    """
    Parse a typical PASCAL VOC-style XML.
    Returns a dict with image size and a list of objects (name + bbox).
    Robust to minor variations; missing fields -> best-effort.
    """
    info = {"objects": []}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # size
        size = root.find("size")
        if size is not None:
            w = size.findtext("width")
            h = size.findtext("height")
            c = size.findtext("depth")
            info["width"] = int(w) if w else None
            info["height"] = int(h) if h else None
            info["channels"] = int(c) if c else None

        for obj in root.findall("object"):
            name = obj.findtext("name") or "object"
            bb = obj.find("bndbox")
            bbox = None
            if bb is not None:
                try:
                    xmin = int(float(bb.findtext("xmin")))
                    ymin = int(float(bb.findtext("ymin")))
                    xmax = int(float(bb.findtext("xmax")))
                    ymax = int(float(bb.findtext("ymax")))
                    bbox = (xmin, ymin, xmax, ymax)
                except Exception:
                    bbox = None
            info["objects"].append({"name": name, "bbox": bbox})
    except Exception:
        # If parsing fails, return minimal info
        info["parse_error"] = True
    return info


class AMSLQuads(Dataset):
    """
    Torch Dataset for the AMSL quadrilateral images.

    Returns a dict:
      {
        "image": Tensor [C,H,W] in [0,1],
        "path": str,
        "family": str,     # e.g. sized_squares_filled
        "split": str,      # train/val/test
        "annotation": dict or None
      }
    """
    def __init__(self, cfg: AMSLQuadsConfig):
        self.cfg = cfg
        self.root = Path(cfg.root)
        if not self.root.exists():
            raise FileNotFoundError(f"Root not found: {self.root.resolve()}")

        if cfg.split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train|val|test, got {cfg.split}")

        families = cfg.families or SUBFOLDERS
        files: List[Tuple[Path, str]] = []  # (image_path, family)

        for fam in families:
            img_dir = self.root / fam / cfg.split
            if not img_dir.exists():
                raise FileNotFoundError(f"Missing dir: {img_dir}")
            for p in img_dir.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    files.append((p, fam))

        if not files:
            raise RuntimeError(f"No images found under {self.root}/*/{cfg.split} with exts {IMG_EXTS}")

        # Stable order for reproducibility
        files.sort(key=lambda x: str(x[0]).lower())

        if cfg.max_items is not None:
            files = files[: cfg.max_items]

        self.files = files

        # Transforms
        tf: List[torch.nn.Module] = [
            T.Resize((cfg.img_size, cfg.img_size), interpolation=InterpolationMode.NEAREST)
        ]
        if cfg.grayscale:
            tf.append(T.Grayscale(num_output_channels=1))
        tf.append(T.ToTensor())  # <-- NOTE the parentheses
        self.transform = T.Compose(tf)

    def __len__(self) -> int:
        return len(self.files)

    def _ann_path_for(self, img_path: Path, family: str) -> Path:
        stem = img_path.stem  # filename without extension
        return self.root / family / "annotations" / f"{stem}.xml"

    def __getitem__(self, idx: int) -> Dict:
        img_path, family = self.files[idx]

        # Always open via PIL; convert('RGB') keeps loader robust
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            tensor: Tensor = self.transform(im)  # [C,H,W] float32 in [0,1]

        sample: Dict = {
            "image": tensor,
            "path": str(img_path),
            "family": family,
            "split": self.cfg.split,
        }

        if self.cfg.include_annotations:
            xml_path = self._ann_path_for(img_path, family)
            sample["annotation"] = _parse_voc_xml(xml_path) if xml_path.exists() else None

        if self.cfg.include_meta:
            # Quick metadata flags
            sample["is_square"] = "squares" in family
            sample["is_filled"] = "filled" in family

        return sample


def AE_collate(batch: List[Dict]) -> Dict[str, Union[Tensor, List]]:
    """
    Collate for autoencoder training:
      returns dict with "images" tensor [B,C,H,W] and lists for optional fields.
    """
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    paths = [b["path"] for b in batch]
    families = [b["family"] for b in batch]
    anns = [b.get("annotation") for b in batch]
    return {"images": imgs, "paths": paths, "families": families, "annotations": anns}


def build_dataloader(
    cfg: AMSLQuadsConfig,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> Tuple[AMSLQuads, DataLoader]:
    ds = AMSLQuads(cfg)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=AE_collate,
        drop_last=False,
    )
    return ds, dl
