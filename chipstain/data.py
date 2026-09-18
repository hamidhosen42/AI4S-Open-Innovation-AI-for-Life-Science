"""Dataset utilities for the HeLa "Kyoto" paired bright-field / H2B dataset.

Layout (after scripts/download_data.py):
    data/raw/hela_kyoto/{train,test}/<name>_{bf,fluo,nuclei,...}.tif
    fluo.tif has 2 channels: [0]=EGFP-alpha-tubulin, [1]=mCherry-H2B (nuclei)

Naming: "20210904_TL2 - R05-C05-F12-050_bf.tif" -> well R05-C05, field F12, timepoint 050.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

NUCLEI_CHANNEL = 1  # index into *_fluo.tif
# Global target normalisation (train-set percentiles of the H2B channel).
TARGET_LO, TARGET_HI = 600.0, 20000.0

_NAME_RE = re.compile(r"(R\d+-C\d+)-F(\d+)-(\d+)_bf\.tif$")


@dataclass
class Sample:
    base: str  # path prefix without "_<suffix>.tif"
    well: str
    field: int
    timepoint: int


def list_samples(split_dir: str) -> list[Sample]:
    out = []
    for f in sorted(glob.glob(os.path.join(split_dir, "*_bf.tif"))):
        m = _NAME_RE.search(f)
        if not m:
            continue
        out.append(Sample(f[: -len("_bf.tif")], m.group(1), int(m.group(2)), int(m.group(3))))
    return out


def make_splits(root: str, val_fields: tuple[int, ...] = (20, 21, 22, 23, 24)):
    """Well-level test split (separate well) + field-level validation split inside train wells.

    All timepoints of a field go to the same side, so no near-duplicate leakage.
    """
    train_all = list_samples(os.path.join(root, "train"))
    test = list_samples(os.path.join(root, "test"))
    val = [s for s in train_all if s.field in val_fields]
    train = [s for s in train_all if s.field not in val_fields]
    return train, val, test


def normalize_input(bf: np.ndarray) -> np.ndarray:
    """Per-image robust z-score so the model is agnostic to camera offset/gain."""
    bf = bf.astype(np.float32)
    lo, hi = np.percentile(bf, [1, 99])
    return (np.clip((bf - lo) / max(hi - lo, 1e-6), -0.5, 1.5) * 2 - 1).astype(np.float32)  # roughly [-2, 2]


def normalize_target(fluo: np.ndarray) -> np.ndarray:
    fluo = fluo.astype(np.float32)
    return np.clip((fluo - TARGET_LO) / (TARGET_HI - TARGET_LO), 0.0, 1.0).astype(np.float32)


def denormalize_target(x: np.ndarray) -> np.ndarray:
    return x * (TARGET_HI - TARGET_LO) + TARGET_LO


def load_pair(s: Sample):
    bf = tifffile.imread(s.base + "_bf.tif")
    fl = tifffile.imread(s.base + "_fluo.tif")[NUCLEI_CHANNEL]
    return bf, fl


def load_mask(s: Sample) -> np.ndarray:
    return tifffile.imread(s.base + "_nuclei.tif").astype(np.int32)


def pad_to_multiple(x: np.ndarray, m: int = 32):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph == 0 and pw == 0:
        return x, (0, 0)
    pad = [(0, 0)] * (x.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(x, pad, mode="reflect"), (ph, pw)


class PairDataset(Dataset):
    """Returns (bf[1,H,W], target[1,H,W]) tensors. Random crops + dihedral aug in train mode."""

    def __init__(self, samples: list[Sample], crop: int | None = 256, augment: bool = True, cache: bool = True):
        self.samples = samples
        self.crop = crop
        self.augment = augment
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.samples)

    def _get(self, i):
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        bf, fl = load_pair(self.samples[i])
        x, y = normalize_input(bf), normalize_target(fl)
        if self._cache is not None:
            self._cache[i] = (x, y)
        return x, y

    def __getitem__(self, i):
        x, y = self._get(i)
        if self.crop:
            h, w = x.shape
            r, c = np.random.randint(0, h - self.crop + 1), np.random.randint(0, w - self.crop + 1)
            x, y = x[r : r + self.crop, c : c + self.crop], y[r : r + self.crop, c : c + self.crop]
        else:
            x, _ = pad_to_multiple(x)
            y, _ = pad_to_multiple(y)
        if self.augment:
            k = np.random.randint(4)
            x, y = np.rot90(x, k), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = np.fliplr(x), np.fliplr(y)
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.from_numpy(np.ascontiguousarray(y))[None]
