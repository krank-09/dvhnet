"""
dataset.py
==========
PyTorch Dataset/DataLoader glue for DVHnet. Expects slice-level examples
(as produced by preprocessing.build_slice_dataset) either held in memory or
cached to disk as .npz shards, keyed by patient so that a patient-level split
can be respected without slice leakage.
"""

from __future__ import annotations

import os
import glob
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object


class DVHSliceDataset(Dataset):
    """
    Each item is:
        input:  [2, H, W]  float32   (channel 0 = target/PTV mask, channel 1 = OAR mask)
        label:  [num_bins] float32   (cumulative DVH, in [0, 1])
        voxel_count: scalar int      (for volume-weighted 2D->3D aggregation at inference)
        patient_id, slice_index: bookkeeping for reconstructing per-patient DVHs
    """

    def __init__(self, examples: List[Dict], augment: bool = False):
        self.examples = examples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        target = ex["target_mask"]
        oar = ex["oar_mask"]

        if self.augment:
            target, oar = self._augment(target, oar)

        inp = np.stack([target, oar], axis=0).astype(np.float32)  # [2, H, W]
        label = ex["dvh_label"].astype(np.float32)                # [num_bins]

        item = {
            "input": torch.from_numpy(inp),
            "label": torch.from_numpy(label),
            "voxel_count": ex["voxel_count"],
            "patient_id": ex["patient_id"],
            "slice_index": ex["slice_index"],
        }
        return item

    @staticmethod
    def _augment(target: np.ndarray, oar: np.ndarray):
        """Geometric augmentation on the mask pair only. The DVH label is derived
        from per-voxel dose counts, not mask shape, so flips are exactly
        label-preserving and small rotations/translations are approximately so
        (rotation ~preserves voxel area with nearest-neighbour resampling;
        translation preserves it exactly modulo boundary clipping). Aggressive
        scaling is deliberately NOT applied -- it changes voxel count enough to
        shift the true DVH the fixed label no longer matches.

        The same transform is applied to `target` and `oar` so they stay
        spatially registered.
        """
        if np.random.rand() < 0.5:
            target = target[:, ::-1]
            oar = oar[:, ::-1]
        if np.random.rand() < 0.5:
            target = target[::-1, :]
            oar = oar[::-1, :]

        if np.random.rand() < 0.5:
            angle = np.random.uniform(-15.0, 15.0)
            target = ndimage.rotate(target, angle, order=0, reshape=False,
                                    mode="constant", cval=0)
            oar = ndimage.rotate(oar, angle, order=0, reshape=False,
                                 mode="constant", cval=0)

        if np.random.rand() < 0.5:
            h, w = target.shape
            sy = np.random.uniform(-0.08, 0.08) * h
            sx = np.random.uniform(-0.08, 0.08) * w
            target = ndimage.shift(target, (sy, sx), order=0,
                                   mode="constant", cval=0)
            oar = ndimage.shift(oar, (sy, sx), order=0, mode="constant", cval=0)

        return np.ascontiguousarray(target), np.ascontiguousarray(oar)


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "input": torch.stack([b["input"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "voxel_count": torch.tensor([b["voxel_count"] for b in batch], dtype=torch.float32),
        "patient_id": [b["patient_id"] for b in batch],
        "slice_index": [b["slice_index"] for b in batch],
    }


def load_examples_from_shards(shard_dir: str, patient_ids: Optional[List[str]] = None) -> List[Dict]:
    """
    Load slice-level examples cached as one .npz per patient (recommended over
    keeping everything in memory during preprocessing). Each shard is expected
    to contain arrays: target_masks [S,H,W], oar_masks [S,H,W], dvh_labels
    [S,num_bins], voxel_counts [S], slice_indices [S].
    """
    examples = []
    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
    for path in shard_paths:
        pid = os.path.splitext(os.path.basename(path))[0]
        if patient_ids is not None and pid not in patient_ids:
            continue
        data = np.load(path)
        n = data["target_masks"].shape[0]
        for i in range(n):
            examples.append({
                "patient_id": pid,
                "slice_index": int(data["slice_indices"][i]),
                # Downcast on load regardless of what's stored on disk: masks
                # are binary and don't need float32's 4x memory -- with ~500
                # patients / 10K+ slice examples held in one Python list
                # (see this function's docstring), float32 masks alone push
                # this past 5GB and OOM-kill on an 8GB machine.
                "target_mask": data["target_masks"][i].astype(np.uint8),
                "oar_mask": data["oar_masks"][i].astype(np.uint8),
                "dvh_label": data["dvh_labels"][i],
                "voxel_count": int(data["voxel_counts"][i]),
            })
    return examples


def save_patient_shard(shard_dir: str, patient_id: str, examples: List[Dict]) -> None:
    os.makedirs(shard_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(shard_dir, f"{patient_id}.npz"),
        target_masks=np.stack([e["target_mask"] for e in examples]),
        oar_masks=np.stack([e["oar_mask"] for e in examples]),
        dvh_labels=np.stack([e["dvh_label"] for e in examples]),
        voxel_counts=np.array([e["voxel_count"] for e in examples], dtype=np.int32),
        slice_indices=np.array([e["slice_index"] for e in examples], dtype=np.int32),
    )
