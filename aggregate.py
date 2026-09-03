"""
aggregate.py
============
Reconstructs the patient-level (3D) DVH from per-slice predictions via a
voxel-count-weighted average, since a parotid's superior/inferior pole slices
contribute far less organ volume than its equatorial slices and should not be
weighted equally with them.

    D_patient = sum_s(V_s * d_s) / sum_s(V_s)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def aggregate_patient_dvh(slice_predictions: List[np.ndarray],
                           slice_voxel_counts: List[int]) -> np.ndarray:
    """
    Args:
        slice_predictions: list of [num_bins] arrays, one predicted cumulative
                            DVH per axial slice for a single patient/OAR.
        slice_voxel_counts: matching list of OAR voxel counts per slice.

    Returns:
        [num_bins] patient-level cumulative DVH.
    """
    if len(slice_predictions) == 0:
        raise ValueError("No slices provided for aggregation.")
    weights = np.asarray(slice_voxel_counts, dtype=np.float64)
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("Total OAR voxel count across slices must be positive.")
    stacked = np.stack(slice_predictions, axis=0)  # [S, num_bins]
    weighted = (weights[:, None] * stacked).sum(axis=0) / total_weight
    return weighted.astype(np.float32)


def aggregate_from_batch_outputs(patient_ids: List[str], slice_indices: List[int],
                                  preds: np.ndarray, voxel_counts: np.ndarray
                                  ) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper for evaluation: given flat per-slice model outputs
    (as accumulated over an entire val/test loader) plus their patient IDs
    and voxel counts, group by patient and aggregate to a single 3D DVH per
    patient. Handles multiple slices arriving in arbitrary order.
    """
    grouped_preds: Dict[str, List[np.ndarray]] = defaultdict(list)
    grouped_weights: Dict[str, List[int]] = defaultdict(list)
    for pid, pred, w in zip(patient_ids, preds, voxel_counts):
        grouped_preds[pid].append(pred)
        grouped_weights[pid].append(int(w))
    patient_dvhs = {}
    for pid in grouped_preds:
        patient_dvhs[pid] = aggregate_patient_dvh(grouped_preds[pid], grouped_weights[pid])
    return patient_dvhs
