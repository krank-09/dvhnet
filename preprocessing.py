"""
preprocessing.py
=================
Data preparation for DVHnet: loads paired DICOM-CT / RTSTRUCT / RTDOSE studies,
resamples dose onto the CT grid, builds 2-channel (PTV, OAR) masks per axial
slice, and constructs the 256-bin cumulative-DVH ground-truth vector for each
slice.

Dependencies: pydicom, numpy, scipy, rt-utils (or your own RTSTRUCT parser).
Install with:
    pip install pydicom numpy scipy rt-utils --break-system-packages
"""

from __future__ import annotations

import os
import glob
import json
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pydicom
except ImportError:  # pragma: no cover - allows syntax checking without the dep
    pydicom = None

try:
    from scipy.ndimage import zoom, map_coordinates
except ImportError:  # pragma: no cover
    zoom = None
    map_coordinates = None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class PreprocessConfig:
    target_matrix: Tuple[int, int] = (256, 256)     # in-plane resample size
    target_slice_thickness_mm: float = 3.0           # uniform slice spacing
    dose_max_gy: float = 80.0                         # site-specific ceiling (Nasopharynx: 75-80 Gy)
    num_bins: int = 256
    oar_names: List[str] = field(default_factory=lambda: [
        "Brainstem", "SpinalCord", "ParotidL", "ParotidR", "Mandible",
        "Larynx", "Esophagus", "OralCavity", "Lens_L", "Lens_R",
        "OpticNerve_L", "OpticNerve_R", "Chiasm", "TemporalLobe_L",
        "TemporalLobe_R", "InnerEar_L",
    ])
    target_names: List[str] = field(default_factory=lambda: ["PTV", "CTV"])


# --------------------------------------------------------------------------- #
# Study container
# --------------------------------------------------------------------------- #

@dataclass
class PatientStudy:
    patient_id: str
    ct_volume: np.ndarray            # [Z, H, W] in HU (not strictly needed for DVHnet input, kept for QA)
    dose_volume: np.ndarray          # [Z, H, W] in Gy, resampled to CT grid
    target_mask: np.ndarray          # [Z, H, W] binary, union of PTV/CTV
    oar_masks: Dict[str, np.ndarray] # name -> [Z, H, W] binary
    spacing: Tuple[float, float, float]  # (dz, dy, dx) mm


# --------------------------------------------------------------------------- #
# DICOM loading
# --------------------------------------------------------------------------- #

def load_ct_series(ct_dir: str) -> Tuple[np.ndarray, Tuple[float, float, float],
                                          Tuple[float, float, float], List[pydicom.Dataset]]:
    """Load an axial CT series into a [Z, H, W] HU volume, sorted by slice position."""
    files = glob.glob(os.path.join(ct_dir, "*.dcm"))
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))

    hu_volume = np.stack([
        (s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1.0))
         + float(getattr(s, "RescaleIntercept", 0.0)))
        for s in slices
    ], axis=0)

    # DICOM PixelSpacing = [row spacing, column spacing]; the array is
    # [Z, H, W] = [Z, rows, cols], so row_spacing maps to H and col_spacing to W.
    row_spacing, col_spacing = [float(v) for v in slices[0].PixelSpacing]
    z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
    dz = abs(np.median(np.diff(z_positions))) if len(z_positions) > 1 else float(slices[0].SliceThickness)
    origin = tuple(float(v) for v in slices[0].ImagePositionPatient)

    return hu_volume, (dz, row_spacing, col_spacing), origin, slices


def load_rtdose(rtdose_path: str) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """Load an RTDOSE file, returning the dose grid in Gy, its voxel spacing, and its origin."""
    ds = pydicom.dcmread(rtdose_path)
    scaling = float(ds.DoseGridScaling)
    dose = ds.pixel_array.astype(np.float32) * scaling  # [Z, H, W] in Gy

    # DICOM PixelSpacing = [row spacing, column spacing]; see load_ct_series.
    row_spacing, col_spacing = [float(v) for v in ds.PixelSpacing]
    dz = float(ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0]) if len(ds.GridFrameOffsetVector) > 1 \
        else float(ds.SliceThickness)
    origin = tuple(float(v) for v in ds.ImagePositionPatient)

    return dose, (dz, row_spacing, col_spacing), origin


def load_rtstruct_masks(rtstruct_path: str, ct_series_files: List[str],
                         structure_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Rasterize requested ROI contours from an RTSTRUCT into binary masks aligned
    to the CT grid. Using rt-utils here since implementing contour-to-mask
    rasterization from scratch is error-prone; swap in your own if needed.
    """
    from rt_utils import RTStructBuilder  # local import: optional heavy dependency

    ct_dir = os.path.dirname(ct_series_files[0])
    rtstruct = RTStructBuilder.create_from(dicom_series_path=ct_dir, rt_struct_path=rtstruct_path)

    available = set(rtstruct.get_roi_names())
    masks = {}
    for name in structure_names:
        if name not in available:
            continue
        # rt-utils returns [H, W, Z]; reorder to [Z, H, W] for consistency
        mask = rtstruct.get_roi_mask_by_name(name)
        masks[name] = np.transpose(mask, (2, 0, 1)).astype(np.uint8)
    return masks


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #

def resample_volume(volume: np.ndarray,
                     src_spacing: Tuple[float, float, float],
                     dst_spacing: Tuple[float, float, float],
                     dst_shape: Optional[Tuple[int, int, int]] = None,
                     order: int = 1) -> np.ndarray:
    """
    Resample a [Z, H, W] volume from src_spacing to dst_spacing using spline
    interpolation. order=1 (linear) for continuous dose; order=0 (nearest)
    should be used for binary masks to preserve hard edges.
    """
    zoom_factors = [s / d for s, d in zip(src_spacing, dst_spacing)]
    resampled = zoom(volume, zoom_factors, order=order)

    if dst_shape is not None:
        resampled = _center_crop_or_pad(resampled, dst_shape)
    return resampled


def _center_crop_or_pad(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(target_shape, dtype=vol.dtype)
    src_shape = vol.shape
    slices_src, slices_dst = [], []
    for s_src, s_dst in zip(src_shape, target_shape):
        if s_src >= s_dst:
            start = (s_src - s_dst) // 2
            slices_src.append(slice(start, start + s_dst))
            slices_dst.append(slice(0, s_dst))
        else:
            start = (s_dst - s_src) // 2
            slices_src.append(slice(0, s_src))
            slices_dst.append(slice(start, start + s_src))
    out[tuple(slices_dst)] = vol[tuple(slices_src)]
    return out


def _resample_to_grid(src_volume: np.ndarray,
                       src_spacing: Tuple[float, float, float],
                       src_origin: Tuple[float, float, float],
                       dst_shape: Tuple[int, int, int],
                       dst_spacing: Tuple[float, float, float],
                       dst_origin: Tuple[float, float, float],
                       order: int = 1, cval: float = 0.0) -> np.ndarray:
    """
    Resample `src_volume` onto an arbitrary destination grid, respecting the
    physical offset between the two volumes' origins (DICOM
    ImagePositionPatient, given as (x, y, z)). Volumes are [Z, H, W];
    spacings are (dz, dy, dx).

    Unlike a plain spacing-ratio zoom, this maps each destination voxel to
    its physical (x, y, z) position and samples the source volume at the
    corresponding source voxel coordinate. This matters because RTDOSE grids
    are typically smaller than, and offset from, the CT field of view.
    Destination voxels that fall outside the source volume are filled with
    `cval` (0 Gy by default) -- make sure the calculated dose grid actually
    covers every OAR of interest.
    """
    if map_coordinates is None:  # pragma: no cover
        raise ImportError("scipy is required for _resample_to_grid")

    dz_dst, dy_dst, dx_dst = dst_spacing
    dz_src, dy_src, dx_src = src_spacing
    ox_dst, oy_dst, oz_dst = dst_origin
    ox_src, oy_src, oz_src = src_origin

    zz, yy, xx = np.meshgrid(
        np.arange(dst_shape[0]), np.arange(dst_shape[1]), np.arange(dst_shape[2]),
        indexing="ij",
    )
    phys_x = ox_dst + xx * dx_dst
    phys_y = oy_dst + yy * dy_dst
    phys_z = oz_dst + zz * dz_dst

    src_coords = np.stack([
        (phys_z - oz_src) / dz_src,
        (phys_y - oy_src) / dy_src,
        (phys_x - ox_src) / dx_src,
    ], axis=0)

    return map_coordinates(src_volume, src_coords, order=order, cval=cval, mode="constant")


def align_dose_to_ct(dose: np.ndarray, dose_spacing, dose_origin,
                      ct_shape: Tuple[int, int, int], ct_spacing, ct_origin,
                      cfg: PreprocessConfig) -> np.ndarray:
    """
    Resample RTDOSE onto the CT grid, respecting the spatial offset between
    the dose and CT volume origins -- RTDOSE grids are usually smaller than,
    and offset from, the CT field of view, so a plain spacing-ratio zoom
    would silently misalign dose against anatomy unless the two origins
    happened to coincide -- then resample both onto the configured target
    in-plane matrix / slice thickness.
    """
    # Step 1: dose grid -> CT-native grid, in the CT's own physical frame.
    dose_on_ct_native = _resample_to_grid(
        dose, dose_spacing, dose_origin,
        ct_shape, ct_spacing, ct_origin,
        order=1, cval=0.0,
    )

    # Step 2: CT-native spacing -> target uniform spacing / matrix (same
    # coordinate frame as the CT, so no further origin shift is needed).
    target_spacing = (cfg.target_slice_thickness_mm,
                       ct_spacing[1] * ct_shape[1] / cfg.target_matrix[0],
                       ct_spacing[2] * ct_shape[2] / cfg.target_matrix[1])
    target_shape = (round(ct_shape[0] * ct_spacing[0] / cfg.target_slice_thickness_mm),
                    cfg.target_matrix[0], cfg.target_matrix[1])
    dose_resampled = resample_volume(dose_on_ct_native, ct_spacing, target_spacing,
                                      dst_shape=target_shape, order=1)
    return dose_resampled


def resample_mask_to_target(mask: np.ndarray, ct_spacing, ct_shape,
                             cfg: PreprocessConfig) -> np.ndarray:
    target_spacing = (cfg.target_slice_thickness_mm,
                       ct_spacing[1] * ct_shape[1] / cfg.target_matrix[0],
                       ct_spacing[2] * ct_shape[2] / cfg.target_matrix[1])
    target_shape = (round(ct_shape[0] * ct_spacing[0] / cfg.target_slice_thickness_mm),
                    cfg.target_matrix[0], cfg.target_matrix[1])
    resampled = resample_volume(mask.astype(np.float32), ct_spacing, target_spacing,
                                 dst_shape=target_shape, order=0)
    return (resampled > 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Slice-level DVH label construction
# --------------------------------------------------------------------------- #

def compute_slice_cumulative_dvh(dose_slice: np.ndarray, oar_mask_slice: np.ndarray,
                                  cfg: PreprocessConfig) -> Optional[np.ndarray]:
    """
    Build the length-`num_bins` cumulative DVH vector for one axial slice:
        y[k] = fraction of OAR voxels on this slice with dose >= D_k
    Returns None if the OAR is absent on this slice (caller should skip it).
    """
    voxel_doses = dose_slice[oar_mask_slice > 0]
    n = voxel_doses.size
    if n == 0:
        return None

    bin_edges = np.linspace(0.0, cfg.dose_max_gy, cfg.num_bins)
    # Vectorized: for each bin threshold, fraction of voxels >= threshold.
    # voxel_doses[:, None] >= bin_edges[None, :] -> [N, num_bins] boolean
    cumulative = (voxel_doses[:, None] >= bin_edges[None, :]).mean(axis=0)
    return cumulative.astype(np.float32)  # monotonic non-increasing, in [0, 1]


def build_slice_dataset(study: PatientStudy, cfg: PreprocessConfig,
                         oar_name: str) -> List[Dict]:
    """
    For a single patient/OAR pair, produce one training example per axial
    slice on which the OAR is present: 2-channel input mask + DVH label +
    the slice's OAR voxel count (needed later for 2D->3D volume-weighted
    aggregation).
    """
    examples = []
    oar_mask = study.oar_masks.get(oar_name)
    if oar_mask is None:
        return examples

    for z in range(study.dose_volume.shape[0]):
        oar_slice = oar_mask[z]
        if oar_slice.sum() == 0:
            continue  # organ not present on this slice

        target_slice = study.target_mask[z]
        dvh = compute_slice_cumulative_dvh(study.dose_volume[z], oar_slice, cfg)
        if dvh is None:
            continue

        examples.append({
            "patient_id": study.patient_id,
            "oar_name": oar_name,
            "slice_index": z,
            "target_mask": target_slice.astype(np.float32),   # channel 1
            "oar_mask": oar_slice.astype(np.float32),         # channel 2
            "dvh_label": dvh,                                  # [num_bins]
            "voxel_count": int(oar_slice.sum()),               # for weighted aggregation
        })
    return examples


# --------------------------------------------------------------------------- #
# Patient-level split (no slice leakage)
# --------------------------------------------------------------------------- #

def split_patients(patient_ids: List[str], train_frac=0.8, val_frac=0.1,
                    seed: int = 42) -> Dict[str, List[str]]:
    """Patient-level 80/10/10 (or paper's fixed 153/27 style) split. Never split by slice."""
    rng = np.random.RandomState(seed)
    ids = list(patient_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def save_manifest(examples: List[Dict], out_path: str) -> None:
    """Persist slice-level example metadata (not the raw arrays) for reproducibility."""
    meta = [{k: v for k, v in ex.items() if k not in ("target_mask", "oar_mask", "dvh_label")}
            for ex in examples]
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
