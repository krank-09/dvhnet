"""
gdp_hmm_adapter.py
===================
Converts a single GDP-HMM AAPM Challenge patient .npz into a dvhnet training
shard (the exact per-patient .npz format train.py already consumes via
dataset.load_examples_from_shards).

This is NOT a key-rename. GDP-HMM ships full 3D dose volumes + per-structure
3D masks; dvhnet trains on *slice-level cumulative DVH curves* derived from
(dose, OAR mask). This adapter does that derivation for real, by building a
preprocessing.PatientStudy from the GDP-HMM arrays and handing it to
preprocessing.build_slice_dataset -- the same function dvhnet's own
DICOM-based path uses -- rather than reimplementing the DVH math.

Key facts this adapter depends on (verified against a real downloaded file,
NOT the data_visual_understand.ipynb notebook -- that notebook's own
captured cell outputs turned out to be STALE for the same filename; see
conversation. Re-verify against your actual files if the GDP-HMM dataset
version changes):
  - The npz is `np.load(path, allow_pickle=True)['arr_0'].item()`, a pickled
    dict, one key per anatomical structure (uint8 [Z,H,W] binary masks) plus
    fixed keys: img, dose, dose_scale, spacing, origin, direction, size,
    isVMAT, isocenter, angle_list, angle_plate, beam_plate, all_mask (a
    *list* of structure-key names, NOT a merged mask array).
  - `dose` is a raw integer grid; real dose in Gy = dose * dose_scale.
  - `img` and `dose` are ALREADY on the same [Z,H,W] grid (GDP-HMM's own
    DICOM2NPZ.py resamples dose onto the CT grid before packaging) -- unlike
    dvhnet's own DICOM path, this adapter does NOT need
    preprocessing.align_dose_to_ct at all.
  - `spacing` is SimpleITK's (x, y, z) order; preprocessing.PatientStudy
    expects (dz, dy, dx) matching its [Z, H, W] axis convention -- reordered
    below.
  - Structure naming is heterogeneous across patients/cohorts (confirmed:
    lung-cohort files in the same data drop use a completely different
    structure vocabulary from head-and-neck ones). PTV_KEY_CANDIDATES and
    the `oar_name` argument are resolved per-file, not hardcoded globally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import PatientStudy, PreprocessConfig, build_slice_dataset  # noqa: E402
from dataset import save_patient_shard  # noqa: E402

# Preference order for resolving "the PTV/target" channel -- GDP-HMM has no
# single canonical "PTV" key. PTV_Total (union across all dose levels) is
# preferred when present; falls back toward more specific single-level keys.
PTV_KEY_CANDIDATES = ["PTV_Total", "PTV70", "PTVHighOPT", "PTV", "CTV"]

# Keys that are never structure masks, even though many of them are ndarrays.
NON_STRUCTURE_KEYS = {"img", "dose", "dose_scale", "spacing", "origin", "direction",
                       "size", "isVMAT", "isocenter", "angle_list", "angle_plate",
                       "beam_plate", "all_mask"}


def load_gdp_hmm_npz(path: str) -> Dict:
    npz = np.load(path, allow_pickle=True)
    return dict(npz)["arr_0"].item()


def structure_keys(patient_dict: Dict) -> List[str]:
    return sorted(k for k, v in patient_dict.items()
                  if k not in NON_STRUCTURE_KEYS and isinstance(v, np.ndarray))


def resolve_target_mask(patient_dict: Dict,
                         candidates: List[str] = PTV_KEY_CANDIDATES) -> Tuple[np.ndarray, str]:
    for key in candidates:
        mask = patient_dict.get(key)
        if isinstance(mask, np.ndarray) and mask.sum() > 0:
            return (mask > 0).astype(np.uint8), key
    raise KeyError(
        f"none of the candidate PTV/target keys {candidates} were found with nonzero "
        f"voxels in this file; available structure keys: {structure_keys(patient_dict)}"
    )


def dose_in_gy(patient_dict: Dict) -> np.ndarray:
    """Real dose (Gy) = raw dose grid * dose_scale (DoseGridScaling)."""
    raw = patient_dict["dose"].astype(np.float64)
    scale = float(patient_dict["dose_scale"])
    return (raw * scale).astype(np.float32)


def gdp_hmm_spacing_to_dvhnet(spacing_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """GDP-HMM stores SimpleITK (x, y, z) spacing; dvhnet.preprocessing wants
    (dz, dy, dx) matching its [Z, H, W] array axis order."""
    sx, sy, sz = spacing_xyz
    return (sz, sy, sx)


def convert_patient(npz_path: str, patient_id: str, oar_name: str,
                     out_shard_dir: str, cfg: Optional[PreprocessConfig] = None,
                     ptv_key_candidates: List[str] = PTV_KEY_CANDIDATES) -> Dict:
    """
    Convert one GDP-HMM patient .npz -> one dvhnet shard (out_shard_dir/{patient_id}.npz).
    Returns a summary dict for reporting (shard path, example count, which
    PTV key was used, etc.) -- does not delete the source npz.
    """
    cfg = cfg or PreprocessConfig()
    d = load_gdp_hmm_npz(npz_path)

    if oar_name not in d:
        raise KeyError(f"OAR {oar_name!r} not present in {npz_path!r}; "
                        f"available structure keys: {structure_keys(d)}")

    ct_volume = d["img"].astype(np.float32)
    dose_volume = dose_in_gy(d)
    target_mask, target_key_used = resolve_target_mask(d, ptv_key_candidates)
    oar_mask = (d[oar_name] > 0).astype(np.uint8)
    spacing = gdp_hmm_spacing_to_dvhnet(tuple(d["spacing"]))

    study = PatientStudy(
        patient_id=patient_id,
        ct_volume=ct_volume,
        dose_volume=dose_volume,
        target_mask=target_mask,
        oar_masks={oar_name: oar_mask},
        spacing=spacing,
    )

    examples = build_slice_dataset(study, cfg, oar_name)
    if not examples:
        raise ValueError(
            f"build_slice_dataset produced zero slice examples for OAR={oar_name!r} "
            f"patient={patient_id!r} -- mask may be empty, or every slice's OAR "
            f"voxel count is zero after the shapes involved here"
        )

    save_patient_shard(out_shard_dir, patient_id, examples)
    shard_path = Path(out_shard_dir) / f"{patient_id}.npz"

    return {
        "patient_id": patient_id,
        "npz_source": npz_path,
        "shard_path": str(shard_path),
        "shard_size_bytes": shard_path.stat().st_size,
        "n_slice_examples": len(examples),
        "oar_name": oar_name,
        "oar_voxels_total": int(oar_mask.sum()),
        "target_key_used": target_key_used,
        "dose_max_gy_observed": float(dose_volume.max()),
        "example_target_mask_shape": tuple(examples[0]["target_mask"].shape),
        "example_dvh_label_shape": tuple(examples[0]["dvh_label"].shape),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz_path", required=True, help="path to a GDP-HMM patient .npz")
    parser.add_argument("--patient_id", required=True, help="output shard id, e.g. HNC_001_A4Ac")
    parser.add_argument("--oar", required=True, help="structure key to use as the OAR, e.g. Parotids")
    parser.add_argument("--out_shard_dir", required=True, help="e.g. ./shards/Parotids")
    parser.add_argument("--dose_max_gy", type=float, default=80.0)
    parser.add_argument("--num_bins", type=int, default=256)
    args = parser.parse_args()

    cfg = PreprocessConfig(dose_max_gy=args.dose_max_gy, num_bins=args.num_bins)
    summary = convert_patient(args.npz_path, args.patient_id, args.oar, args.out_shard_dir, cfg)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
