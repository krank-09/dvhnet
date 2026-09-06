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

PTV/prescription-dose resolution: prefers the challenge's own
meta_files/PTV_DICT.json (exact prescribed dose + the correct PTV
optimization-structure key per patient, e.g. "PTVHighOPT") and
meta_files/Pat_Obj_DICT.json (validates the chosen OAR was actually used in
that patient's plan), following the same convention as the official
baseline's data_loader.py (see gdp_hmm_reference/). Falls back to a fixed
PTV-key guess-chain when metadata isn't available for a given patient
(unknown prescription dose in that case) -- this is a graceful degradation,
not a requirement, since not every patient we might process is guaranteed
to be in these specific meta files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import PatientStudy, PreprocessConfig, build_slice_dataset, resample_volume  # noqa: E402
from dataset import save_patient_shard  # noqa: E402

# Fallback preference order for resolving "the PTV/target" channel when
# metadata isn't available -- GDP-HMM has no single canonical "PTV" key.
PTV_KEY_CANDIDATES = ["PTV_Total", "PTV70", "PTVHighOPT", "PTV", "CTV"]

# Keys that are never structure masks, even though many of them are ndarrays.
NON_STRUCTURE_KEYS = {"img", "dose", "dose_scale", "spacing", "origin", "direction",
                       "size", "isVMAT", "isocenter", "angle_list", "angle_plate",
                       "beam_plate", "all_mask"}

# Repo-relative default: the cloned challenge repo is a sibling of dvhnet/.
DEFAULT_META_DIR = Path(__file__).resolve().parent.parent.parent / "pipeline" / "meta_files"


def load_gdp_hmm_npz(path: str) -> Dict:
    npz = np.load(path, allow_pickle=True)
    return dict(npz)["arr_0"].item()


def structure_keys(patient_dict: Dict) -> List[str]:
    return sorted(k for k, v in patient_dict.items()
                  if k not in NON_STRUCTURE_KEYS and isinstance(v, np.ndarray))


def meta_patient_id_from_npz_path(npz_path: str) -> str:
    """GDP-HMM npz filenames are '{PatientID}+{PlanID}+{hash}.npz'; the meta
    JSON/CSV files are keyed by PatientID alone. Matches the official
    data_loader.py's own convention (`ID.split('+')[0]`) exactly, rather
    than relying on this adapter's own (possibly plan-suffixed) output
    patient_id."""
    return Path(npz_path).stem.split("+")[0]


def load_meta_dicts(meta_dir: Optional[str]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Returns (PTV_DICT, Pat_Obj_DICT), or (None, None) if meta_dir isn't
    usable -- metadata-based resolution is an enhancement, not a hard
    requirement."""
    if meta_dir is None:
        return None, None
    meta_dir = Path(meta_dir)
    ptv_path = meta_dir / "PTV_DICT.json"
    obj_path = meta_dir / "Pat_Obj_DICT.json"
    if not (ptv_path.exists() and obj_path.exists()):
        return None, None
    return json.loads(ptv_path.read_text()), json.loads(obj_path.read_text())


def resolve_target_mask(patient_dict: Dict,
                         candidates: List[str] = PTV_KEY_CANDIDATES) -> Tuple[np.ndarray, str]:
    """Fallback PTV resolution: first candidate key present with nonzero
    voxels. No known prescription dose in this path."""
    for key in candidates:
        mask = patient_dict.get(key)
        if isinstance(mask, np.ndarray) and mask.sum() > 0:
            return (mask > 0).astype(np.uint8), key
    raise KeyError(
        f"none of the candidate PTV/target keys {candidates} were found with nonzero "
        f"voxels in this file; available structure keys: {structure_keys(patient_dict)}"
    )


def resolve_target_from_metadata(patient_dict: Dict, ptv_dict_entry: Dict) -> Tuple[np.ndarray, str, float]:
    """Real prescription-dose-backed PTV resolution: uses PTV_DICT.json's
    'PTV_High' level (highest-dose PTV, the one that drives plan
    acceptance) -- its 'OPTName' is the actual optimization-structure key,
    'PDose' the real prescribed dose in Gy. Raises if that level or key is
    missing; caller should fall back to resolve_target_mask on failure."""
    if "PTV_High" not in ptv_dict_entry:
        raise KeyError("PTV_DICT entry has no 'PTV_High' level")
    high = ptv_dict_entry["PTV_High"]
    key = high["OPTName"]
    mask = patient_dict.get(key)
    if not (isinstance(mask, np.ndarray) and mask.sum() > 0):
        raise KeyError(f"metadata-resolved PTV key {key!r} not present (or empty) in this npz's structure keys")
    return (mask > 0).astype(np.uint8), key, float(high["PDose"])


def validate_oar_against_metadata(oar_name: str, meta_patient_id: str,
                                   pat_obj_dict: Optional[Dict]) -> Optional[bool]:
    """True/False if we can check, None if this patient has no Pat_Obj_DICT
    entry to check against (metadata unavailable, not a validation failure)."""
    if pat_obj_dict is None or meta_patient_id not in pat_obj_dict:
        return None
    return oar_name in pat_obj_dict[meta_patient_id]


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
                     ptv_key_candidates: List[str] = PTV_KEY_CANDIDATES,
                     meta_dir: Optional[str] = DEFAULT_META_DIR) -> Dict:
    """
    Convert one GDP-HMM patient .npz -> one dvhnet shard (out_shard_dir/{patient_id}.npz).
    Returns a summary dict for reporting (shard path, example count, which
    PTV key was used, whether metadata was used, prescription dose if
    known, OAR-validation result, etc.) -- does not delete the source npz.
    """
    cfg = cfg or PreprocessConfig()
    d = load_gdp_hmm_npz(npz_path)

    if oar_name not in d:
        raise KeyError(f"OAR {oar_name!r} not present in {npz_path!r}; "
                        f"available structure keys: {structure_keys(d)}")

    meta_patient_id = meta_patient_id_from_npz_path(npz_path)
    ptv_dict, pat_obj_dict = load_meta_dicts(meta_dir)

    prescription_dose_gy: Optional[float] = None
    used_metadata = False
    if ptv_dict is not None and meta_patient_id in ptv_dict:
        try:
            target_mask, target_key_used, prescription_dose_gy = resolve_target_from_metadata(
                d, ptv_dict[meta_patient_id])
            used_metadata = True
        except KeyError as e:
            print(f"[gdp_hmm_adapter] metadata-based PTV resolution failed for "
                  f"{meta_patient_id!r} ({e}); falling back to guess-chain.")
            target_mask, target_key_used = resolve_target_mask(d, ptv_key_candidates)
    else:
        target_mask, target_key_used = resolve_target_mask(d, ptv_key_candidates)

    oar_validated = validate_oar_against_metadata(oar_name, meta_patient_id, pat_obj_dict)
    if oar_validated is False:
        print(f"[gdp_hmm_adapter] WARNING: OAR {oar_name!r} is not in "
              f"Pat_Obj_DICT[{meta_patient_id!r}]'s validated structure list -- "
              f"it may still be present as a mask key, but wasn't part of the "
              f"actual optimization objective for this patient's plan.")

    ct_volume = d["img"].astype(np.float32)
    dose_volume = dose_in_gy(d)
    oar_mask = (d[oar_name] > 0).astype(np.uint8)
    spacing = gdp_hmm_spacing_to_dvhnet(tuple(d["spacing"]))

    # img/dose are on the same native grid (GDP-HMM's own DICOM2NPZ.py already
    # resampled dose onto the CT grid), so align_dose_to_ct's step 1 (dose->CT
    # alignment) is correctly skipped above. But that grid's in-plane size
    # varies per patient/institution -- without also doing align_dose_to_ct's
    # step 2 (CT-native spacing -> cfg.target_matrix), slices from different
    # patients have different (H, W) and can't be batched together (only
    # stayed hidden because the one prior smoke test used two plans of the
    # SAME patient, which share a native grid). Mirrors
    # resample_mask_to_target's target_spacing/target_shape math exactly.
    ct_shape = ct_volume.shape
    target_spacing = (cfg.target_slice_thickness_mm,
                       spacing[1] * ct_shape[1] / cfg.target_matrix[0],
                       spacing[2] * ct_shape[2] / cfg.target_matrix[1])
    target_shape = (round(ct_shape[0] * spacing[0] / cfg.target_slice_thickness_mm),
                     cfg.target_matrix[0], cfg.target_matrix[1])
    ct_volume = resample_volume(ct_volume, spacing, target_spacing, dst_shape=target_shape, order=1)
    dose_volume = resample_volume(dose_volume, spacing, target_spacing, dst_shape=target_shape, order=1)
    target_mask = (resample_volume(target_mask.astype(np.float32), spacing, target_spacing,
                                    dst_shape=target_shape, order=0) > 0.5).astype(np.uint8)
    oar_mask = (resample_volume(oar_mask.astype(np.float32), spacing, target_spacing,
                                 dst_shape=target_shape, order=0) > 0.5).astype(np.uint8)
    spacing = target_spacing

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
        "meta_patient_id": meta_patient_id,
        "npz_source": npz_path,
        "shard_path": str(shard_path),
        "shard_size_bytes": shard_path.stat().st_size,
        "n_slice_examples": len(examples),
        "oar_name": oar_name,
        "oar_voxels_total": int(oar_mask.sum()),
        "oar_validated_against_metadata": oar_validated,
        "target_key_used": target_key_used,
        "used_metadata_for_ptv": used_metadata,
        "prescription_dose_gy": prescription_dose_gy,
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
    parser.add_argument("--meta_dir", default=str(DEFAULT_META_DIR),
                         help="dir containing PTV_DICT.json / Pat_Obj_DICT.json; "
                              "pass empty string to disable metadata lookup")
    args = parser.parse_args()

    cfg = PreprocessConfig(dose_max_gy=args.dose_max_gy, num_bins=args.num_bins)
    meta_dir = args.meta_dir or None
    summary = convert_patient(args.npz_path, args.patient_id, args.oar, args.out_shard_dir,
                               cfg, meta_dir=meta_dir)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
