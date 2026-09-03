# DVHnet: 2D Slice-Level Cumulative-DVH Prediction

A reference implementation of the pipeline you specified: DICOM-CT/RTSTRUCT/RTDOSE
preprocessing → 2-channel (target, OAR) slice masks → CNN regression of a
256-bin cumulative DVH per slice → volume-weighted 2D→3D aggregation →
clinical-metric evaluation (D2%, D50%, Dmean, MAD).

## Files

| File | Purpose |
|---|---|
| `preprocessing.py` | DICOM loading, dose→CT resampling, mask/channel generation, slice-level DVH ground truth, patient-level train/val/test split |
| `dataset.py` | PyTorch `Dataset`/collate function, `.npz` shard I/O keyed by patient |
| `model.py` | `DVHNet`: staged ResNet-style conv blocks → GAP → FC(1024→512→256) → sigmoid; `enforce_monotonic()` post-processing fallback |
| `losses.py` | `DVHLoss` = fidelity (Smooth L1) + λ₁·clinical-point penalty (D2%/D50%) + λ₂·monotonicity hinge |
| `aggregate.py` | Voxel-count-weighted 2D→3D DVH reconstruction |
| `metrics.py` | Dose-metric inversion (D2%, D50%, Dmean via AUC), MAD, cohort summary, acceptance-benchmark check |
| `train.py` | End-to-end training/evaluation CLI |

> All 7 files are present in this package and syntax-checked clean.

## Setup

```bash
pip install torch numpy scipy pydicom rt-utils --break-system-packages
```

`rt-utils` is used for RTSTRUCT contour rasterization in `load_rtstruct_masks`;
swap in your own rasterizer if you already have one (e.g. `dicom-contour`,
or a custom polygon-fill routine per `pydicom` `ContourSequence`).

## 1–2. Preprocessing & label construction

```python
from preprocessing import PreprocessConfig, load_ct_series, load_rtdose, \
    load_rtstruct_masks, align_dose_to_ct, resample_mask_to_target, \
    PatientStudy, build_slice_dataset
from dataset import save_patient_shard

cfg = PreprocessConfig(target_matrix=(256, 256), target_slice_thickness_mm=3.0,
                        dose_max_gy=80.0, num_bins=256)

ct_vol, ct_spacing, ct_origin, ct_slices = load_ct_series("path/to/CT")
dose, dose_spacing, dose_origin = load_rtdose("path/to/RTDOSE.dcm")
raw_masks = load_rtstruct_masks("path/to/RTSTRUCT.dcm",
                                 [s.filename for s in ct_slices],
                                 structure_names=cfg.target_names + cfg.oar_names)

dose_resampled = align_dose_to_ct(dose, dose_spacing, dose_origin,
                                   ct_vol.shape, ct_spacing, ct_origin, cfg)
resampled_masks = {name: resample_mask_to_target(m, ct_spacing, ct_vol.shape, cfg)
                    for name, m in raw_masks.items()}

target_union = sum(resampled_masks[n] for n in cfg.target_names if n in resampled_masks)
study = PatientStudy(patient_id="P001", ct_volume=ct_vol, dose_volume=dose_resampled,
                      target_mask=(target_union > 0).astype("uint8"),
                      oar_masks={k: v for k, v in resampled_masks.items() if k in cfg.oar_names},
                      spacing=(cfg.target_slice_thickness_mm,) + tuple(ct_spacing[1:]))

for oar_name in cfg.oar_names:
    examples = build_slice_dataset(study, cfg, oar_name)
    if examples:
        save_patient_shard(f"./shards/{oar_name}", "P001", examples)
```

Run this per patient, per OAR, across your cohort. This is the natural place
to parallelize (multiprocessing over patients) since each patient's
resampling/rasterization is independent.

## 3–4. Model & loss

`DVHNet` matches your spec: strided-conv downsampling (H/2 → H/16, plus the
stem's extra /4 for a typical 512×512-ish input), GAP bottleneck, FC
1024→512→256, sigmoid output. `DVHLoss` implements all three terms; tune
`lambda_clinical` / `lambda_mono` and `clinical_percents` as needed —
defaults land on D2%/D50% since those were the points you named.

## 5. Training

```bash
python train.py --shard_dir ./shards/Parotid_L --oar Parotid_L \
    --epochs 100 --batch_size 32 --lr 1e-3 \
    --dose_max_gy 80 --lambda_clinical 0.5 --lambda_mono 0.1 \
    --out_dir ./runs/parotid_l
```

Train one model per OAR (matching the per-organ 2-channel input design), or
adapt `DVHNet`'s stem to accept a one-hot OAR-identity channel if you'd
rather train a single multi-organ model — that's a straightforward extension
but changes the input contract, so it's left out of this reference version.

The split (`split_patients`) is patient-level by construction — `train.py`
lists shard filenames (one per patient) and splits *those*, then loads
per-patient slices only from within each split, so no slice from a training
patient ever leaks into val/test.

## 6. Evaluation

`train.py` runs a final patient-level test evaluation automatically:
predicted and ground-truth *slice* DVHs are aggregated to *patient* DVHs via
`aggregate.aggregate_from_batch_outputs` (voxel-count weighted), then scored
with `metrics.evaluate_patient` (D2%, D50%, Dmean errors + MAD) and rolled up
with `metrics.summarize_results` / `check_acceptance_benchmarks`
(≤1.0 Gy mean-dose error, D2% within 2.0–2.5 Gy, matching the targets you
listed).

## Design notes / things you'll likely need to adapt

- **RTSTRUCT rasterization** is delegated to `rt-utils`; if your contours
  have non-standard orientation or holes, verify the rasterized masks
  visually before trusting the pipeline end-to-end.
- **Resampling order**: dose is resampled with linear interpolation (`order=1`),
  masks with nearest-neighbor (`order=0`) to keep hard boundaries — mixing
  these up will silently soften OAR edges and bias the DVH labels.
- **Dose/CT origin alignment**: `align_dose_to_ct` resamples the RTDOSE grid
  onto the CT's native grid using each volume's own `ImagePositionPatient`
  origin, not just the spacing ratio — RTDOSE grids are usually smaller than
  and offset from the CT field of view, so pass the real `ct_origin` from
  `load_ct_series` (not `None`). Voxels outside the calculated dose volume
  are filled with 0 Gy, so make sure your dose grid covers every OAR you care
  about.
- **Monotonicity**: the loss uses a squared hinge (smoother gradients than
  the raw `max(0, ·)` you specified) — swap `losses.DVHLoss` back to
  `torch.clamp(diffs, min=0.0).mean()` if you want the exact formulation
  from your spec.
- **D2%/D50% in the loss** are approximated by fixed bin indices (a cheap,
  differentiable proxy) rather than by inverting the curve at every training
  step; the *real* dose-metric inversion (`metrics.dose_at_volume`) is only
  used at evaluation time, where differentiability doesn't matter.
- This code hasn't been run against real DICOM data or trained — the
  DVH-construction, metric-inversion, and aggregation math was verified
  numerically and every module was syntax-checked, but you should sanity-check
  outputs against a known plan before trusting it clinically.
