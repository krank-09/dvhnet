"""
metrics.py
==========
Evaluation utilities: inverting a predicted cumulative DVH to clinically
meaningful dose metrics (D2%, D50%, Dmean), computing Mean Absolute DVH
Deviation (MAD) against ground truth, and checking results against the
paper's acceptance benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _bin_edges(num_bins: int, dose_max_gy: float) -> np.ndarray:
    return np.linspace(0.0, dose_max_gy, num_bins)


def dose_at_volume(dvh: np.ndarray, volume_percent: float, dose_max_gy: float) -> float:
    """
    Invert a cumulative DVH curve y[k] = P(dose >= D_k) to find D_x%: the dose
    at which the curve crosses volume fraction x/100.

    Since y is (approximately) monotonic non-increasing in dose, and dose is
    monotonic increasing in bin index, we linearly interpolate y(dose) to
    solve for the dose where y == volume_percent / 100.
    """
    num_bins = dvh.shape[-1]
    doses = _bin_edges(num_bins, dose_max_gy)
    target = volume_percent / 100.0

    # Ensure strict monotonic non-increasing curve for a well-posed inversion
    y = np.minimum.accumulate(dvh)

    if target >= y[0]:
        return float(doses[0])
    if target <= y[-1]:
        return float(doses[-1])

    # np.interp requires increasing x; y is decreasing, so flip both arrays
    y_rev = y[::-1]
    doses_rev = doses[::-1]
    return float(np.interp(target, y_rev, doses_rev))


def d2(dvh: np.ndarray, dose_max_gy: float) -> float:
    """Near-maximum dose (hot-spot proxy): dose received by 2% of the OAR volume."""
    return dose_at_volume(dvh, 2.0, dose_max_gy)


def d50(dvh: np.ndarray, dose_max_gy: float) -> float:
    """Median dose: dose received by 50% of the OAR volume."""
    return dose_at_volume(dvh, 50.0, dose_max_gy)


def d_mean(dvh: np.ndarray, dose_max_gy: float) -> float:
    """
    Mean dose = area under the cumulative DVH curve (integral of y(dose) ddose),
    since y(dose) is the survival function of the per-voxel dose distribution
    and E[dose] = integral_0^max P(dose >= d) dd for a nonnegative dose.
    """
    num_bins = dvh.shape[-1]
    doses = _bin_edges(num_bins, dose_max_gy)
    y = np.minimum.accumulate(dvh)
    return float(_trapz(y, doses))


def mean_absolute_dvh_deviation(pred: np.ndarray, target: np.ndarray) -> float:
    """MAD = (1/num_bins) * sum |pred_k - target_k|."""
    return float(np.mean(np.abs(pred - target)))


@dataclass
class PatientMetricResult:
    patient_id: str
    oar_name: str
    d2_pred: float
    d2_true: float
    d50_pred: float
    d50_true: float
    dmean_pred: float
    dmean_true: float
    mad: float

    @property
    def d2_error(self) -> float:
        return abs(self.d2_pred - self.d2_true)

    @property
    def d50_error(self) -> float:
        return abs(self.d50_pred - self.d50_true)

    @property
    def dmean_error(self) -> float:
        return abs(self.dmean_pred - self.dmean_true)


def evaluate_patient(patient_id: str, oar_name: str, pred_dvh: np.ndarray,
                      true_dvh: np.ndarray, dose_max_gy: float) -> PatientMetricResult:
    return PatientMetricResult(
        patient_id=patient_id,
        oar_name=oar_name,
        d2_pred=d2(pred_dvh, dose_max_gy), d2_true=d2(true_dvh, dose_max_gy),
        d50_pred=d50(pred_dvh, dose_max_gy), d50_true=d50(true_dvh, dose_max_gy),
        dmean_pred=d_mean(pred_dvh, dose_max_gy), dmean_true=d_mean(true_dvh, dose_max_gy),
        mad=mean_absolute_dvh_deviation(pred_dvh, true_dvh),
    )


def summarize_results(results: list) -> Dict[str, float]:
    """Aggregate per-patient PatientMetricResult objects into cohort-level summary stats."""
    if not results:
        return {}
    d2_err = np.array([r.d2_error for r in results])
    d50_err = np.array([r.d50_error for r in results])
    dmean_err = np.array([r.dmean_error for r in results])
    mad = np.array([r.mad for r in results])

    return {
        "n_patients": len(results),
        "D2_MAE_gy": float(d2_err.mean()), "D2_std_gy": float(d2_err.std()),
        "D50_MAE_gy": float(d50_err.mean()), "D50_std_gy": float(d50_err.std()),
        "Dmean_MAE_gy": float(dmean_err.mean()), "Dmean_std_gy": float(dmean_err.std()),
        "MAD_mean": float(mad.mean()), "MAD_std": float(mad.std()),
    }


def check_acceptance_benchmarks(summary: Dict[str, float],
                                 dmean_threshold_gy: float = 1.0,
                                 d2_threshold_gy: float = 2.5) -> Dict[str, bool]:
    """
    Compare cohort summary stats against the paper's acceptance criteria:
    mean-dose error <= 1.0 Gy, D2% error within 2.0-2.5 Gy.
    """
    return {
        "dmean_within_threshold": summary.get("Dmean_MAE_gy", float("inf")) <= dmean_threshold_gy,
        "d2_within_threshold": summary.get("D2_MAE_gy", float("inf")) <= d2_threshold_gy,
    }
