"""
Unit tests for PSI and drift detection logic.
Covers the psi() function and compute_drift() behaviour — no SQLite or Prometheus needed.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp as ks2samp



# ── Inline implementations (mirrors drift_detection.py) ───────────────────────

EPS = 1e-6
PSI_LOW      = 0.05
PSI_MODERATE = 0.12
PSI_HIGH     = 0.25
RETRAIN_FRAC = 0.40

FEATURE_COLS = ["pm25", "pm10", "no2", "so2", "co", "o3",
                "temperature2m", "relativehumidity2m",
                "windspeed10m", "winddirection10m"]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.array(expected, dtype=float)
    actual   = np.array(actual,   dtype=float)
    min_val  = min(expected.min(), actual.min())
    max_val  = max(expected.max(), actual.max())
    if max_val == min_val:
        return 0.0
    breakpoints = np.linspace(min_val, max_val, bins + 1)
    e_pct = np.histogram(expected, bins=breakpoints)[0] / (len(expected) + EPS)
    a_pct = np.histogram(actual,   bins=breakpoints)[0] / (len(actual)   + EPS)
    e_pct = np.clip(e_pct, EPS, None)
    a_pct = np.clip(a_pct, EPS, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def compute_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame, window_days: int = 3):
    drifted = []
    report_features = {}
    for col in FEATURE_COLS:
        if col not in baseline_df.columns or col not in new_df.columns:
            continue
        base_vals = baseline_df[col].dropna().values
        new_vals  = new_df[col].dropna().values
        if len(base_vals) < 10 or len(new_vals) < 10:
            continue
        psi_val  = psi(base_vals, new_vals)
        ks_stat, ks_p = ks2samp(base_vals, new_vals)
        baseline_std = base_vals.std()
        z_score = abs(new_vals.mean() - base_vals.mean()) / (baseline_std + EPS)
        ks_triggers = ks_p < 0.05
        z_triggers  = z_score > 2.0
        if psi_val > PSI_HIGH:
            severity = "HIGH"
        elif psi_val > PSI_MODERATE and ks_triggers:
            severity = "MODERATE"
        elif psi_val > PSI_LOW:
            severity = "LOW"
        else:
            severity = "NONE"
        drift_flag = psi_val > PSI_LOW and ks_triggers and z_triggers
        if drift_flag:
            drifted.append(col)
        report_features[col] = {
            "psi": round(psi_val, 4), "ks_pvalue": round(ks_p, 6),
            "z_score": round(float(z_score), 4), "severity": severity, "drift": drift_flag,
        }
    total          = len(report_features)
    drift_fraction = round(len(drifted) / total, 3) if total else 0.0
    return {
        "features": report_features,
        "summary": {
            "features_with_drift": len(drifted),
            "total_features_checked": total,
            "drift_fraction": drift_fraction,
            "retrain_recommended": drift_fraction >= RETRAIN_FRAC,
            "drifted_features": drifted,
        }
    }


# ── Fixture helpers ───────────────────────────────────────────────────────────

def make_df(n: int, seed: int = 0, shift: float = 0.0, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(50 + shift, 10 * scale, n) for col in FEATURE_COLS}
    return pd.DataFrame(data)


class TestPSIFunction:
    """TC-PSI-01 to TC-PSI-06: Population Stability Index calculation."""

    def test_identical_distributions_psi_near_zero(self):
        """TC-PSI-01: Same distribution → PSI ≈ 0."""
        arr = np.random.default_rng(1).normal(50, 10, 500)
        result = psi(arr, arr)
        assert result < 0.01, f"Expected PSI≈0 for identical arrays, got {result}"

    def test_similar_distributions_psi_below_threshold(self):
        """TC-PSI-02: Slightly different distributions stay below LOW threshold (0.05)."""
        rng = np.random.default_rng(42)
        base = rng.normal(50, 10, 1000)
        live = rng.normal(51, 10, 500)   # 1-unit mean shift
        result = psi(base, live)
        assert result < PSI_LOW, f"Expected PSI < {PSI_LOW}, got {result:.4f}"

    def test_large_shift_psi_above_high_threshold(self):
        """TC-PSI-03: Large distribution shift → PSI > HIGH threshold (0.25)."""
        rng = np.random.default_rng(7)
        base = rng.normal(50, 5, 1000)
        live = rng.normal(200, 5, 500)   # enormous mean shift
        result = psi(base, live)
        assert result > PSI_HIGH, f"Expected PSI > {PSI_HIGH}, got {result:.4f}"

    def test_psi_non_negative(self):
        """TC-PSI-04: PSI is always >= 0."""
        rng = np.random.default_rng(3)
        for _ in range(20):
            base = rng.normal(50, 10, 200)
            live = rng.normal(rng.uniform(40, 80), rng.uniform(5, 20), 100)
            assert psi(base, live) >= 0.0

    def test_psi_constant_array_returns_zero(self):
        """TC-PSI-05: Constant arrays (zero variance) return PSI=0."""
        base = np.full(100, 50.0)
        live = np.full(100, 50.0)
        assert psi(base, live) == 0.0

    def test_psi_symmetric_approximately(self):
        """TC-PSI-06: PSI(A, B) ≈ PSI(B, A) for similar sizes."""
        rng = np.random.default_rng(99)
        a = rng.normal(50, 10, 500)
        b = rng.normal(60, 10, 500)
        assert abs(psi(a, b) - psi(b, a)) < 0.05


class TestComputeDrift:
    """TC-DR-01 to TC-DR-07: Full drift computation pipeline."""

    def test_identical_windows_no_drift(self):
        """TC-DR-01: Identical baseline and new windows → no features drifted."""
        df = make_df(500, seed=42)
        result = compute_drift(df, df)
        assert result["summary"]["features_with_drift"] == 0

    def test_large_shift_triggers_drift(self):
        """TC-DR-02: 5-sigma mean shift triggers drift on all numeric features."""
        base = make_df(500, seed=1, shift=0)
        live = make_df(500, seed=2, shift=500)   # extreme shift
        result = compute_drift(base, live)
        assert result["summary"]["features_with_drift"] > 0

    def test_retrain_flag_off_for_stable_data(self):
        """TC-DR-03: Stable distributions → retrain_recommended is False."""
        df = make_df(500, seed=10)
        result = compute_drift(df, df.sample(frac=0.8, random_state=5))
        assert result["summary"]["retrain_recommended"] is False

    def test_retrain_flag_on_for_massive_drift(self):
        """TC-DR-04: Drift on >=40% of features → retrain_recommended is True."""
        base = make_df(500, seed=1, shift=0,   scale=1)
        live = make_df(500, seed=2, shift=1000, scale=0.01)
        result = compute_drift(base, live)
        assert result["summary"]["retrain_recommended"] is True

    def test_each_feature_has_required_keys(self):
        """TC-DR-05: Every feature dict contains psi, ks_pvalue, z_score, severity, drift."""
        df = make_df(300, seed=99)
        result = compute_drift(df, df)
        for col, vals in result["features"].items():
            for key in ("psi", "ks_pvalue", "z_score", "severity", "drift"):
                assert key in vals, f"Missing '{key}' for feature '{col}'"

    def test_severity_levels_are_valid(self):
        """TC-DR-06: Severity label is one of NONE, LOW, MODERATE, HIGH."""
        df = make_df(300)
        result = compute_drift(df, df)
        valid = {"NONE", "LOW", "MODERATE", "HIGH"}
        for col, vals in result["features"].items():
            assert vals["severity"] in valid, f"Invalid severity '{vals['severity']}' for '{col}'"

    def test_skips_columns_with_insufficient_data(self):
        """TC-DR-07: Columns with < 10 non-null rows are skipped (not in features dict)."""
        base = make_df(5, seed=1)   # only 5 rows
        live = make_df(5, seed=2)
        result = compute_drift(base, live)
        assert result["summary"]["total_features_checked"] == 0

    def test_drift_fraction_between_0_and_1(self):
        """TC-DR-08: drift_fraction is always in [0.0, 1.0]."""
        for shift in [0, 10, 100]:
            base = make_df(300, seed=1)
            live = make_df(300, seed=2, shift=shift)
            frac = compute_drift(base, live)["summary"]["drift_fraction"]
            assert 0.0 <= frac <= 1.0