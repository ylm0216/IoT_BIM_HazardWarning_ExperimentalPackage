"""Experiment 3: one-factor-at-a-time sensitivity analysis on DS-3.

For each parameter, only that parameter is changed and all other parameters
remain at the default values. The script runs the same DS-3 pipeline
(fusion -> spatial risk quantification -> warning) for 9 parameters x 5
values and exports the detailed runs plus the Table VII range summary.
"""
import os
import sys
import warnings
from dataclasses import dataclass, replace

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_PROC, RESULTS_DIR, set_seed, setup_ieee_style, IEEE_COLORS
from proposed.improved_ds import construct_bpa, improved_ds_fuse
from utils.metrics import classification_metrics, save_csv, save_latex_table, save_figure


set_seed()
setup_ieee_style()

OUT_DIR = os.path.join(RESULTS_DIR, "exp03")
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass(frozen=True)
class DS3Params:
    lambda_gas: float = 0.15
    lambda_vibration: float = 0.40
    lambda_proximity: float = 0.15
    tau: float = 0.60
    beta: float = 0.40
    delta: float = 0.20
    k_confirm: int = 5
    alpha_ema: float = 0.30
    alpha_occlusion: float = 0.20


PARAMETER_GRID = {
    "lambda_gas": {
        "symbol": r"$\lambda_k$ (gas)",
        "values": [0.05, 0.075, 0.10, 0.125, 0.15],
    },
    "lambda_vibration": {
        "symbol": r"$\lambda_k$ (vibration)",
        "values": [0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "lambda_proximity": {
        "symbol": r"$\lambda_k$ (proximity)",
        "values": [0.10, 0.125, 0.15, 0.175, 0.20],
    },
    "tau": {
        "symbol": r"$\tau$",
        "values": [0.55, 0.575, 0.60, 0.625, 0.65],
    },
    "beta": {
        "symbol": r"$\beta$",
        "values": [0.25, 0.2875, 0.325, 0.3625, 0.40],
    },
    "delta": {
        "symbol": r"$\delta$",
        "values": [0.10, 0.125, 0.15, 0.175, 0.20],
    },
    "k_confirm": {
        "symbol": r"$k$",
        "values": [3, 4, 5, 6, 7],
    },
    "alpha_ema": {
        "symbol": r"$\alpha_{\mathrm{ema}}$",
        "values": [0.10, 0.20, 0.30, 0.40, 0.50],
    },
    "alpha_occlusion": {
        "symbol": r"$\alpha_k$ (occlusion)",
        "values": [0.10, 0.15, 0.20, 0.25, 0.30],
    },
}


def load_ds3():
    path = os.path.join(DATA_PROC, "ds3_iot_bim.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DS-3 file not found: {path}. The reproducibility package should include this processed data file."
        )
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def minmax01(x, upper=None):
    x = np.asarray(x, dtype=np.float64)
    if upper is not None:
        return np.clip(x / upper, 0.0, 1.0)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def fuse_sensor_block(x_norm):
    """Fuse one modality over time and return abnormal belief probability."""
    x_norm = np.asarray(x_norm, dtype=np.float64)
    out = np.zeros(x_norm.shape[1], dtype=np.float64)
    for t in range(x_norm.shape[1]):
        bpas = construct_bpa(x_norm[:, t])
        if bpas.ndim == 1:
            bpas = bpas.reshape(1, 3)
        out[t] = improved_ds_fuse(bpas)[1]
    return out


def proximity_signals(worker_pos):
    crane_center = np.array([27.0, 14.0])
    dist = np.linalg.norm(worker_pos[:, :, :2] - crane_center, axis=2)
    return np.clip(1.0 - dist / 15.0, 0.0, 1.0)


def modality_intensities(ds3):
    """Convert raw DS-3 measurements to physical hazard intensities in [0, 1]."""
    mac = 10.0
    a_lim = 0.15
    gas_intensity = np.clip((np.max(ds3["gas_signals"], axis=0) / mac - 1.0) / 4.0, 0.0, 1.0)
    vib_intensity = np.clip((np.max(ds3["vib_rms"], axis=0) / a_lim - 0.2) / 0.6, 0.0, 1.0)
    prox_intensity = np.max(proximity_signals(ds3["worker_pos"]), axis=0)
    return gas_intensity, vib_intensity, prox_intensity


def spatial_gain(coords, source, lam, alpha_occlusion):
    coords = np.asarray(coords, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    dist = np.linalg.norm(coords - source, axis=1)
    vertical_gap = np.abs(coords[:, 2] - source[2])
    occlusion = np.where(vertical_gap > 2.5, alpha_occlusion, 0.0)
    gain = np.exp(-lam * dist) * (1.0 - occlusion)
    return float(np.clip(np.mean(gain), 0.02, 1.0))


def grade_from_score(score):
    bins = [0.25, 0.45, 0.65]
    return np.digitize(score, bins, right=False) + 1


def grade_from_score_with_tau(score, tau):
    bins = np.array([tau - 0.35, tau - 0.15, tau + 0.05], dtype=np.float64)
    bins = np.clip(bins, [0.15, 0.30, 0.50], [0.40, 0.60, 0.85])
    return np.digitize(score, bins, right=False) + 1


def confirm_warning(grades, k):
    """Require repeated evidence before escalating above blue."""
    grades = np.asarray(grades, dtype=np.int32)
    confirmed = grades.copy()
    for t in range(len(grades)):
        if grades[t] <= 1:
            continue
        start = max(0, t - k + 1)
        window = grades[start : t + 1]
        if np.sum(window >= grades[t]) < max(2, int(np.ceil(k / 2))):
            confirmed[t] = max(1, grades[t] - 1)
    return confirmed


def ema_smooth(score, alpha):
    smoothed = np.zeros_like(score, dtype=np.float64)
    smoothed[0] = score[0]
    for t in range(1, len(score)):
        smoothed[t] = alpha * score[t] + (1.0 - alpha) * smoothed[t - 1]
    return smoothed


def run_ds3_pipeline(ds3, params):
    risk_score, warning_grade = predict_ds3_warning_grades(ds3, params)
    f1 = classification_metrics(ds3["risk_global"], warning_grade, average="macro")["f1"]
    gar = float(np.mean(warning_grade == ds3["risk_global"]))
    high_risk = ds3["risk_global"] >= 3
    mr = float(np.sum((warning_grade < ds3["risk_global"]) & high_risk) / max(np.sum(high_risk), 1))

    return {
        "f1": f1,
        "gar": gar,
        "mr": mr,
        "mean_risk_score": float(np.mean(risk_score)),
        "max_risk_score": float(np.max(risk_score)),
    }


def predict_ds3_warning_grades(ds3, params):
    p_gas = ds3["p_gas"]
    p_vib = ds3["p_vib"]
    p_prox = ds3["p_prox"]
    gas_intensity = ds3["gas_intensity"]
    vib_intensity = ds3["vib_intensity"]
    prox_intensity = ds3["prox_intensity"]

    gas_gain = spatial_gain(
        ds3["gas_coords"], [7.0, 4.0, 1.5], params.lambda_gas, params.alpha_occlusion
    )
    vib_gain = spatial_gain(
        ds3["vib_coords"], [35.0, 10.0, 7.5], params.lambda_vibration, params.alpha_occlusion
    )
    prox_gain = spatial_gain(
        ds3["uwb_coords"], [27.0, 14.0, 1.5], params.lambda_proximity, params.alpha_occlusion
    )
    gas_mult = np.clip(gas_gain / ds3["default_gas_gain"], 0.70, 1.25)
    vib_mult = np.clip(vib_gain / ds3["default_vib_gain"], 0.75, 1.20)
    prox_mult = np.clip(prox_gain / ds3["default_prox_gain"], 0.70, 1.25)

    agreement = np.clip(
        1.0 - (1.0 - p_gas) * (1.0 - p_vib) * (1.0 - p_prox) / 0.16,
        0.0,
        1.0,
    )
    consistency = (1.0 - params.delta) + params.delta * agreement

    gas_score = 0.20 + 0.80 * np.clip(gas_intensity * gas_mult, 0.0, 1.0)
    vib_score = 0.20 + 0.38 * np.clip(vib_intensity * vib_mult, 0.0, 1.0)
    prox_score = 0.16 + 0.78 * np.clip(prox_intensity * prox_mult, 0.0, 1.0)
    base_score = np.maximum.reduce([gas_score, vib_score, prox_score])
    cascade = params.beta * (
        0.45 * gas_intensity * prox_intensity
        + 0.30 * vib_intensity * prox_intensity
        + 0.20 * gas_intensity * vib_intensity
    )
    risk_score = np.clip((base_score + cascade) * consistency, 0.0, 1.0)
    warning_score = ema_smooth(risk_score, params.alpha_ema)
    warning_grade = confirm_warning(grade_from_score_with_tau(warning_score, params.tau), params.k_confirm)
    return risk_score, warning_grade


def prepare_pipeline_inputs(ds3):
    """Precompute parameter-independent fusion signals once."""
    ds3 = dict(ds3)
    gas = minmax01(ds3["gas_signals"], upper=50.0)
    vib = minmax01(ds3["vib_rms"], upper=0.15)
    prox = proximity_signals(ds3["worker_pos"])
    print("  Precomputing DS-3 fused modality probabilities...")
    ds3["p_gas"] = fuse_sensor_block(gas)
    ds3["p_vib"] = fuse_sensor_block(vib)
    ds3["p_prox"] = fuse_sensor_block(prox)
    ds3["gas_intensity"], ds3["vib_intensity"], ds3["prox_intensity"] = modality_intensities(ds3)
    defaults = DS3Params()
    ds3["default_gas_gain"] = spatial_gain(
        ds3["gas_coords"], [7.0, 4.0, 1.5], defaults.lambda_gas, defaults.alpha_occlusion
    )
    ds3["default_vib_gain"] = spatial_gain(
        ds3["vib_coords"], [35.0, 10.0, 7.5], defaults.lambda_vibration, defaults.alpha_occlusion
    )
    ds3["default_prox_gain"] = spatial_gain(
        ds3["uwb_coords"], [27.0, 14.0, 1.5], defaults.lambda_proximity, defaults.alpha_occlusion
    )
    return ds3


def format_range(values):
    return f"{np.min(values):.4f}-{np.max(values):.4f}"


def main():
    print("=" * 60)
    print("Experiment 3: DS-3 One-Factor Sensitivity Analysis")
    print("=" * 60)

    ds3 = prepare_pipeline_inputs(load_ds3())
    default_params = DS3Params()

    detail_rows = []
    print("\nBaseline run with default parameters")
    baseline = run_ds3_pipeline(ds3, default_params)
    print(
        f"  F1={baseline['f1']:.4f}, GAR={baseline['gar']:.4f}, MR={baseline['mr']:.4f}"
    )

    for param_name, spec in PARAMETER_GRID.items():
        print(f"\nParameter: {param_name}")
        for value in spec["values"]:
            params = replace(default_params, **{param_name: value})
            metrics = run_ds3_pipeline(ds3, params)
            detail_rows.append(
                {
                    "parameter": param_name,
                    "symbol": spec["symbol"],
                    "value": value,
                    **metrics,
                }
            )
            print(
                f"  {param_name}={value}: "
                f"F1={metrics['f1']:.4f}, GAR={metrics['gar']:.4f}, MR={metrics['mr']:.4f}"
            )

    detail_df = pd.DataFrame(detail_rows)
    save_csv(detail_df, os.path.join(OUT_DIR, "sensitivity_detail.csv"))

    summary_rows = []
    for param_name, spec in PARAMETER_GRID.items():
        sub = detail_df[detail_df["parameter"] == param_name]
        summary_rows.append(
            {
                "Parameter": spec["symbol"],
                "Values": ", ".join(str(v) for v in spec["values"]),
                "F1-Score Range": format_range(sub["f1"]),
                "GAR Range": format_range(sub["gar"]),
                "MR Range": format_range(sub["mr"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, os.path.join(OUT_DIR, "table_vii_sensitivity_ranges.csv"))
    save_latex_table(
        summary_df,
        os.path.join(OUT_DIR, "tab_sensitivity_ranges.tex"),
        caption="One-factor-at-a-time parameter sensitivity analysis on DS-3.",
        label="tab:sensitivity",
        bold_best=False,
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), sharex=False)
    metrics = [("f1", "F1-Score"), ("gar", "GAR"), ("mr", "MR")]
    for ax, (metric, label) in zip(axes, metrics):
        for i, param_name in enumerate(PARAMETER_GRID):
            sub = detail_df[detail_df["parameter"] == param_name]
            x = np.arange(len(sub))
            ax.plot(
                x,
                sub[metric].values,
                marker="o",
                linewidth=0.9,
                markersize=2.5,
                color=IEEE_COLORS[i % len(IEEE_COLORS)],
                label=param_name,
            )
        ax.set_title(label)
        ax.set_xlabel("Value index")
        ax.grid(alpha=0.3, linewidth=0.3)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(fontsize=4.8, loc="best")
    fig.tight_layout()
    save_figure(fig, os.path.join(OUT_DIR, "fig_sensitivity_profiles.png"))

    print("\n" + "=" * 60)
    print("Experiment 3 Complete!")
    print(f"Results saved to {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
