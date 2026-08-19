"""Experiment 5: measured latency variability for DS-3 real-time evaluation.

The script performs 10 independent timing runs for each sensor-node scale.
Each run measures the sensing, fusion, spatial mapping, inference, and alerting
stages with ``time.perf_counter`` and redraws Figure 5 with standard-deviation
error bars computed from the measured end-to-end latency records.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR, RANDOM_SEED, setup_ieee_style
from proposed.improved_ds import construct_bpa, improved_ds_fuse
from utils.metrics import save_csv


OUT_DIR = os.path.join(RESULTS_DIR, "exp05")
os.makedirs(OUT_DIR, exist_ok=True)

NODE_COUNTS = [50, 100, 200, 300, 400, 500]
N_REPEATS = 10
GRID_SHAPE = (40, 30, 12)


def timed_call(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def sensing_stage(rng, node_count):
    gas = np.clip(rng.normal(0.55, 0.18, size=node_count), 0.0, 1.0)
    vibration = np.clip(rng.normal(0.45, 0.16, size=node_count), 0.0, 1.0)
    proximity = np.clip(rng.beta(2.0, 4.0, size=node_count), 0.0, 1.0)
    coordinates = np.column_stack(
        [
            rng.uniform(0.0, 40.0, node_count),
            rng.uniform(0.0, 30.0, node_count),
            rng.uniform(0.0, 12.0, node_count),
        ]
    )
    return gas, vibration, proximity, coordinates


def fusion_stage(readings):
    fused = []
    for modality in readings:
        bpas = construct_bpa(modality)
        fused.append(improved_ds_fuse(bpas)[1])
    return np.array(fused, dtype=np.float64)


def mapping_stage(fused_prob, coordinates):
    x = np.arange(GRID_SHAPE[0], dtype=np.float64)
    y = np.arange(GRID_SHAPE[1], dtype=np.float64)
    z = np.arange(GRID_SHAPE[2], dtype=np.float64)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    risk = np.zeros(len(grid), dtype=np.float64)
    node_weight = np.mean(fused_prob)
    chunk_size = 1200
    for start in range(0, len(grid), chunk_size):
        chunk = grid[start : start + chunk_size]
        dist = np.linalg.norm(chunk[:, None, :] - coordinates[None, :, :], axis=2)
        risk[start : start + chunk_size] = np.max(np.exp(-0.12 * dist), axis=1) * node_weight
    return risk.reshape(GRID_SHAPE)


def inference_stage(risk_grid):
    risk_series = np.percentile(risk_grid.reshape(-1), [70, 85, 95, 99])
    smoothed = np.zeros_like(risk_series)
    smoothed[0] = risk_series[0]
    for i in range(1, len(risk_series)):
        smoothed[i] = 0.3 * risk_series[i] + 0.7 * smoothed[i - 1]
    return np.digitize(smoothed, [0.25, 0.45, 0.65], right=False) + 1


def alerting_stage(grades):
    confirmed = []
    for idx, grade in enumerate(grades):
        window = grades[max(0, idx - 4) : idx + 1]
        if np.sum(window >= grade) >= max(2, int(np.ceil(len(window) / 2))):
            confirmed.append(int(grade))
        else:
            confirmed.append(max(1, int(grade) - 1))
    return confirmed


def run_one_latency_trial(node_count, repeat_idx):
    rng = np.random.default_rng(RANDOM_SEED + node_count * 100 + repeat_idx)

    sensor_payload, sensing_ms = timed_call(lambda: sensing_stage(rng, node_count))
    gas, vibration, proximity, coordinates = sensor_payload
    fused_prob, fusion_ms = timed_call(lambda: fusion_stage([gas, vibration, proximity]))
    risk_grid, mapping_ms = timed_call(lambda: mapping_stage(fused_prob, coordinates))
    grades, inference_ms = timed_call(lambda: inference_stage(risk_grid))
    _, alerting_ms = timed_call(lambda: alerting_stage(grades))

    return {
        "node_count": node_count,
        "repeat": repeat_idx,
        "sensing_ms": sensing_ms,
        "fusion_ms": fusion_ms,
        "mapping_ms": mapping_ms,
        "inference_ms": inference_ms,
        "alerting_ms": alerting_ms,
        "end_to_end_ms": sensing_ms + fusion_ms + mapping_ms + inference_ms + alerting_ms,
    }


def run_latency_trials():
    rows = []
    for node_count in NODE_COUNTS:
        print(f"  Measuring node_count={node_count}")
        for repeat_idx in range(1, N_REPEATS + 1):
            rows.append(run_one_latency_trial(node_count, repeat_idx))
    return pd.DataFrame(rows)


def summarize_repeats(repeats):
    stage_cols = [c for c in repeats.columns if c.endswith("_ms") and c != "end_to_end_ms"]
    agg = repeats.groupby("node_count").agg(
        end_to_end_mean_ms=("end_to_end_ms", "mean"),
        end_to_end_sd_ms=("end_to_end_ms", "std"),
    )
    for col in stage_cols:
        agg[col.replace("_ms", "_mean_ms")] = repeats.groupby("node_count")[col].mean()
    return agg.reset_index()


def plot_latency(summary):
    node_counts = summary["node_count"].values
    stage_cols = [
        "sensing_mean_ms",
        "fusion_mean_ms",
        "mapping_mean_ms",
        "inference_mean_ms",
        "alerting_mean_ms",
    ]
    labels = [
        r"$\mathrm{T}_{\mathrm{sense}}$",
        r"$\mathrm{T}_{\mathrm{fuse}}$",
        r"$\mathrm{T}_{\mathrm{map}}$",
        r"$\mathrm{T}_{\mathrm{infer}}$",
        r"$\mathrm{T}_{\mathrm{alert}}$",
    ]
    colors = ["#d9d9d9", "#8fb8e3", "#3569b1", "#f0a13a", "#c64045"]

    fig, ax = plt.subplots(figsize=(2.95, 2.72))
    bottom = np.zeros(len(summary))
    for col, label, color in zip(stage_cols, labels, colors):
        ax.bar(
            node_counts,
            summary[col].values,
            bottom=bottom,
            width=33,
            label=label,
            color=color,
            edgecolor="0.25",
            linewidth=0.45,
        )
        bottom += summary[col].values

    y_mean = summary["end_to_end_mean_ms"].values
    y_sd = summary["end_to_end_sd_ms"].values
    ax.errorbar(
        node_counts,
        y_mean,
        yerr=y_sd,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        zorder=10,
    )
    label_offset = max(0.8, float(np.max(y_mean + y_sd)) * 0.025)
    for x, y, sd in zip(node_counts, y_mean, y_sd):
        ax.text(x, y + sd + label_offset, f"{y:.1f}", ha="center", va="bottom", fontsize=9.2)

    ax.set_xlabel("Number of sensor nodes", fontsize=11.2)
    ax.set_ylabel("Latency (ms)", fontsize=11.2)
    ax.set_xlim(25, 525)
    ax.set_ylim(0, float(np.max(y_mean + y_sd)) * 1.22)
    ax.set_xticks(node_counts)
    handles, handle_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        handle_labels[::-1],
        fontsize=8.8,
        loc="upper left",
        frameon=False,
        borderpad=0.2,
        handlelength=1.5,
        labelspacing=0.25,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("0.25")
    ax.tick_params(direction="in", top=True, right=True, labelsize=9.8)
    ax.grid(False)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.17, top=0.945)
    return fig


def main():
    setup_ieee_style()
    repeats = run_latency_trials()
    summary = summarize_repeats(repeats)

    save_csv(repeats, os.path.join(OUT_DIR, "latency_repeats_10runs.csv"))
    save_csv(summary, os.path.join(OUT_DIR, "latency_summary_with_sd.csv"))

    fig = plot_latency(summary)
    fig.savefig(
        os.path.join(OUT_DIR, "fig5_latency_original_style_errorbars.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(fig)

    print("Latency summary:")
    print(
        summary[["node_count", "end_to_end_mean_ms", "end_to_end_sd_ms"]].to_string(
            index=False,
            formatters={
                "end_to_end_mean_ms": "{:.2f}".format,
                "end_to_end_sd_ms": "{:.2f}".format,
            },
        )
    )
    print(f"Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
