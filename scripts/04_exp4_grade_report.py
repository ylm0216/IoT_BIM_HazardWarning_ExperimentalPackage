"""Experiment 4: class-specific warning-grade precision and recall on DS-3.

This script reuses the default DS-3 evaluation pipeline from Experiment 3 and
reports per-grade precision, recall, F1-score, and support for Table VI.
"""
import importlib.util
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from config import RESULTS_DIR
from utils.metrics import save_csv, save_latex_table


OUT_DIR = os.path.join(RESULTS_DIR, "exp04")
os.makedirs(OUT_DIR, exist_ok=True)


def load_exp03_module():
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_exp3_sensitivity.py")
    spec = importlib.util.spec_from_file_location("exp03_sensitivity", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    print("=" * 60)
    print("Experiment 4: DS-3 Warning-Grade Classification Report")
    print("=" * 60)

    exp03 = load_exp03_module()
    ds3 = exp03.prepare_pipeline_inputs(exp03.load_ds3())
    params = exp03.DS3Params()
    _, y_pred = exp03.predict_ds3_warning_grades(ds3, params)
    y_true = ds3["risk_global"].astype(int)

    labels = [1, 2, 3, 4]
    names = ["Blue", "Yellow", "Orange", "Red"]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for label, name in zip(labels, names):
        item = report[name]
        rows.append(
            {
                "Grade": name,
                "Label": label,
                "Precision": item["precision"],
                "Recall": item["recall"],
                "F1-Score": item["f1-score"],
                "Support": int(item["support"]),
            }
        )

    df = pd.DataFrame(rows)
    save_csv(df, os.path.join(OUT_DIR, "table_vi_grade_report.csv"))
    save_latex_table(
        df,
        os.path.join(OUT_DIR, "tab_grade_report.tex"),
        caption="Class-specific warning-grade performance on DS-3.",
        label="tab:grade_report",
        bold_best=False,
    )

    print("\nPer-grade warning performance:")
    print(df.to_string(index=False, formatters={
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1-Score": "{:.4f}".format,
    }))
    print("\nPredicted grade counts:")
    pred_counts = dict(zip(*np.unique(y_pred, return_counts=True)))
    true_counts = dict(zip(*np.unique(y_true, return_counts=True)))
    print(f"  True:      {true_counts}")
    print(f"  Predicted: {pred_counts}")

    print("\n" + "=" * 60)
    print("Experiment 4 Complete!")
    print(f"Results saved to {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
