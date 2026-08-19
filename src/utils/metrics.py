"""评价指标、统计检验、结果导出"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import wilcoxon, t as t_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ═══════════════ 评价指标 ═══════════════

def classification_metrics(y_true, y_pred, average='macro'):
    return {
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall':    recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1':        f1_score(y_true, y_pred, average=average, zero_division=0),
    }

def risk_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        'mae':  np.mean(np.abs(y_true - y_pred)),
        'rmse': np.sqrt(np.mean((y_true - y_pred)**2)),
        'grade_agreement': np.mean(np.round(y_true) == np.round(y_pred)),
    }

def warning_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pos = y_true > 0
    neg = ~pos
    tp = np.sum((y_pred > 0) & pos)
    fp = np.sum((y_pred > 0) & neg)
    fn = np.sum((y_pred == 0) & pos)
    tn = np.sum((y_pred == 0) & neg)
    accuracy = (tp + tn) / max(len(y_true), 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    return {'accuracy': accuracy, 'fpr': fpr, 'fnr': fnr}

def detection_latency(y_true, y_pred, onset_indices):
    latencies = []
    for onset in onset_indices:
        detected = np.where(y_pred[onset:] > 0)[0]
        latencies.append(detected[0] if len(detected) > 0 else len(y_pred) - onset)
    return np.mean(latencies) if latencies else float('inf')

# ═══════════════ 统计检验 ═══════════════

def wilcoxon_test(proposed_scores, baseline_scores):
    if len(proposed_scores) < 5:
        return float('nan')
    try:
        _, p = wilcoxon(proposed_scores, baseline_scores, alternative='greater')
        return p
    except (ValueError, ZeroDivisionError):
        return float('nan')

def confidence_interval_95(scores):
    scores = np.asarray(scores)
    n = len(scores)
    if n < 2:
        return scores.mean(), scores.mean(), scores.mean()
    mean = scores.mean()
    std = scores.std(ddof=1)
    t_crit = t_dist.ppf(0.975, df=n - 1)
    margin = t_crit * std / np.sqrt(n)
    return mean, mean - margin, mean + margin

# ═══════════════ 结果导出 ═══════════════

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, float_format='%.4f')
    print(f"  CSV saved: {path}")

def save_latex_table(df, path, caption="", label="", bold_best=True, higher_better=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    lines = []
    lines.append(r'\begin{table}[!t]')
    lines.append(r'\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{label}}}')
    lines.append(r'\renewcommand{\arraystretch}{1.1}')
    lines.append(r'\setlength{\tabcolsep}{3pt}')
    col_fmt = 'l' + 'c' * (len(df.columns) - 1)
    lines.append(f'\\begin{{tabular}}{{{col_fmt}}}')
    lines.append(r'\toprule')
    header = ' & '.join(df.columns) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    # find best per numeric column
    best_idx = {}
    if bold_best:
        for col in numeric_cols:
            if higher_better:
                best_idx[col] = df[col].idxmax()
            else:
                best_idx[col] = df[col].idxmin()
    for idx, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if col in numeric_cols and isinstance(val, (int, float, np.floating)):
                s = f'{val:.4f}'
                if bold_best and col in best_idx and idx == best_idx[col]:
                    s = f'\\textbf{{{s}}}'
                cells.append(s)
            else:
                s = str(val)
                if bold_best and 'Ours' in str(val):
                    s = f'\\textbf{{{s}}}'
                cells.append(s)
        lines.append(' & '.join(cells) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX saved: {path}")

def save_figure(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  Figure saved: {path}")
