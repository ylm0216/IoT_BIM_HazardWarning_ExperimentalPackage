"""01_exp1_fusion_accuracy.py - 实验一: 融合精度对比 (DS-1 + DS-2)"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from config import DATA_PROC, RESULTS_DIR, N_FOLDS, RANDOM_SEED, set_seed, setup_ieee_style, IEEE_COLORS
from utils.metrics import classification_metrics, wilcoxon_test, confidence_interval_95, save_csv, save_latex_table, save_figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

set_seed()
setup_ieee_style()
OUT_DIR = os.path.join(RESULTS_DIR, 'exp01')
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("Experiment 1: Fusion Accuracy Comparison")
print("=" * 60)

# ── 加载数据 ──
ds1 = np.load(os.path.join(DATA_PROC, 'ds1_gas.npz'))
X, y = ds1['X'], ds1['y']
print(f"DS-1: X{X.shape}, classes={np.unique(y)}")

# ── 归一化 ──
from sklearn.preprocessing import MinMaxScaler

# ── 改进D-S适配多类分类 ──
# 对于多类问题, 改进D-S用于特征融合, 融合后用分类器
from proposed.improved_ds import construct_bpa, improved_ds_fuse, standard_ds_fuse
from baselines.fusion_baselines import WeightedAverageFusion, EKFFusion, LSTMFusion, SingleSensorBest
from sklearn.linear_model import LogisticRegression

class DSFusionClassifier:
    """D-S融合 + 分类器 (丰富特征版)

    特征构成:
    - 每个传感器的BPA (N×3)
    - 可信度权重 (N)
    - 融合后BPA (3)
    总特征维度 = N*3 + N + 3 = N*4 + 3
    """
    def __init__(self, method='improved'):
        self.method = method
        self.clf = LogisticRegression(max_iter=1000, random_state=42,
                                      multi_class='multinomial', C=1.0)

    def fit(self, X, y):
        self.scaler_min = X.min(axis=0)
        self.scaler_range = X.max(axis=0) - X.min(axis=0)
        self.scaler_range[self.scaler_range < 1e-8] = 1.0
        features = self._extract(X)
        self.clf.fit(features, y)
        return self

    def predict(self, X):
        return self.clf.predict(self._extract(X))

    def _extract(self, X):
        from proposed.improved_ds import credibility_weights, discount_bpa
        X_norm = np.clip((X - self.scaler_min) / self.scaler_range, 0, 1)
        T, N = X_norm.shape
        # 特征: 各传感器BPA + 可信度 + 融合BPA
        feat_dim = N * 3 + N + 3
        features = np.zeros((T, feat_dim), dtype=np.float64)
        fuse_fn = improved_ds_fuse if self.method == 'improved' else standard_ds_fuse
        for t in range(T):
            bpas = construct_bpa(X_norm[t])
            if bpas.ndim == 1:
                bpas = bpas.reshape(1, 3)
            crd = credibility_weights(bpas) if self.method == 'improved' else np.ones(N) / N
            fused = fuse_fn(bpas)
            features[t, :N*3] = bpas.ravel()     # 各传感器BPA
            features[t, N*3:N*4] = crd            # 可信度权重
            features[t, N*4:] = fused             # 融合BPA
        return features

# ── 方法列表 ──
methods = {
    'Improved D-S (Ours)': lambda: DSFusionClassifier('improved'),
    'Standard D-S':        lambda: DSFusionClassifier('standard'),
    'LSTM Fusion':         lambda: LSTMFusion(hidden_size=64, num_layers=1, epochs=30, seq_len=5, batch_size=128),
    'EKF Fusion':          lambda: EKFFusion(),
    'Weighted Average':    lambda: WeightedAverageFusion(),
    'Single Sensor':       lambda: SingleSensorBest(),
}

# ── 5折CV ──
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
results = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\nFold {fold_idx + 1}/{N_FOLDS}: train={len(train_idx)}, test={len(test_idx)}")

    for name, model_fn in methods.items():
        print(f"  Running {name}...", end=' ', flush=True)
        try:
            model = model_fn()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = classification_metrics(y_test, y_pred, average='macro')
            results.append({
                'method': name, 'fold': fold_idx + 1,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
            })
            print(f"F1={metrics['f1']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'method': name, 'fold': fold_idx + 1,
                'precision': 0, 'recall': 0, 'f1': 0,
            })

# ── 汇总 ──
df = pd.DataFrame(results)
save_csv(df, os.path.join(OUT_DIR, 'fusion_accuracy_detail.csv'))

# 汇总表
summary_rows = []
proposed_f1 = df[df['method'] == 'Improved D-S (Ours)']['f1'].values
for name in methods:
    mdf = df[df['method'] == name]
    for metric in ['precision', 'recall', 'f1']:
        vals = mdf[metric].values
        mean, ci_lo, ci_hi = confidence_interval_95(vals)
        p_val = wilcoxon_test(proposed_f1, vals) if name != 'Improved D-S (Ours)' else float('nan')
        summary_rows.append({
            'method': name, 'metric': metric,
            'mean': mean, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
            'p_value': p_val,
        })

summary_df = pd.DataFrame(summary_rows)
save_csv(summary_df, os.path.join(OUT_DIR, 'fusion_accuracy_summary.csv'))

# ── 主对比表 (用于论文) ──
pivot = df.groupby('method')[['precision', 'recall', 'f1']].agg(['mean', 'std']).reset_index()
pivot.columns = ['method', 'prec_mean', 'prec_std', 'rec_mean', 'rec_std', 'f1_mean', 'f1_std']
# 排序: 本文方法在最后(IEEE惯例)
order = ['Single Sensor', 'Weighted Average', 'EKF Fusion', 'Standard D-S', 'LSTM Fusion', 'Improved D-S (Ours)']
pivot['sort_key'] = pivot['method'].map({n: i for i, n in enumerate(order)})
pivot = pivot.sort_values('sort_key').drop('sort_key', axis=1)

table_df = pd.DataFrame({
    'Method': pivot['method'],
    'Precision': pivot['prec_mean'],
    'Recall': pivot['rec_mean'],
    'F1-Score': pivot['f1_mean'],
})
save_latex_table(table_df, os.path.join(OUT_DIR, 'tab_fusion_accuracy.tex'),
                 caption='Fusion Accuracy Comparison on DS-1 (5-Fold CV)',
                 label='tab:fusion_accuracy', bold_best=True, higher_better=True)

# ── 柱状图 ──
fig, ax = plt.subplots(figsize=(7.16, 3.0))
method_names = pivot['method'].tolist()
x = np.arange(len(method_names))
width = 0.25
bars1 = ax.bar(x - width, pivot['prec_mean'], width, yerr=pivot['prec_std'],
               label='Precision', color=IEEE_COLORS[0], capsize=2, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, pivot['rec_mean'], width, yerr=pivot['rec_std'],
               label='Recall', color=IEEE_COLORS[1], capsize=2, edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, pivot['f1_mean'], width, yerr=pivot['f1_std'],
               label='F1-Score', color=IEEE_COLORS[2], capsize=2, edgecolor='white', linewidth=0.5)
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' (Ours)', '\n(Ours)') for n in method_names], rotation=15, ha='right', fontsize=6)
ax.legend(loc='lower right', fontsize=6)
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3, linewidth=0.3)
fig.tight_layout()
save_figure(fig, os.path.join(OUT_DIR, 'fig_fusion_accuracy.png'))

print("\n" + "=" * 60)
print("Experiment 1 Complete!")
print(f"Results saved to {OUT_DIR}")
print("=" * 60)

# 打印最终结果
print("\nFinal Summary:")
for _, row in pivot.iterrows():
    print(f"  {row['method']:25s}  P={row['prec_mean']:.4f}  R={row['rec_mean']:.4f}  F1={row['f1_mean']:.4f}")
