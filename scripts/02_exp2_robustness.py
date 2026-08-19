"""02_exp2_robustness.py - 实验二: 鲁棒性与泛化性"""
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
OUT_DIR = os.path.join(RESULTS_DIR, 'exp02')
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("Experiment 2: Robustness and Generalization")
print("=" * 60)

# ── 加载数据 ──
ds1 = np.load(os.path.join(DATA_PROC, 'ds1_gas.npz'))
X, y, batch = ds1['X'], ds1['y'], ds1['batch']

# ── 复用实验一的方法定义 ──
from proposed.improved_ds import construct_bpa, improved_ds_fuse, standard_ds_fuse
from baselines.fusion_baselines import WeightedAverageFusion, EKFFusion, LSTMFusion, SingleSensorBest
from sklearn.linear_model import LogisticRegression

class DSFusionClassifier:
    def __init__(self, method='improved'):
        self.method = method
        self.clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', C=1.0)
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
        from proposed.improved_ds import credibility_weights
        X_norm = np.clip((X - self.scaler_min) / self.scaler_range, 0, 1)
        T, N = X_norm.shape
        feat_dim = N * 3 + N + 3
        features = np.zeros((T, feat_dim))
        fuse_fn = improved_ds_fuse if self.method == 'improved' else standard_ds_fuse
        for t in range(T):
            bpas = construct_bpa(X_norm[t])
            if bpas.ndim == 1: bpas = bpas.reshape(1, 3)
            crd = credibility_weights(bpas) if self.method == 'improved' else np.ones(N) / N
            fused = fuse_fn(bpas)
            features[t, :N*3] = bpas.ravel()
            features[t, N*3:N*4] = crd
            features[t, N*4:] = fused
        return features

# 消融对比方法 (鲁棒性只跑核心3种 + LSTM)
methods_robust = {
    'Improved D-S (Ours)': lambda: DSFusionClassifier('improved'),
    'Standard D-S':        lambda: DSFusionClassifier('standard'),
    'LSTM Fusion':         lambda: LSTMFusion(hidden_size=64, num_layers=1, epochs=20, seq_len=5, batch_size=128),
    'EKF Fusion':          lambda: EKFFusion(),
    'Weighted Average':    lambda: WeightedAverageFusion(),
    'Single Sensor':       lambda: SingleSensorBest(),
}

# ═══ 2a: 通道屏蔽鲁棒性 ═══
print("\n[2a] Channel Dropout Robustness Test")
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
n_repeats = 5
robust_results = []

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)  # 3折加速

for dr in dropout_rates:
    print(f"\n  Dropout rate: {dr:.0%}")
    for rep in range(n_repeats):
        rng = np.random.RandomState(RANDOM_SEED + rep)
        # 屏蔽通道
        X_masked = X.copy()
        if dr > 0:
            mask = rng.random(X.shape) < dr
            X_masked[mask] = 0

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_masked, y)):
            X_tr, X_te = X_masked[train_idx], X_masked[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            for name, model_fn in methods_robust.items():
                try:
                    model = model_fn()
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)
                    f1 = classification_metrics(y_te, y_pred)['f1']
                except:
                    f1 = 0.0
                robust_results.append({
                    'method': name, 'dropout_rate': dr,
                    'repeat': rep, 'fold': fold_idx, 'f1': f1,
                })

    # 打印当前进度
    for name in methods_robust:
        vals = [r['f1'] for r in robust_results if r['method'] == name and r['dropout_rate'] == dr]
        print(f"    {name:25s}: F1={np.mean(vals):.4f} +/- {np.std(vals):.4f}")

robust_df = pd.DataFrame(robust_results)
save_csv(robust_df, os.path.join(OUT_DIR, 'robustness_dropout.csv'))

# ── 衰减曲线图 ──
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for i, name in enumerate(methods_robust):
    means, stds = [], []
    for dr in dropout_rates:
        vals = robust_df[(robust_df['method'] == name) & (robust_df['dropout_rate'] == dr)]['f1']
        means.append(vals.mean())
        stds.append(vals.std())
    ax.errorbar(dropout_rates, means, yerr=stds, marker='o', label=name.replace(' (Ours)', '*'),
                color=IEEE_COLORS[i], capsize=2, linewidth=1, markersize=3)
ax.set_xlabel('Channel Dropout Rate')
ax.set_ylabel('F1-Score')
ax.legend(fontsize=5, loc='lower left')
ax.grid(alpha=0.3, linewidth=0.3)
fig.tight_layout()
save_figure(fig, os.path.join(OUT_DIR, 'fig_robustness_curve.png'))

# ═══ 2b: 跨批次泛化 ═══
print("\n[2b] Cross-Batch Temporal Generalization")
train_mask = np.isin(batch, [1, 2, 3, 4, 5])
test_mask = np.isin(batch, [6, 7, 8, 9, 10])
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
print(f"  Train (Batch 1-5): {len(X_train)}, Test (Batch 6-10): {len(X_test)}")

gen_results = []
for name, model_fn in methods_robust.items():
    print(f"  Running {name}...", end=' ', flush=True)
    try:
        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = classification_metrics(y_test, y_pred)
        gen_results.append({'method': name, **metrics})
        print(f"F1={metrics['f1']:.4f}")
    except Exception as e:
        print(f"ERROR: {e}")
        gen_results.append({'method': name, 'precision': 0, 'recall': 0, 'f1': 0})

gen_df = pd.DataFrame(gen_results)
save_csv(gen_df, os.path.join(OUT_DIR, 'temporal_generalization.csv'))

# ── 泛化柱状图 ──
fig, ax = plt.subplots(figsize=(3.5, 2.5))
x = np.arange(len(gen_df))
ax.bar(x, gen_df['f1'], color=[IEEE_COLORS[i % len(IEEE_COLORS)] for i in range(len(gen_df))],
       edgecolor='white', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' (Ours)', '\n(Ours)') for n in gen_df['method']], rotation=20, ha='right', fontsize=5.5)
ax.set_ylabel('F1-Score')
ax.set_title('Temporal Generalization (Batch 1-5 → 6-10)')
ax.grid(axis='y', alpha=0.3, linewidth=0.3)
fig.tight_layout()
save_figure(fig, os.path.join(OUT_DIR, 'fig_temporal_generalization.png'))

# ═══ 2c: 消融分析 ═══
print("\n[2c] Ablation Analysis")
ablation_methods = {
    'Single Sensor':       lambda: SingleSensorBest(),
    'Standard D-S':        lambda: DSFusionClassifier('standard'),
    'Improved D-S (Ours)': lambda: DSFusionClassifier('improved'),
}
ablation_results = []
skf5 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
for fold_idx, (train_idx, test_idx) in enumerate(skf5.split(X, y)):
    for name, model_fn in ablation_methods.items():
        try:
            model = model_fn()
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            f1 = classification_metrics(y[test_idx], y_pred)['f1']
        except:
            f1 = 0
        ablation_results.append({'method': name, 'fold': fold_idx + 1, 'f1': f1})

abl_df = pd.DataFrame(ablation_results)
save_csv(abl_df, os.path.join(OUT_DIR, 'ablation.csv'))

# 消融贡献
abl_summary = abl_df.groupby('method')['f1'].agg(['mean', 'std']).reset_index()
save_latex_table(
    pd.DataFrame({
        'Method': abl_summary['method'],
        'F1-Score': abl_summary['mean'],
    }),
    os.path.join(OUT_DIR, 'tab_ablation.tex'),
    caption='Ablation Study: Contribution of Each Component',
    label='tab:ablation', bold_best=True
)

# LaTeX for robustness
robust_summary = robust_df.groupby(['method', 'dropout_rate'])['f1'].mean().reset_index()
pivot_rob = robust_summary.pivot(index='method', columns='dropout_rate', values='f1').reset_index()
save_latex_table(pivot_rob, os.path.join(OUT_DIR, 'tab_robustness.tex'),
                 caption='Robustness Under Channel Dropout', label='tab:robustness')

# LaTeX for generalization
save_latex_table(gen_df[['method', 'precision', 'recall', 'f1']],
                 os.path.join(OUT_DIR, 'tab_generalization.tex'),
                 caption='Temporal Generalization (Batch 1-5 to 6-10)', label='tab:generalization')

print("\n" + "=" * 60)
print("Experiment 2 Complete!")
print(f"Results saved to {OUT_DIR}")
print("=" * 60)
