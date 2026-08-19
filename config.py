"""全局配置：路径、随机种子、IEEE绘图样式、颜色方案"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 路径 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW     = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROC    = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results')
SRC_DIR      = os.path.join(PROJECT_ROOT, 'src')

# 确保 src 在 sys.path 中
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── 随机种子 ──
RANDOM_SEED = 42
N_FOLDS = 5

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# ── IEEE 绘图样式 ──
IEEE_COLORS = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]

METHOD_NAMES_FUSION = [
    'Improved D-S (Ours)', 'LSTM Fusion', 'Standard D-S',
    'EKF Fusion', 'Weighted Average', 'Single Sensor',
]

METHOD_COLORS = {name: IEEE_COLORS[i] for i, name in enumerate(METHOD_NAMES_FUSION)}

def setup_ieee_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.figsize': (3.5, 2.5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.5,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'mathtext.fontset': 'stix',
    })

# ── BPA 传感器参数 ──
SENSOR_BPA_PARAMS = {
    'gas':        {'a1': 0.25, 'a2': 0.55, 'b1': 0.45, 'b2': 0.75},
    'vibration':  {'a1': 0.30, 'a2': 0.60, 'b1': 0.50, 'b2': 0.80},
    'temperature':{'a1': 0.35, 'a2': 0.65, 'b1': 0.55, 'b2': 0.85},
    'proximity':  {'a1': 0.20, 'a2': 0.50, 'b1': 0.40, 'b2': 0.70},
    'default':    {'a1': 0.30, 'a2': 0.60, 'b1': 0.50, 'b2': 0.80},
}

# ── 危险源空间衰减参数 ──
HAZARD_DECAY = {
    'gas_leak':       {'alpha': 0.15, 'max_radius': 25},
    'structural':     {'alpha': 0.40, 'max_radius': 10},
    'falling_object': {'alpha': 0.50, 'max_radius': 8},
    'electrical':     {'alpha': 0.60, 'max_radius': 5},
    'crane_zone':     {'alpha': 0.20, 'max_radius': 20},
}

# ── 级联系数 ──
CASCADE_GAMMA = {
    ('gas_leak', 'fire'):            0.35,
    ('fire', 'structural'):          0.25,
    ('structural', 'falling_object'):0.30,
    ('electrical', 'fire'):          0.30,
    ('crane_zone', 'falling_object'):0.20,
    ('gas_leak', 'structural'):      0.05,
}

# ── 预警阈值 ──
ALERT_THRESHOLDS_BASE = {'blue': 0.25, 'yellow': 0.45, 'orange': 0.65, 'red': 0.80}

# ── DS-3 仿真参数 ──
SIM_DURATION = 3600   # 秒
SIM_DT = 1.0          # 基准时间步
N_WORKERS = 20
GRID_SHAPE = (40, 30, 12)  # x, y, z (米)
FLOOR_HEIGHT = 3.0
