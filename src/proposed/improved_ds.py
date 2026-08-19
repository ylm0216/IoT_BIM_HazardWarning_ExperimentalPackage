"""改进D-S证据理论融合算法"""
import numpy as np
from functools import reduce

# Jousselme距离矩阵 (焦元: {normal}, {abnormal}, {Theta})
D_MATRIX = np.array([
    [1.0, 0.0, 0.5],
    [0.0, 1.0, 0.5],
    [0.5, 0.5, 1.0],
], dtype=np.float64)

# 焦元交集表: INTER[i][j] = 交集焦元索引, -1=空集
# 0=normal, 1=abnormal, 2=Theta
INTER = np.array([
    [ 0, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2],
], dtype=np.int8)


def construct_bpa(x, a1=0.30, a2=0.60, b1=0.50, b2=0.80):
    """梯形隶属度构建BPA, x为标量或向量(已归一化到[0,1])"""
    x = np.asarray(x, dtype=np.float64)
    m_normal = np.where(x <= a1, 1.0, np.where(x >= a2, 0.0, (a2 - x) / (a2 - a1)))
    m_abnormal = np.where(x <= b1, 0.0, np.where(x >= b2, 1.0, (x - b1) / (b2 - b1)))
    m_theta = np.clip(1.0 - m_normal - m_abnormal, 0, 1)
    # 归一化
    total = m_normal + m_abnormal + m_theta
    total = np.where(total < 1e-10, 1.0, total)
    bpa = np.stack([m_normal / total, m_abnormal / total, m_theta / total], axis=-1)
    return bpa


def construct_bpa_batch(X_norm, sensor_type='default'):
    """批量构建BPA, X_norm shape=(T, N), 返回 (T, N, 3)"""
    from config import SENSOR_BPA_PARAMS
    params = SENSOR_BPA_PARAMS.get(sensor_type, SENSOR_BPA_PARAMS['default'])
    T, N = X_norm.shape
    bpas = np.zeros((T, N, 3), dtype=np.float64)
    for j in range(N):
        bpas[:, j, :] = construct_bpa(X_norm[:, j], **params)
    return bpas


def jousselme_distance(m1, m2):
    """两个BPA之间的Jousselme距离"""
    diff = m1 - m2
    return np.sqrt(0.5 * diff @ D_MATRIX @ diff)


def distance_matrix(bpas):
    """计算N个BPA的距离矩阵, bpas shape=(N, 3)"""
    N = bpas.shape[0]
    dist = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            d = jousselme_distance(bpas[i], bpas[j])
            dist[i, j] = dist[j, i] = d
    return dist


def credibility_weights(bpas):
    """从BPA集合计算可信度权重, bpas shape=(N, 3)"""
    N = bpas.shape[0]
    dist = distance_matrix(bpas)
    # 支持度
    sup = np.zeros(N)
    for i in range(N):
        sup[i] = np.sum(1.0 - dist[i]) / max(N - 1, 1)
    # 归一化
    total = sup.sum()
    if total < 1e-10:
        return np.ones(N) / N
    return sup / total


def discount_bpa(bpa, crd):
    """Shafer折扣: 低可信度→更多概率到Theta"""
    m = bpa.copy()
    m[0] *= crd  # normal
    m[1] *= crd  # abnormal
    m[2] = 1.0 - m[0] - m[1]
    return m


def dempster_combine(m1, m2):
    """标准Dempster合成规则 (3焦元)"""
    combined = np.zeros(3, dtype=np.float64)
    K = 0.0
    for i in range(3):
        for j in range(3):
            inter = INTER[i, j]
            prod = m1[i] * m2[j]
            if inter == -1:
                K += prod
            else:
                combined[inter] += prod
    if K >= 1.0 - 1e-10:
        return np.array([0, 0, 1.0])  # 完全冲突→全不确定
    combined /= (1.0 - K)
    return combined


def improved_ds_fuse(bpas):
    """改进D-S融合: 可信度加权 + Dempster合成, bpas shape=(N, 3)"""
    N = bpas.shape[0]
    if N == 0:
        return np.array([0, 0, 1.0])
    if N == 1:
        return bpas[0].copy()

    # 1. 计算可信度权重
    crd = credibility_weights(bpas)

    # 2. Shafer折扣修正
    modified = np.array([discount_bpa(bpas[i], crd[i]) for i in range(N)])

    # 3. 迭代Dempster合成
    result = reduce(dempster_combine, modified)
    return result


def standard_ds_fuse(bpas):
    """标准D-S融合 (不做可信度修正)"""
    if bpas.shape[0] == 0:
        return np.array([0, 0, 1.0])
    return reduce(dempster_combine, bpas)


def fuse_timeseries(X_norm, sensor_type='default', method='improved'):
    """
    时序融合, X_norm shape=(T, N), 返回 (T,3) 融合BPA
    method: 'improved' | 'standard'
    """
    from config import SENSOR_BPA_PARAMS
    params = SENSOR_BPA_PARAMS.get(sensor_type, SENSOR_BPA_PARAMS['default'])
    T, N = X_norm.shape
    fused = np.zeros((T, 3), dtype=np.float64)
    fuse_fn = improved_ds_fuse if method == 'improved' else standard_ds_fuse

    for t in range(T):
        bpas = construct_bpa(X_norm[t], **params)
        if bpas.ndim == 1:
            bpas = bpas.reshape(1, 3)
        fused[t] = fuse_fn(bpas)
    return fused


def classify_from_bpa(fused_bpa, tau=0.6):
    """从融合BPA判定类别: 0=normal, 1=abnormal, 2=uncertain"""
    T = fused_bpa.shape[0]
    decisions = np.full(T, 2, dtype=np.int32)  # default uncertain
    decisions[fused_bpa[:, 1] > tau] = 1  # abnormal
    decisions[fused_bpa[:, 0] > tau] = 0  # normal
    return decisions


def temporal_confirmation(raw_decisions, k=5, threshold=3):
    """时序持续性约束: 窗口k步中>=threshold步异常方确认"""
    T = len(raw_decisions)
    confirmed = raw_decisions.copy()
    for t in range(k - 1, T):
        window = raw_decisions[t - k + 1:t + 1]
        if np.sum(window == 1) < threshold:
            if confirmed[t] == 1:
                confirmed[t] = 2  # 降为uncertain
    return confirmed


class ImprovedDSFusion:
    """改进D-S融合分类器接口"""

    def __init__(self, sensor_type='default', tau=0.6):
        self.sensor_type = sensor_type
        self.tau = tau
        self.thresholds_ = None

    def fit(self, X, y):
        # 学习归一化参数
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ < 1e-8] = 1.0
        return self

    def predict(self, X):
        X_norm = (X - self.min_) / self.range_
        X_norm = np.clip(X_norm, 0, 1)
        T, N = X_norm.shape
        fused = fuse_timeseries(X_norm, self.sensor_type, method='improved')
        # 多类分类: 使用 abnormal 置信度排序
        # 简化: abnormal置信度 > 0.5 → 异常类, 否则 → 用最近邻or阈值
        preds = np.zeros(T, dtype=np.int32)
        preds[fused[:, 1] > 0.5] = 1
        return preds

    def predict_confidence(self, X):
        X_norm = (X - self.min_) / self.range_
        X_norm = np.clip(X_norm, 0, 1)
        return fuse_timeseries(X_norm, self.sensor_type, method='improved')
