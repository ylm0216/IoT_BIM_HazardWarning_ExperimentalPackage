"""融合基线方法: 加权平均/EKF/贝叶斯/LSTM/标准D-S/单传感器"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ═══ 1. 加权平均融合 ═══
class WeightedAverageFusion:
    def __init__(self):
        self.models = []
        self.weights = []
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.scaler.fit(X)
        X_s = self.scaler.transform(X)
        n_sensors = X.shape[1]
        self.models = []
        self.weights = []
        for s in range(n_sensors):
            lr = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs',
                                   multi_class='multinomial')
            lr.fit(X_s[:, s:s+1], y)
            acc = lr.score(X_s[:, s:s+1], y)
            self.models.append(lr)
            self.weights.append(acc)
        w = np.array(self.weights)
        self.weights = w / w.sum()
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        n_classes = len(self.models[0].classes_)
        probs = np.zeros((len(X), n_classes))
        for s, (model, w) in enumerate(zip(self.models, self.weights)):
            probs += w * model.predict_proba(X_s[:, s:s+1])
        return self.models[0].classes_[probs.argmax(axis=1)]


# ═══ 2. EKF融合 ═══
class EKFFusion:
    def __init__(self, alpha=0.95, q=0.01):
        self.alpha = alpha
        self.q = q
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=500, random_state=42, multi_class='multinomial')

    def fit(self, X, y):
        self.scaler.fit(X)
        X_filtered = self._filter(self.scaler.transform(X))
        self.classifier.fit(X_filtered, y)
        return self

    def predict(self, X):
        X_filtered = self._filter(self.scaler.transform(X))
        return self.classifier.predict(X_filtered)

    def _filter(self, X):
        T, N = X.shape
        x_est = np.zeros(N)
        P = np.eye(N) * 1.0
        Q = np.eye(N) * self.q
        R = np.diag(np.var(X, axis=0) + 1e-6)
        A = np.eye(N) * self.alpha
        result = np.zeros_like(X)
        for t in range(T):
            # 预测
            x_pred = A @ x_est
            P_pred = A @ P @ A.T + Q
            # 更新
            K = P_pred @ np.linalg.inv(P_pred + R)
            x_est = x_pred + K @ (X[t] - x_pred)
            P = (np.eye(N) - K) @ P_pred
            result[t] = x_est
        return result


# ═══ 3. 贝叶斯融合 ═══
class BayesianFusion:
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.scaler.fit(X)
        X_s = self.scaler.transform(X)
        self.classes_ = np.unique(y)
        self.models = []
        for s in range(X.shape[1]):
            gnb = GaussianNB()
            gnb.fit(X_s[:, s:s+1], y)
            self.models.append(gnb)
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        n_classes = len(self.classes_)
        log_probs = np.zeros((len(X), n_classes))
        for s, model in enumerate(self.models):
            lp = model.predict_log_proba(X_s[:, s:s+1])
            log_probs += lp
        return self.classes_[log_probs.argmax(axis=1)]


# ═══ 4. LSTM融合 ═══
class LSTMFusion:
    def __init__(self, hidden_size=64, num_layers=1, lr=1e-3, epochs=30, seq_len=5, batch_size=128):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y):
        import torch, torch.nn as nn
        self.scaler.fit(X)
        X_s = self.scaler.transform(X).astype(np.float32)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([label_map[yi] for yi in y])

        # 构建序列
        X_seq, y_seq = [], []
        for i in range(self.seq_len, len(X_s)):
            X_seq.append(X_s[i - self.seq_len:i])
            y_seq.append(y_mapped[i])
        X_seq = torch.FloatTensor(np.array(X_seq))
        y_seq = torch.LongTensor(np.array(y_seq))

        # 模型
        class LSTMModel(nn.Module):
            def __init__(self_, input_size, hidden_size, num_layers, n_classes):
                super().__init__()
                self_.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=0.3)
                self_.fc = nn.Sequential(
                    nn.Dropout(0.3), nn.Linear(hidden_size, 64), nn.ReLU(),
                    nn.Dropout(0.2), nn.Linear(64, n_classes))
            def forward(self_, x):
                out, _ = self_.lstm(x)
                return self_.fc(out[:, -1, :])

        device = 'cpu'
        self.model = LSTMModel(X_s.shape[1], self.hidden_size, self.num_layers, n_classes).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X_seq, y_seq)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb.to(device)), yb.to(device))
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        import torch
        X_s = self.scaler.transform(X).astype(np.float32)
        # 序列化
        preds = []
        # 前 seq_len 个样本用第一个窗口
        self.model.eval()
        with torch.no_grad():
            for i in range(len(X_s)):
                start = max(0, i - self.seq_len + 1)
                seq = X_s[start:i + 1]
                if len(seq) < self.seq_len:
                    pad = np.tile(seq[0:1], (self.seq_len - len(seq), 1))
                    seq = np.concatenate([pad, seq])
                out = self.model(torch.FloatTensor(seq).unsqueeze(0))
                preds.append(out.argmax(dim=1).item())
        return self.classes_[np.array(preds)]


# ═══ 5. 标准D-S (消融) ═══
class StandardDSFusion:
    def __init__(self, sensor_type='default', tau=0.5):
        self.sensor_type = sensor_type
        self.tau = tau

    def fit(self, X, y):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ < 1e-8] = 1.0
        return self

    def predict(self, X):
        from proposed.improved_ds import fuse_timeseries
        X_norm = (X - self.min_) / self.range_
        X_norm = np.clip(X_norm, 0, 1)
        fused = fuse_timeseries(X_norm, self.sensor_type, method='standard')
        preds = np.zeros(len(X), dtype=np.int32)
        preds[fused[:, 1] > self.tau] = 1
        return preds


# ═══ 6. 单传感器 (消融) ═══
class SingleSensorBest:
    def __init__(self):
        self.best_sensor = 0
        self.best_model = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.scaler.fit(X)
        X_s = self.scaler.transform(X)
        best_acc = -1
        for s in range(X.shape[1]):
            lr = LogisticRegression(max_iter=500, random_state=42, multi_class='multinomial')
            lr.fit(X_s[:, s:s+1], y)
            acc = lr.score(X_s[:, s:s+1], y)
            if acc > best_acc:
                best_acc = acc
                self.best_sensor = s
                self.best_model = lr
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return self.best_model.predict(X_s[:, self.best_sensor:self.best_sensor+1])
