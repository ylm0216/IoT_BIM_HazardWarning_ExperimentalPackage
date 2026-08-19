"""DS-3: IoT-BIM 施工现场仿真数据集 (纯计算, 零伦理)"""
import numpy as np
import os

# ── 区域类型 ──
ZONE_EMPTY, ZONE_PIT, ZONE_CRANE, ZONE_HIGHWORK = 0, 1, 2, 3
ZONE_STORAGE, ZONE_PASSAGE, ZONE_ELECTRIC, ZONE_CORE = 4, 5, 6, 7

def build_bim_grid():
    """构建40x30x12m施工场景BIM网格"""
    zone = np.zeros((40, 30, 12), dtype=np.uint8)
    struct = np.zeros((40, 30, 12), dtype=np.bool_)

    # 分区 (以1层z=0-2为主，其他层按需)
    zone[0:15, 0:8, 0:3] = ZONE_PIT
    zone[20:35, 8:20, :] = ZONE_CRANE
    zone[30:40, 0:20, :] = ZONE_HIGHWORK
    zone[30:40, 20:30, :] = ZONE_STORAGE
    zone[0:10, 20:30, :] = ZONE_ELECTRIC
    zone[15:25, 10:20, :] = ZONE_CORE
    # 通道: 未分配的区域
    mask_empty = zone == ZONE_EMPTY
    zone[mask_empty] = ZONE_PASSAGE

    # 构件: 楼板
    for z in [3, 6, 9]:
        struct[:, :, z] = True
    # 楼梯间开洞
    struct[18:21, 14:17, [3, 6, 9]] = False
    # 外墙
    struct[0, :, :] = True; struct[39, :, :] = True
    struct[:, 0, :] = True; struct[:, 29, :] = True
    # 核心筒墙
    struct[15, 10:20, :] = True; struct[25, 10:20, :] = True
    struct[15:25, 10, :] = True; struct[15:25, 20, :] = True
    # 柱 (每6m)
    for x in range(0, 40, 6):
        for y in range(0, 30, 6):
            struct[x, y, :] = True

    danger = np.zeros_like(zone, dtype=np.bool_)
    danger |= (zone == ZONE_PIT)
    danger |= (zone == ZONE_CRANE)
    danger |= (zone == ZONE_ELECTRIC)
    for z in range(6, 12):
        danger[:, :, z] |= (zone[:, :, z] == ZONE_HIGHWORK)

    return zone, struct, danger

def deploy_sensors():
    """部署传感器节点"""
    gas = np.array([
        [5,3,1.5],[10,5,1.5],[3,6,1.5],[12,2,1.5],[18,15,1.5],
        [22,15,4.5],[33,25,1.5],[36,22,1.5],[5,25,1.5],[8,28,1.5]
    ], dtype=np.float32)
    vib = np.array([
        [35,10,1.5],[35,15,1.5],[35,10,4.5],[35,15,4.5],
        [35,10,7.5],[35,15,7.5],[35,10,10.5],[35,15,10.5]
    ], dtype=np.float32)
    th = np.array([
        [20,15,1.5],[10,20,1.5],[20,15,4.5],
        [20,15,7.5],[20,15,10.5],[10,5,1.5]
    ], dtype=np.float32)
    uwb = np.array([
        [2,2,2.8],[38,28,2.8],[2,28,5.8],[38,2,5.8],
        [2,2,8.8],[38,28,8.8],[2,28,11.8],[38,2,11.8]
    ], dtype=np.float32)
    return {'gas': gas, 'vib': vib, 'th': th, 'uwb': uwb}

def generate_gas_signals(ds1_X, n_steps=3600, n_sensors=10, seed=42):
    """从DS-1真实信号重标定到施工浓度区间"""
    rng = np.random.RandomState(seed)
    signals = np.zeros((n_sensors, n_steps), dtype=np.float32)
    MAC = 10.0  # H2S MAC mg/m3
    for s in range(n_sensors):
        ch = s % ds1_X.shape[1]
        col = ds1_X[:, ch]
        col_min, col_max = col.min(), col.max()
        rng.shuffle(idx := np.arange(len(col)))
        raw = col[idx[:n_steps]] if len(col) >= n_steps else np.tile(col[idx], n_steps // len(col) + 1)[:n_steps]
        # 重标定: [col_min, col_max] → [0.5, 0.5*MAC]
        normed = (raw - col_min) / (col_max - col_min + 1e-8)
        signals[s] = 0.5 + normed * (0.5 * MAC - 0.5)
        signals[s] += rng.randn(n_steps) * 0.2  # 小噪声
        signals[s] = np.clip(signals[s], 0.1, None)
    return signals

def generate_vib_signals(ds2_accel, n_steps=3600, n_sensors=8, fs=256, seed=42):
    """从DS-2提取RMS/Peak/主频摘要"""
    rng = np.random.RandomState(seed)
    vib_rms = np.zeros((n_sensors, n_steps), dtype=np.float32)
    vib_peak = np.zeros((n_sensors, n_steps), dtype=np.float32)
    n_records = ds2_accel.shape[0]
    baseline_indices = np.where(np.arange(n_records) < 5)[0]  # 前5个是基准

    for s in range(n_sensors):
        ch = s % ds2_accel.shape[2]
        rec_idx = rng.choice(baseline_indices)
        accel = ds2_accel[rec_idx, :, ch]
        # 每秒计算RMS和Peak
        for t in range(n_steps):
            seg_start = (t * fs) % len(accel)
            seg = accel[seg_start:seg_start + fs]
            if len(seg) < fs:
                seg = np.concatenate([seg, accel[:fs - len(seg)]])
            # 缩放到施工振动量级 (RMS ~0.02-0.05 m/s²)
            seg = seg / (np.std(seg) + 1e-8) * 0.03
            vib_rms[s, t] = np.sqrt(np.mean(seg ** 2))
            vib_peak[s, t] = np.max(np.abs(seg))
    return vib_rms, vib_peak

def generate_trajectories(zone, struct, n_workers=20, n_steps=3600, seed=42):
    """约束随机游走生成工人轨迹"""
    rng = np.random.RandomState(seed)
    worker_groups = {
        0: (ZONE_PIT, 0), 1: (ZONE_PIT, 0), 2: (ZONE_PIT, 0), 3: (ZONE_PIT, 0),
        4: (ZONE_CRANE, 0), 5: (ZONE_CRANE, 0), 6: (ZONE_CRANE, 0),
        7: (ZONE_HIGHWORK, 1), 8: (ZONE_HIGHWORK, 1), 9: (ZONE_HIGHWORK, 2), 10: (ZONE_HIGHWORK, 2),
        11: (ZONE_STORAGE, 0), 12: (ZONE_STORAGE, 0),
        13: (ZONE_ELECTRIC, 0), 14: (ZONE_ELECTRIC, 0),
        15: (ZONE_CORE, 0), 16: (ZONE_CORE, 1), 17: (ZONE_CORE, 2),
        18: (ZONE_PASSAGE, 0), 19: (ZONE_PASSAGE, 0),
    }
    positions = np.zeros((n_workers, n_steps, 3), dtype=np.float32)

    for w in range(n_workers):
        zone_type, floor = worker_groups.get(w, (ZONE_PASSAGE, 0))
        z_base = floor * 3.0 + 1.5
        # 找该区域的可用坐标
        zone_mask = zone[:, :, floor * 3] == zone_type
        ys, xs = np.where(zone_mask)
        if len(xs) == 0:
            xs, ys = np.array([20]), np.array([15])
        # 初始位置
        idx = rng.randint(len(xs))
        pos = np.array([xs[idx] + 0.5, ys[idx] + 0.5, z_base])
        target = pos.copy()
        stay_timer = 0

        for t in range(n_steps):
            positions[w, t] = pos
            if stay_timer > 0:
                stay_timer -= 1
                continue
            # 到达目标?
            if np.linalg.norm(pos[:2] - target[:2]) < 1.5:
                stay_timer = int(rng.exponential(30))
                idx = rng.randint(len(xs))
                target = np.array([xs[idx] + 0.5, ys[idx] + 0.5, z_base])
                continue
            # 移动
            v = rng.uniform(0.8, 1.5)
            direction = target[:2] - pos[:2]
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
            step = direction * min(v, dist)
            new_pos = pos.copy()
            new_pos[:2] += step
            # 碰撞检测
            gx, gy, gz = int(np.clip(new_pos[0], 0, 39)), int(np.clip(new_pos[1], 0, 29)), int(np.clip(new_pos[2], 0, 11))
            if not struct[gx, gy, gz]:
                pos = new_pos
            else:
                # 选新目标
                idx = rng.randint(len(xs))
                target = np.array([xs[idx] + 0.5, ys[idx] + 0.5, z_base])

    return positions

def inject_scenarios(gas_signals, vib_rms, vib_peak, positions, zone, n_steps=3600):
    """注入6类危险场景"""
    scenario_active = np.zeros((6, n_steps), dtype=np.bool_)
    MAC = 10.0
    a_lim = 0.15

    # S1: 基坑H2S积聚 (300-600s)
    for t in range(300, 600):
        ramp = min(1.0, (t - 300) / 120)
        for s in range(4):  # 前4个是基坑传感器
            gas_signals[s, t] += ramp * 3.0 * MAC  # 最高达3xMAC
    scenario_active[0, 300:600] = True

    # S2: 脚手架异常振动 (900-1200s)
    for t in range(900, 1200):
        ramp = min(1.0, (t - 900) / 100)
        vib_rms[2, t] *= (1.5 + 1.5 * ramp)  # 传感器2 (2F), 幅值增大
        vib_peak[2, t] *= (1.5 + 1.5 * ramp)
    scenario_active[1, 900:1200] = True

    # S3: 工人闯入吊装禁区 (1500-1800s)
    crane_center = np.array([27.0, 14.0, 1.5])
    for t in range(1500, 1800):
        progress = (t - 1500) / 300
        orig = positions[18, 1500].copy()
        positions[18, t, :2] = orig[:2] + (crane_center[:2] - orig[:2]) * progress
        positions[18, t, 2] = 1.5
    scenario_active[2, 1500:1800] = True

    # S4: S1+S3耦合 (2100-2500s)
    for t in range(2100, 2500):
        ramp = min(1.0, (t - 2100) / 100)
        for s in range(4):
            gas_signals[s, t] += ramp * 5.0 * MAC
    # 工人0继续在基坑, 工人19被引入
    pit_pos = np.array([7.0, 4.0, 1.5])
    for t in range(2100, 2500):
        progress = min(1.0, (t - 2100) / 200)
        positions[19, t, :2] = positions[19, 2100, :2] + (pit_pos[:2] - positions[19, 2100, :2]) * progress
    scenario_active[3, 2100:2500] = True

    # S5: S2+工人接近 (2700-3000s)
    for t in range(2700, 3000):
        ramp = min(1.0, (t - 2700) / 100)
        vib_rms[4, t] *= (1.5 + 2.0 * ramp)  # 3F振动传感器
        vib_peak[4, t] *= (1.5 + 2.0 * ramp)
    vib_pos = np.array([35.0, 10.0, 7.5])
    for t in range(2700, 3000):
        progress = min(1.0, (t - 2700) / 200)
        positions[9, t, :2] = positions[9, 2700, :2] + (vib_pos[:2] - positions[9, 2700, :2]) * progress
    scenario_active[4, 2700:3000] = True

    # S6: 链式演化 (3200-3600s)
    for t in range(3200, 3600):
        phase = (t - 3200) / 400
        if phase < 0.3:  # 气体缓慢积聚
            gas_signals[0, t] += phase / 0.3 * 2.0 * MAC
        elif phase < 0.5:  # 浓度继续升高
            gas_signals[0, t] += 2.0 * MAC + (phase - 0.3) / 0.2 * 3.0 * MAC
        else:  # 多源耦合
            gas_signals[0, t] += 5.0 * MAC * min(1.0, (phase - 0.3) / 0.4)
            gas_signals[1, t] += 3.0 * MAC * min(1.0, (phase - 0.5) / 0.3)
            # 工人进入
            pit_pos2 = np.array([5.0, 3.0, 1.5])
            prog = min(1.0, (phase - 0.5) / 0.3)
            positions[18, t, :2] = positions[18, 3200, :2] + (pit_pos2[:2] - positions[18, 3200, :2]) * prog
    scenario_active[5, 3200:3600] = True

    return gas_signals, vib_rms, vib_peak, positions, scenario_active

def generate_risk_labels(gas_signals, vib_rms, positions, zone, sensors, scenario_active, n_steps=3600):
    """纯规则自动生成风险等级标签 (1=蓝, 2=黄, 3=橙, 4=红)"""
    MAC = 10.0
    a_lim = 0.15
    d_safe = 10.0

    # 气体风险
    risk_gas = np.ones((gas_signals.shape[0], n_steps), dtype=np.uint8)
    for s in range(gas_signals.shape[0]):
        for t in range(n_steps):
            c = gas_signals[s, t]
            if c > 5 * MAC:   risk_gas[s, t] = 4
            elif c > 2 * MAC: risk_gas[s, t] = 3
            elif c > MAC:     risk_gas[s, t] = 2
            else:             risk_gas[s, t] = 1

    # 振动风险
    risk_vib = np.ones((vib_rms.shape[0], n_steps), dtype=np.uint8)
    for s in range(vib_rms.shape[0]):
        for t in range(n_steps):
            ratio = vib_rms[s, t] / a_lim
            if ratio > 5:   risk_vib[s, t] = 4
            elif ratio > 2: risk_vib[s, t] = 3
            elif ratio > 1: risk_vib[s, t] = 2
            else:           risk_vib[s, t] = 1

    # 越界风险 (工人到最近危险区边界距离)
    crane_center = np.array([27.0, 14.0])
    risk_boundary = np.ones((positions.shape[0], n_steps), dtype=np.uint8)
    for w in range(positions.shape[0]):
        for t in range(n_steps):
            d = np.linalg.norm(positions[w, t, :2] - crane_center)
            ratio = d / d_safe
            if ratio <= 0.5:   risk_boundary[w, t] = 4
            elif ratio <= 1.0: risk_boundary[w, t] = 3
            elif ratio <= 1.5: risk_boundary[w, t] = 2
            else:              risk_boundary[w, t] = 1

    # 全局风险时序 (每步取最大值)
    risk_global = np.ones(n_steps, dtype=np.uint8)
    for t in range(n_steps):
        levels = [risk_gas[:, t].max(), risk_vib[:, t].max(), risk_boundary[:, t].max()]
        max_level = max(levels)
        n_above_yellow = sum(1 for l in levels if l >= 2)
        if n_above_yellow >= 2:
            risk_global[t] = min(max_level + 1, 4)
        else:
            risk_global[t] = max_level

    return risk_gas, risk_vib, risk_boundary, risk_global

def build_ds3(ds1_data, ds2_data, data_proc_dir, seed=42):
    """构建完整DS-3仿真数据集"""
    proc_path = os.path.join(data_proc_dir, 'ds3_iot_bim.npz')
    if os.path.exists(proc_path):
        print("  DS-3 processed file exists, loading...")
        d = np.load(proc_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    print("  Building DS-3 IoT-BIM simulation...")
    n_steps = 3600

    # 1. BIM空间
    zone, struct, danger = build_bim_grid()
    print(f"    Grid: {zone.shape}, zones: {np.unique(zone)}")

    # 2. 传感器
    sensors = deploy_sensors()

    # 3. 信号生成
    gas_signals = generate_gas_signals(ds1_data['X'], n_steps, seed=seed)
    vib_rms, vib_peak = generate_vib_signals(ds2_data['accel'], n_steps, seed=seed)
    print(f"    Gas signals: {gas_signals.shape}, Vib RMS: {vib_rms.shape}")

    # 4. 人员轨迹
    positions = generate_trajectories(zone, struct, n_steps=n_steps, seed=seed)
    print(f"    Worker positions: {positions.shape}")

    # 5. 场景注入
    gas_signals, vib_rms, vib_peak, positions, scenario_active = \
        inject_scenarios(gas_signals, vib_rms, vib_peak, positions, zone, n_steps)
    print(f"    Scenarios injected: {scenario_active.sum(axis=1)}")

    # 6. 风险标签
    risk_gas, risk_vib, risk_boundary, risk_global = \
        generate_risk_labels(gas_signals, vib_rms, positions, zone, sensors, scenario_active, n_steps)
    print(f"    Risk labels: global max={risk_global.max()}, >1 count={np.sum(risk_global > 1)}")

    # 保存
    os.makedirs(data_proc_dir, exist_ok=True)
    np.savez_compressed(proc_path,
        zone=zone, struct=struct, danger=danger,
        gas_coords=sensors['gas'], vib_coords=sensors['vib'],
        th_coords=sensors['th'], uwb_coords=sensors['uwb'],
        gas_signals=gas_signals, vib_rms=vib_rms, vib_peak=vib_peak,
        worker_pos=positions, scenario_active=scenario_active,
        risk_gas=risk_gas, risk_vib=risk_vib,
        risk_boundary=risk_boundary, risk_global=risk_global,
    )
    print(f"  DS-3 saved to {proc_path}")
    return {'gas_signals': gas_signals, 'vib_rms': vib_rms, 'vib_peak': vib_peak,
            'worker_pos': positions, 'scenario_active': scenario_active,
            'risk_gas': risk_gas, 'risk_vib': risk_vib,
            'risk_boundary': risk_boundary, 'risk_global': risk_global,
            'zone': zone, 'struct': struct}
