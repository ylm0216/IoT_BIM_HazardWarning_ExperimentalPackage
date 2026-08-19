"""DS-2: IASC-ASCE SHM Benchmark 合成（模态叠加法）"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
import os

def synthesize_ds2(data_proc_dir, seed=42):
    proc_path = os.path.join(data_proc_dir, 'ds2_shm.npz')
    if os.path.exists(proc_path):
        print("  DS-2 processed file exists, loading...")
        d = np.load(proc_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    np.random.seed(seed)
    print("  Synthesizing DS-2 (modal superposition)...")

    n_floors = 4
    fs = 256
    T = 60
    n_samples = fs * T  # 15360
    target_freq = np.array([5.6, 16.1, 24.8, 31.2])
    mass = np.ones(n_floors) * 5.0e5

    # 反推刚度: 构建质量和刚度矩阵, 优化使频率匹配
    def build_K(k_vec):
        K = np.zeros((n_floors, n_floors))
        for i in range(n_floors):
            K[i, i] = k_vec[i] + (k_vec[i + 1] if i < n_floors - 1 else 0)
            if i > 0:
                K[i, i - 1] = -k_vec[i]
            if i < n_floors - 1:
                K[i, i + 1] = -k_vec[i + 1]
        return K

    def freq_error(log_k):
        k_vec = np.exp(log_k)
        M = np.diag(mass)
        K = build_K(k_vec)
        eigvals = np.linalg.eigvalsh(np.linalg.solve(M, K))
        eigvals = np.sort(np.abs(eigvals))
        freqs = np.sqrt(eigvals) / (2 * np.pi)
        return np.sum((freqs - target_freq) ** 2)

    res = minimize(freq_error, np.log(np.ones(n_floors) * 2e7), method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-10})
    k_base = np.exp(res.x)

    M = np.diag(mass)
    K_base = build_K(k_base)
    eigvals_base, phi_base = np.linalg.eigh(np.linalg.solve(M, K_base))
    freq_achieved = np.sqrt(np.abs(np.sort(eigvals_base))) / (2 * np.pi)
    print(f"  Target freq: {target_freq} Hz")
    print(f"  Achieved freq: {np.round(freq_achieved, 2)} Hz")

    # Rayleigh阻尼
    zeta = 0.01
    omega = 2 * np.pi * target_freq[:2]
    alpha_r = 2 * zeta * omega[0] * omega[1] / (omega[0] + omega[1])
    beta_r = 2 * zeta / (omega[0] + omega[1])

    # 15种工况定义: (label, damage_description, stiffness_ratios)
    cases = [
        (0,  'Baseline',           [1.0, 1.0, 1.0, 1.0]),
        (1,  'Floor1 -10%',        [0.9, 1.0, 1.0, 1.0]),
        (2,  'Floor2 -10%',        [1.0, 0.9, 1.0, 1.0]),
        (3,  'Floor3 -10%',        [1.0, 1.0, 0.9, 1.0]),
        (4,  'Floor4 -10%',        [1.0, 1.0, 1.0, 0.9]),
        (5,  'Floor1 -25%',        [0.75, 1.0, 1.0, 1.0]),
        (6,  'Floor2 -25%',        [1.0, 0.75, 1.0, 1.0]),
        (7,  'Floor3 -25%',        [1.0, 1.0, 0.75, 1.0]),
        (8,  'Floor4 -25%',        [1.0, 1.0, 1.0, 0.75]),
        (9,  'Floor1 -50%',        [0.5, 1.0, 1.0, 1.0]),
        (10, 'Floor3 -50%',        [1.0, 1.0, 0.5, 1.0]),
        (11, 'Floor1,3 -20%',      [0.8, 1.0, 0.8, 1.0]),
        (12, 'Floor2,4 -15%',      [1.0, 0.85, 1.0, 0.85]),
        (13, 'All -5%',            [0.95, 0.95, 0.95, 0.95]),
        (14, 'Floor1-40% Floor2-20%', [0.6, 0.8, 1.0, 1.0]),
    ]
    n_repeats = 5
    t = np.arange(n_samples) / fs

    # 带通激励
    def gen_excitation():
        noise = np.random.randn(n_samples) * 0.05 * 9.81  # ~0.05g RMS
        b, a = butter(4, [0.5, 60], btype='band', fs=fs)
        return filtfilt(b, a, noise)

    all_accel = []
    all_labels = []
    all_damage = []

    for case_id, desc, ratios in cases:
        k_damaged = k_base * np.array(ratios)
        K_d = build_K(k_damaged)
        C_d = alpha_r * M + beta_r * K_d
        eigvals_d, phi_d = np.linalg.eigh(np.linalg.solve(M, K_d))
        omega_d = np.sqrt(np.abs(eigvals_d))

        for rep in range(n_repeats):
            excitation = gen_excitation()
            # 模态叠加: 对每个模态用传递函数求解
            accel = np.zeros((n_samples, n_floors))
            gamma = phi_d.T @ M @ np.ones(n_floors)  # 参与系数
            for mode in range(n_floors):
                wn = omega_d[mode]
                if wn < 1e-6:
                    continue
                modal_force = gamma[mode] * excitation
                # SDOF 传递函数: H(s) = 1/(s^2 + 2*zeta*wn*s + wn^2)
                # 位移响应 u, 加速度 = -2*zeta*wn*u_dot - wn^2*u
                # 使用 scipy.signal.lfilter (数值稳定)
                from scipy.signal import lfilter
                # 离散化: 双线性变换 (Tustin) 近似
                dt = 1.0 / fs
                c1 = 2.0 / dt
                a0 = c1**2 + 2*zeta*wn*c1 + wn**2
                b_tf = np.array([1.0/a0, 2.0/a0, 1.0/a0])
                a_tf = np.array([1.0,
                                 (2*wn**2 - 2*c1**2)/a0,
                                 (c1**2 - 2*zeta*wn*c1 + wn**2)/a0])
                u_resp = lfilter(b_tf, a_tf, modal_force)
                # 加速度 from 位移: a = f - 2*zeta*wn*v - wn^2*u
                # 近似 v ≈ diff(u)/dt, a ≈ diff(v)/dt
                modal_accel = np.gradient(np.gradient(u_resp, dt), dt)
                accel += np.outer(modal_accel, phi_d[:, mode])

            all_accel.append(accel.astype(np.float32))
            all_labels.append(case_id)
            all_damage.append(ratios)

    accel_arr = np.array(all_accel)  # (75, 15360, 4)
    labels_arr = np.array(all_labels, dtype=np.int32)
    damage_arr = np.array(all_damage, dtype=np.float32)

    os.makedirs(data_proc_dir, exist_ok=True)
    np.savez_compressed(proc_path,
                        accel=accel_arr, labels=labels_arr, damage=damage_arr,
                        fs=fs, k_base=k_base, freq_achieved=freq_achieved)
    print(f"  DS-2 synthesized: accel{accel_arr.shape}, {len(cases)} cases x {n_repeats} repeats")
    return {'accel': accel_arr, 'labels': labels_arr, 'damage': damage_arr}
