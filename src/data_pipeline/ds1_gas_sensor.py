"""DS-1 local UCI Gas Sensor Array Drift Dataset parser."""
import os
import zipfile

import numpy as np


def parse_libsvm_line(line):
    parts = line.strip().split()
    if len(parts) < 2:
        return None, None
    label = int(parts[0].split(";")[0]) if ";" in parts[0] else int(parts[0])
    features = np.zeros(128)
    for p in parts[1:]:
        if ":" in p:
            idx_s, val_s = p.split(":")
            idx = int(idx_s) - 1
            if 0 <= idx < 128:
                features[idx] = float(val_s)
    return label, features


def load_ds1(data_raw_dir, data_proc_dir):
    proc_path = os.path.join(data_proc_dir, "ds1_gas.npz")
    if os.path.exists(proc_path):
        print("  DS-1 processed file exists, loading...")
        d = np.load(proc_path)
        return {"X": d["X"], "y": d["y"], "batch": d["batch"], "X_full": d["X_full"]}

    cache_dir = os.path.join(data_raw_dir, "gas_sensor")
    zip_path = os.path.join(cache_dir, "gas_sensor.zip")
    extract_dir = os.path.join(cache_dir, "extracted")

    if not os.path.exists(extract_dir) and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    elif not os.path.exists(extract_dir):
        raise FileNotFoundError(
            f"Neither extracted DS-1 files nor local archive were found under {cache_dir}."
        )

    batch_files = []
    for root, _, files in os.walk(extract_dir):
        for f in sorted(files):
            if f.lower().startswith("batch") and f.endswith(".dat"):
                batch_files.append(os.path.join(root, f))

    if not batch_files:
        raise FileNotFoundError(f"No batch*.dat files found in {extract_dir}")

    all_labels, all_features, all_batches = [], [], []
    for bi, bf in enumerate(batch_files):
        batch_id = bi + 1
        with open(bf, "r") as f:
            for line in f:
                label, features = parse_libsvm_line(line)
                if label is not None:
                    all_labels.append(label)
                    all_features.append(features)
                    all_batches.append(batch_id)

    X_full = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    batch = np.array(all_batches, dtype=np.int32)

    # Use one steady-state response feature per gas sensor channel.
    X = X_full[:, [s * 8 for s in range(16)]]

    os.makedirs(data_proc_dir, exist_ok=True)
    np.savez_compressed(proc_path, X=X, y=y, batch=batch, X_full=X_full)
    print(f"  DS-1 loaded: X{X.shape}, y{y.shape}, batches {np.unique(batch)}")
    return {"X": X, "y": y, "batch": batch, "X_full": X_full}
