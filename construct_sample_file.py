from pathlib import Path
import re
import numpy as np
import pandas as pd


# ====== 1. 配置你的训练集路径 ======
root = Path(r"D:\workspace\proj\mlt\machine-learning\dataset\Data_Challenge_PHM2023_training_data\Data_Challenge_PHM2023_training_data")

# 先不要太大，先把逻辑跑通
WINDOW_SIZE = 4096
STRIDE = 4096
MAX_WINDOWS_PER_FILE = 20   # 每个txt最多取20个窗口，先别爆数据量


def parse_label_from_folder(folder_name: str) -> int:
    """
    从文件夹名中提取 label。
    例如：
    Pitting_degradation_level_0 (Healthy) -> 0
    Pitting_degradation_level_3 -> 3
    """
    m = re.search(r"level_(\d+)", folder_name)
    if m is None:
        raise ValueError(f"Cannot parse label from folder name: {folder_name}")
    return int(m.group(1))


def read_signal_txt(path: Path) -> np.ndarray:
    """
    读取空格分隔的txt信号文件。
    返回 shape = [采样点数, 通道数] 的 numpy 数组。
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    data = df.to_numpy(dtype=float)
    return data


def extract_window_features(window: np.ndarray, prefix: str = "") -> dict:
    """
    对一个窗口提取基础统计特征。
    window shape = [window_size, n_channels]
    """
    features = {}
    eps = 1e-12

    n_channels = window.shape[1]

    for ch in range(n_channels):
        x = window[:, ch]

        mean = np.mean(x)
        std = np.std(x)
        rms = np.sqrt(np.mean(x ** 2))
        min_v = np.min(x)
        max_v = np.max(x)
        ptp = max_v - min_v
        abs_mean = np.mean(np.abs(x))

        centered = x - mean
        skew = np.mean(centered ** 3) / ((std + eps) ** 3)
        kurt = np.mean(centered ** 4) / ((std + eps) ** 4)

        features[f"{prefix}ch{ch}_mean"] = mean
        features[f"{prefix}ch{ch}_std"] = std
        features[f"{prefix}ch{ch}_rms"] = rms
        features[f"{prefix}ch{ch}_min"] = min_v
        features[f"{prefix}ch{ch}_max"] = max_v
        features[f"{prefix}ch{ch}_ptp"] = ptp
        features[f"{prefix}ch{ch}_abs_mean"] = abs_mean
        features[f"{prefix}ch{ch}_skew"] = skew
        features[f"{prefix}ch{ch}_kurt"] = kurt

    return features


def cut_windows(data: np.ndarray, window_size: int, stride: int, max_windows: int | None = None):
    """
    从长信号中切窗口。
    """
    windows = []

    n = data.shape[0]
    start = 0

    while start + window_size <= n:
        end = start + window_size
        windows.append((start, end, data[start:end, :]))

        if max_windows is not None and len(windows) >= max_windows:
            break

        start += stride

    return windows


rows = []

# ====== 2. 每个标签目录抽一个 txt ======
for label_dir in sorted(root.iterdir()):
    if not label_dir.is_dir():
        continue

    label_name = label_dir.name
    label = parse_label_from_folder(label_name)

    txt_files = sorted(label_dir.glob("*.txt"))

    if len(txt_files) == 0:
        print(f"[WARN] no txt file in {label_dir}")
        continue

    # 每个标签只抽第一个txt，先跑通逻辑
    selected_file = txt_files[0]

    print(f"\n[LOAD] label={label}, folder={label_name}")
    print(f"       file={selected_file.name}")

    data = read_signal_txt(selected_file)

    print(f"       raw shape = {data.shape}")

    windows = cut_windows(
        data,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        max_windows=MAX_WINDOWS_PER_FILE
    )

    print(f"       windows = {len(windows)}")

    for win_idx, (start, end, win) in enumerate(windows):
        feat = extract_window_features(win)

        feat["label"] = label
        feat["label_name"] = label_name
        feat["source_file"] = selected_file.name
        feat["start"] = start
        feat["end"] = end
        feat["window_index"] = win_idx

        rows.append(feat)


# ====== 3. 生成机器学习样本表 ======
feature_df = pd.DataFrame(rows)

print("\n==== feature_df shape ====")
print(feature_df.shape)

print("\n==== head ====")
print(feature_df.head())

print("\n==== label counts ====")
print(feature_df["label"].value_counts().sort_index())

print("\n==== columns ====")
print(feature_df.columns.tolist())

feature_df.to_csv("mini_features.csv", index=False)
print("\nSaved: mini_features.csv")