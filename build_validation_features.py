from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# 1. 配置路径和窗口参数
# =========================

validation_root = Path(
    r"E:\workspace\WorkingProject\mlt\machine-learning\dateset\Data_Challenge_PHM2023_validation_data"
)

OUT_CSV = "validation_features.csv"

# 必须和训练集保持一致
WINDOW_SIZE = 4096
STRIDE = 4096


# =========================
# 2. 读取原始 txt 信号
# =========================

def read_signal_txt(path: Path) -> np.ndarray:
    """
    读取空格分隔的 txt 信号文件。
    返回 shape = [采样点数, 通道数] 的 numpy 数组。
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    data = df.to_numpy(dtype=float)
    return data


# =========================
# 3. 切窗口
# =========================

def cut_windows(data: np.ndarray, window_size: int, stride: int):
    """
    从长信号中切完整窗口。

    返回：
    [
        (start, end, window_data),
        ...
    ]

    window_data.shape = [window_size, n_channels]
    """
    windows = []

    n = data.shape[0]
    start = 0

    while start + window_size <= n:
        end = start + window_size
        windows.append((start, end, data[start:end, :]))
        start += stride

    return windows


# =========================
# 4. 提取窗口统计特征
# =========================

def extract_window_features(window: np.ndarray) -> dict:
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

        features[f"ch{ch}_mean"] = mean
        features[f"ch{ch}_std"] = std
        features[f"ch{ch}_rms"] = rms
        features[f"ch{ch}_min"] = min_v
        features[f"ch{ch}_max"] = max_v
        features[f"ch{ch}_ptp"] = ptp
        features[f"ch{ch}_abs_mean"] = abs_mean
        features[f"ch{ch}_skew"] = skew
        features[f"ch{ch}_kurt"] = kurt

    return features


# =========================
# 5. 构造 validation_features.csv
# =========================

rows = []

txt_files = sorted(validation_root.rglob("*.txt"))

print("==== validation txt count ====")
print(len(txt_files))

if len(txt_files) == 0:
    raise RuntimeError(f"No txt files found in: {validation_root}")


for selected_file in txt_files:
    print(f"\n[LOAD VALIDATION] {selected_file.relative_to(validation_root)}")

    data = read_signal_txt(selected_file)

    print(f"    raw shape = {data.shape}")

    windows = cut_windows(
        data,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    print(f"    windows = {len(windows)}")

    source_path = str(selected_file.relative_to(validation_root))
    source_id = source_path

    for win_idx, (start, end, win) in enumerate(windows):
        feat = extract_window_features(win)

        # 注意：validation 没有 label
        feat["split"] = "validation"
        feat["source_file"] = selected_file.name
        feat["source_path"] = source_path
        feat["source_id"] = source_id
        feat["start"] = start
        feat["end"] = end
        feat["window_index"] = win_idx

        rows.append(feat)


val_df = pd.DataFrame(rows)

print("\n==== validation feature shape ====")
print(val_df.shape)

print("\n==== windows per source ====")
print(val_df.groupby("source_id").size().describe())

print("\n==== columns ====")
print(val_df.columns.tolist())

val_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")