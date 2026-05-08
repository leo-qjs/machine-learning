from pathlib import Path
import numpy as np
import pandas as pd


# ====== 1. 配置路径 ======
in_path = Path("validation_prediction_file_level.csv")
out_path = Path("validation_prediction_file_level_11class.csv")

# 平滑参数
# 越小：概率更集中在已知类别
# 越大：越容易把概率扩散到 5、7、9、10
SIGMA = 0.75


# ====== 2. 读取文件级预测结果 ======
df = pd.read_csv(in_path)

print("==== input shape ====")
print(df.shape)

print("\n==== columns ====")
print(df.columns.tolist())


# ====== 3. 找到当前随机森林输出的已知类别概率列 ======
known_pairs = []

for col in df.columns:
    if col.startswith("proba_"):
        label = int(col.replace("proba_", ""))
        known_pairs.append((label, col))

known_pairs = sorted(known_pairs, key=lambda x: x[0])

known_labels = np.array([x[0] for x in known_pairs], dtype=float)
known_proba_cols = [x[1] for x in known_pairs]

print("\nknown_labels:", known_labels.tolist())
print("known_proba_cols:", known_proba_cols)


# ====== 4. 取出已知类别概率矩阵 ======
P_known = df[known_proba_cols].to_numpy(dtype=float)

# 防御：每行归一化，避免数值误差
P_known = P_known / np.maximum(P_known.sum(axis=1, keepdims=True), 1e-12)


# ====== 5. 构造 0~10 的目标类别 ======
target_labels = np.arange(0, 11, dtype=float)


# ====== 6. 构造平滑核矩阵 ======
# kernel[k_idx, j_idx] 表示：
# 已知类别 k 的概率，扩散到目标类别 j 的比例
kernel = []

for k in known_labels:
    weights = np.exp(-((target_labels - k) ** 2) / (2 * SIGMA ** 2))
    weights = weights / weights.sum()
    kernel.append(weights)

kernel = np.vstack(kernel)  # shape = [n_known_classes, 11]

print("\nkernel shape:", kernel.shape)


# ====== 7. 扩展为 11 类概率 ======
# P_known: [n_files, n_known_classes]
# kernel:  [n_known_classes, 11]
# Q:       [n_files, 11]
Q = P_known @ kernel

# 防御：再次归一化
Q = Q / np.maximum(Q.sum(axis=1, keepdims=True), 1e-12)


# ====== 8. 写入 prob_0 ~ prob_10 ======
for idx, label in enumerate(range(11)):
    df[f"prob_{label}"] = Q[:, idx]


# ====== 9. 得到 11 类预测结果 ======
pred_11 = Q.argmax(axis=1)
confidence_11 = Q.max(axis=1)

df["pred_label_11"] = pred_11
df["confidence_11"] = confidence_11


# ====== 10. 可选：连续严重度期望 ======
severity_score = Q @ target_labels
df["severity_score_11"] = severity_score


# ====== 11. 保存结果 ======
df.to_csv(out_path, index=False)

print("\n==== output shape ====")
print(df.shape)

print("\n==== pred_label_11 count ====")
print(df["pred_label_11"].value_counts().sort_index())

print("\n==== head ====")
show_cols = [
    "source_id",
    "pred_label",
    "confidence",
    "pred_label_11",
    "confidence_11",
    "severity_score_11",
]
print(df[show_cols].head())

print(f"\nSaved: {out_path}")