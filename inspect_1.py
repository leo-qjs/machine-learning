import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

feature_path = "mini_features.csv"

df = pd.read_csv(feature_path)

print("==== shape ====")
print(df.shape)

print("\n==== columns ====")
print(df.columns.tolist())

print("\n==== label counts ====")
print(df["label"].value_counts().sort_index())

# 不参与训练的元信息列
meta_cols = [
    "label",
    "label_name",
    "source_file",
    "start",
    "end",
    "window_index",
]

feature_cols = [c for c in df.columns if c not in meta_cols]

print("\n==== feature columns count ====")
print(len(feature_cols))

# 检查 NaN / inf
print("\n==== NaN count ====")
print(df[feature_cols].isna().sum().sort_values(ascending=False).head(20))

inf_count = np.isinf(df[feature_cols].to_numpy()).sum()
print("\n==== inf count ====")
print(inf_count)

# 检查常数列
constant_cols = []
for c in feature_cols:
    if df[c].nunique(dropna=False) <= 1:
        constant_cols.append(c)

print("\n==== constant columns ====")
print(constant_cols)

# 按 label 看均值
group_mean = df.groupby("label")[feature_cols].mean()

print("\n==== group mean head ====")
print(group_mean.head())

# 保存均值表
group_mean.to_csv("feature_mean_by_label.csv")
print("\nSaved: feature_mean_by_label.csv")

# 画几个核心特征
plot_features = [
    "ch0_rms",
    "ch0_std",
    "ch0_kurt",
    "ch0_ptp",
    "ch1_rms",
    "ch2_rms",
]

for feat in plot_features:
    if feat not in df.columns:
        print(f"[skip] {feat} not found")
        continue

    plt.figure(figsize=(8, 4))
    labels = sorted(df["label"].unique())

    data = [df[df["label"] == lab][feat].values for lab in labels]

    plt.boxplot(data, labels=labels)
    plt.xlabel("label")
    plt.ylabel(feat)
    plt.title(f"{feat} distribution by label")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()