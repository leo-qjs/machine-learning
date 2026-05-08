import pandas as pd
import numpy as np
from pathlib import Path


# ====== 配置 ======
feature_path = "features_all_windows.csv"
out_path = "features_all_windows_with_split.csv"

VAL_RATIO = 0.2
RANDOM_STATE = 42

df = pd.read_csv(feature_path)

print("==== original shape ====")
print(df.shape)

print("\n==== columns ====")
print(df.columns.tolist())

# 优先使用 source_id；没有的话用 source_path
if "source_id" in df.columns:
    group_col = "source_id"
elif "source_path" in df.columns:
    group_col = "source_path"
else:
    raise ValueError("CSV must contain source_id or source_path to avoid leakage.")

if "label" not in df.columns:
    raise ValueError("CSV must contain label column.")

rng = np.random.default_rng(RANDOM_STATE)

val_sources = set()
train_sources = set()

# ====== 按 label 分层，再按 source 分组切分 ======
for label, label_df in df.groupby("label"):
    sources = sorted(label_df[group_col].unique())
    n_sources = len(sources)

    print(f"\nlabel={label}, source count={n_sources}")

    if n_sources < 2:
        # 只有一个txt时，无法做文件级验证
        # 这种label只能放训练集，否则验证集/训练集会缺类
        print(f"[WARN] label={label} has less than 2 sources, all assigned to train.")
        train_sources.update(sources)
        continue

    n_val = int(round(n_sources * VAL_RATIO))

    # 至少给验证集1个source，但也不能把该label全拿走
    n_val = max(1, n_val)
    n_val = min(n_val, n_sources - 1)

    sources_arr = np.array(sources)
    rng.shuffle(sources_arr)

    label_val_sources = set(sources_arr[:n_val])
    label_train_sources = set(sources_arr[n_val:])

    val_sources.update(label_val_sources)
    train_sources.update(label_train_sources)

    print(f"  train sources={len(label_train_sources)}, val sources={len(label_val_sources)}")


# ====== 写入 split 列 ======
df["split"] = "unknown"
df.loc[df[group_col].isin(train_sources), "split"] = "train"
df.loc[df[group_col].isin(val_sources), "split"] = "val"

# 检查是否还有 unknown
unknown_count = (df["split"] == "unknown").sum()
if unknown_count != 0:
    raise RuntimeError(f"There are {unknown_count} rows with unknown split.")

# ====== 泄漏检查：同一个 source 不能同时出现在 train 和 val ======
source_split_count = df.groupby(group_col)["split"].nunique()
leak_sources = source_split_count[source_split_count > 1]

if len(leak_sources) > 0:
    print("\n[ERROR] leakage detected:")
    print(leak_sources)
    raise RuntimeError("Same source appears in both train and val.")
else:
    print("\nLeakage check passed: no source appears in both train and val.")


# ====== 查看切分结果 ======
print("\n==== row count by split ====")
print(df["split"].value_counts())

print("\n==== label count by split ====")
print(pd.crosstab(df["label"], df["split"]))

print("\n==== source count by split ====")
print(df.groupby("split")[group_col].nunique())

# 保存
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")