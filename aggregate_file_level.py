from pathlib import Path
import pandas as pd


# ====== 1. 配置路径 ======
window_pred_path = Path("validation_prediction_window_level.csv")
out_file_path = Path("validation_prediction_file_level.csv")


# ====== 2. 读取窗口级预测结果 ======
df = pd.read_csv(window_pred_path)

print("==== window-level prediction shape ====")
print(df.shape)

print("\n==== columns ====")
print(df.columns.tolist())


# ====== 3. 检查必要列 ======
if "source_id" not in df.columns:
    raise ValueError("window prediction csv must contain source_id column.")

if "pred_label" not in df.columns:
    raise ValueError("window prediction csv must contain pred_label column.")


# 找到概率列，例如 proba_0, proba_1, proba_2...
proba_cols = [c for c in df.columns if c.startswith("proba_")]

print("\n==== proba columns ====")
print(proba_cols)


file_rows = []


# ====== 4. 按 txt 文件汇总 ======
for source_id, g in df.groupby("source_id"):
    row = {
        "source_id": source_id,
        "n_windows": len(g),
    }

    # 尽量保留 source_file/source_path，方便查看
    if "source_file" in g.columns:
        row["source_file"] = g["source_file"].iloc[0]

    if "source_path" in g.columns:
        row["source_path"] = g["source_path"].iloc[0]

    # ====== 优先使用平均概率 ======
    if len(proba_cols) > 0:
        mean_proba = g[proba_cols].mean()

        # 保存每个类别的平均概率
        for c in proba_cols:
            row[c] = mean_proba[c]

        # 找平均概率最大的类别
        best_col = mean_proba.idxmax()
        pred_label = int(best_col.replace("proba_", ""))

        row["pred_label"] = pred_label
        row["confidence"] = float(mean_proba[best_col])
        row["method"] = "mean_proba"

    # ====== 如果没有概率列，则退化为多数投票 ======
    else:
        vote_counts = g["pred_label"].value_counts()

        pred_label = int(vote_counts.idxmax())
        confidence = vote_counts.max() / len(g)

        row["pred_label"] = pred_label
        row["confidence"] = float(confidence)
        row["method"] = "majority_vote"

        # 保存投票统计，方便排查
        row["vote_detail"] = vote_counts.to_dict()

    file_rows.append(row)


# ====== 5. 保存文件级预测结果 ======
file_df = pd.DataFrame(file_rows)

print("\n==== file-level prediction shape ====")
print(file_df.shape)

print("\n==== prediction count ====")
print(file_df["pred_label"].value_counts().sort_index())

print("\n==== head ====")
print(file_df.head())

file_df.to_csv(out_file_path, index=False)
print(f"\nSaved: {out_file_path}")