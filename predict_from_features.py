import json
import joblib
import pandas as pd


# ====== 1. 配置路径 ======
feature_path = "validation_features.csv"
model_path = "models/random_forest.joblib"
metadata_path = "models/metadata.json"
out_path = "validation_prediction_window_level.csv"


# ====== 2. 加载模型和元信息 ======
model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

feature_cols = metadata["feature_cols"]
labels = metadata["labels"]

print("loaded model:", model_path)
print("feature count:", len(feature_cols))
print("labels:", labels)


# ====== 3. 读取特征表 ======
df = pd.read_csv(feature_path)

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"validation_features.csv missing feature columns: {missing_cols}")

X = df[feature_cols]

print("X_Valid:", X.shape)
# ====== 4. 推理 ======
pred = model.predict(X)

print("\n==== prediction head ====")
print(pred[:20])

df["pred_label"] = pred


# 如果模型支持 predict_proba，则输出概率
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)

    print("\n==== probability head ====")
    print(proba[:5])

    pred_confidence = proba.max(axis=1)
    df["confidence"] = pred_confidence

    class_labels = model.classes_

    for idx, cls in enumerate(class_labels):
        df[f"proba_{cls}"] = proba[:, idx]

else:
    df["confidence"] = None


# ====== 5. 保存窗口级预测结果 ======
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")


# ====== 6. 如果原始表里有真实 label，则简单对比 ======
if "label" in df.columns:
    acc = (df["label"] == df["pred_label"]).mean()
    print("\nfull-table accuracy:", acc)
else:
    print("\nNo label column found. Skip accuracy evaluation.")