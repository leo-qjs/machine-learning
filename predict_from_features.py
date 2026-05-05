import json
import joblib
import pandas as pd


# ====== 1. 配置路径 ======
feature_path = "mini_features.csv"
model_path = "models/logistic_regression.joblib"
metadata_path = "models/metadata.json"


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

X = df[feature_cols]


# ====== 4. 推理 ======
pred = model.predict(X)

print("\n==== prediction head ====")
print(pred[:20])


# 如果模型支持 predict_proba，则输出概率
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)

    print("\n==== probability head ====")
    print(proba[:5])

    pred_confidence = proba.max(axis=1)

    df["pred_label"] = pred
    df["confidence"] = pred_confidence

else:
    df["pred_label"] = pred
    df["confidence"] = None


# ====== 5. 保存预测结果 ======
df.to_csv("mini_prediction_result.csv", index=False)
print("\nSaved: mini_prediction_result.csv")


# ====== 6. 如果原始表里有真实 label，则简单对比 ======
if "label" in df.columns:
    acc = (df["label"] == df["pred_label"]).mean()
    print("\nfull-table accuracy:", acc)