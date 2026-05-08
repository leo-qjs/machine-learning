import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# feature_path = "features_all_windows_with_split.csv"

df = pd.read_csv("features_all_windows_with_split.csv")

print("==== data shape ====")
print(df.shape)

print("\n==== label counts ====")
print(df["label"].value_counts().sort_index())


# 这些列不是训练特征
meta_cols = [
    "label",
    "label_name",
    "source_file",
    "start",
    "end",
    "source_path",
    "source_id",
    "window_index",
    "split",
]

feature_cols = [c for c in df.columns if c not in meta_cols]

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()

X_train = train_df[feature_cols]
y_train = train_df["label"]

X_val = val_df[feature_cols]
y_val = val_df["label"]


print("\n==== feature columns ====")
print(len(feature_cols))
print(feature_cols)


# 先用随机切分验证流程
# 注意：当前只是 mini 数据，不代表真实泛化能力
# X_train, X_val, y_train, y_val = train_test_split(
#     X,
#     y,
#     test_size=0.3,                                  #30%标签作为验证集
#     random_state=42,
#     stratify=y,              #保证训练集和验证集里lable比例一致
# )

print("\n==== train/val shape ====")
print("train:", X_train.shape,y_train.shape)
print("val:", X_val.shape,y_val.shape)


models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
        )),
    ]),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    ),
}
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

for name, model in models.items():
    print("\n" + "=" * 80)
    print(f"MODEL: {name}")
    print("=" * 80)

    model.fit(X_train, y_train)

    pred = model.predict(X_val)

    acc = accuracy_score(y_val, pred)
    macro_f1 = f1_score(y_val, pred, average="macro")

    print("accuracy:", acc)
    print("macro_f1:", macro_f1)

    print("\nclassification report:")
    print(classification_report(y_val, pred, digits=4))

    print("\nconfusion matrix:")
    labels = sorted(y_train.unique())
    print("labels:", labels)
    print(confusion_matrix(y_val, pred, labels=labels))

    # 保存训练好的模型
    model_path = model_dir / f"{name}.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved model: {model_path}")

# 保存元信息：特征列顺序、标签列表
metadata = {
    "feature_cols": feature_cols,
    "labels": sorted(y_train.unique().tolist()),
}

with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print("\nSaved metadata: models/metadata.json")