import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)

EVAL_THRESHOLD = 0.70

# Đảm bảo luôn ghi vào sqlite file cục bộ dù env var có được set hay không
if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
if not os.environ.get("MLFLOW_ARTIFACT_ROOT"):
    os.environ["MLFLOW_ARTIFACT_ROOT"] = "./mlartifacts"


def check_label_distribution(y, split_name="train"):
    """Bonus 5: Cảnh báo nếu bất kỳ lớp nào chiếm < 10% tổng mẫu."""
    total = len(y)
    dist = {}
    for cls in [0, 1, 2]:
        ratio = (y == cls).sum() / total
        dist[cls] = round(ratio, 4)
        if ratio < 0.10:
            print(f"[WARNING] {split_name}: lớp {cls} chỉ chiếm {ratio:.1%} < 10% — có thể mất cân bằng dữ liệu!")
    print(f"[INFO] Phân phối nhãn ({split_name}): {dist}")
    return dist


def get_model(params: dict):
    """Bonus 2: Chọn thuật toán dựa trên tham số model_type."""
    model_type = params.pop("model_type", "random_forest")
    if model_type == "gradient_boosting":
        return model_type, GradientBoostingClassifier(**params, random_state=42)
    elif model_type == "logistic_regression":
        lr_params = {k: v for k, v in params.items() if k in ["max_iter", "C"]}
        return model_type, LogisticRegression(**lr_params, random_state=42, max_iter=params.get("max_iter", 1000))
    else:
        return "random_forest", RandomForestClassifier(**params, random_state=42)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Kiểm tra phân phối nhãn
    label_dist = check_label_distribution(y_train, "train")

    # Bonus 2: Chọn model theo model_type
    params_copy = params.copy()
    model_type, model = get_model(params_copy)

    with mlflow.start_run():
        mlflow.log_params({**params, "model_type": model_type})

        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")
        precision = precision_score(y_eval, preds, average="weighted", zero_division=0)
        recall = recall_score(y_eval, preds, average="weighted", zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        os.makedirs("outputs", exist_ok=True)

        # Bonus 5: Ghi phân phối nhãn vào metrics.json
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "label_distribution": label_dist,
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Bonus 3: Tạo báo cáo hiệu suất chi tiết
        cm = confusion_matrix(y_eval, preds)
        precision_per_class = precision_score(y_eval, preds, average=None, zero_division=0)
        recall_per_class = recall_score(y_eval, preds, average=None, zero_division=0)

        report_lines = [
            f"Model type   : {model_type}",
            f"Params       : {params}",
            f"Accuracy     : {acc:.4f}",
            f"F1 (weighted): {f1:.4f}",
            "",
            "Per-class Precision & Recall:",
            f"  Lớp 0 (thấp)      - Precision: {precision_per_class[0]:.4f} | Recall: {recall_per_class[0]:.4f}",
            f"  Lớp 1 (trung bình)- Precision: {precision_per_class[1]:.4f} | Recall: {recall_per_class[1]:.4f}",
            f"  Lớp 2 (cao)       - Precision: {precision_per_class[2]:.4f} | Recall: {recall_per_class[2]:.4f}",
            "",
            "Confusion Matrix:",
            "  (rows=actual, cols=predicted)",
        ]
        for row in cm:
            report_lines.append("  " + "  ".join(f"{v:4d}" for v in row))

        report_lines += [
            "",
            "Label Distribution (train):",
        ]
        for cls, ratio in label_dist.items():
            warn = " [WARNING < 10%]" if ratio < 0.10 else ""
            report_lines.append(f"  Lớp {cls}: {ratio:.1%}{warn}")

        report = "\n".join(report_lines)
        print(report)
        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
