"""
src/utils.py
============
Các tiện ích dùng chung trong pipeline:
  - setup_logging : khởi tạo logger ghi ra console + file
  - evaluate       : tính MAE / RMSE / R² và in log
  - log_shap_summary : tính SHAP values và log plot lên MLflow
"""

from __future__ import annotations

import logging
from pathlib import Path


# --------- Logging ---------


def setup_logging(log_file: str = "logs/train.log") -> logging.Logger:
    """
    Khởi tạo logging ghi đồng thời ra console và file.
    Gọi hàm này một lần duy nhất ở đầu chương trình.

    Args:
        log_file: Đường dẫn file log (sẽ tự tạo thư mục nếu chưa có).

    Returns:
        Logger instance (__main__).
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )
    log = logging.getLogger(__name__)
    log.info(f"Logging initialised → {log_path.resolve()}")
    return log


# --------- Evaluation ---------


def evaluate(y_true, y_pred, label: str = "", log: logging.Logger | None = None) -> dict:
    """
    Tính MAE, RMSE, R² và in ra log.

    Args:
        y_true  : Giá trị thực (Series / ndarray).
        y_pred  : Giá trị dự báo (ndarray).
        label   : Tên model hiển thị trong log.
        log     : Logger; nếu None thì dùng print.

    Returns:
        dict với keys: mae, rmse, r2.
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)

    msg = f"{label:<14s}  MAE={mae:>12,.0f}  RMSE={rmse:>12,.0f}  R²={r2:.4f}"
    if log:
        log.info(msg)

    return dict(mae=mae, rmse=rmse, r2=float(r2))


# --------- SHAP Interpretation ---------


def log_shap_summary(
    model,
    X,
    target_name: str,
    model_name: str,
    mlflow=None,
    log: logging.Logger | None = None,
) -> str:
    """
    Tính SHAP values bằng TreeExplainer và lưu / log summary plot.

    Args:
        model       : Mô hình đã train (LGBM / XGB).
        X           : Feature DataFrame (dùng 500 dòng cuối làm mẫu).
        target_name : Tên target (Revenue / COGS).
        model_name  : Tên model hiển thị (LGBM / XGB).
        mlflow      : MLflow module; nếu None thì không log artifact.
        log         : Logger.

    Returns:
        Đường dẫn đến file PNG đã lưu.
    """
    import shap
    import matplotlib.pyplot as plt

    _log = log.info if log else print
    _log(f"[SHAP] Explaining {model_name} for {target_name}...")

    # TreeExplainer tối ưu cho Decision Tree–based models
    explainer   = shap.TreeExplainer(model)
    X_sample    = X.tail(500)           # 500 dòng cuối đại diện xu hướng gần nhất
    shap_values = explainer.shap_values(X_sample)

    # Summary Plot (dot) — hiển thị hướng và độ lớn đóng góp của từng feature
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP Summary: {target_name} ({model_name})")
    plt.tight_layout()

    plot_path = f"shap_summary_{target_name}_{model_name}.png"
    plt.savefig(plot_path, dpi=150)

    if mlflow is not None:
        mlflow.log_artifact(plot_path, artifact_path="plots")

    _log(f"[SHAP] {plot_path} saved{' and logged to MLflow' if mlflow else ''}")
    return plot_path
