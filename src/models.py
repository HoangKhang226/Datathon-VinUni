"""
src/models.py
=============
Định nghĩa feature set, target set, và hàm inference.

Kiến trúc Stacking:
  Level 0 — Base models (tree + linear để tăng diversity):
    - LightGBM   : gradient boosting mạnh, xử lý tốt time-series lag features
    - XGBoost    : gradient boosting, bias/structure khác LGB
    - Ridge      : linear baseline — "bias stabilizer", bù đắp khi tree overfit phần tuyến tính

  Level 1 — Meta model (Ridge):
    Học cách kết hợp (blend) dự đoán từ Level 0 dựa trên OOF predictions.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# ── Feature & Target definitions ─────────────────────────────────────────────

FEATURES: list[str] = [
    # Đặc trưng thời gian (calendar)
    "year", "month", "day", "day_of_week", "day_of_year",
    "week_of_year", "quarter",
    "is_weekend", "is_month_end", "is_month_start",
    "is_year_end", "is_year_start",
    "sin_month", "cos_month", "sin_dow", "cos_dow",
    "days_to_tet",

    # Biến lag — ghi nhớ giá trị quá khứ, quan trọng bậc nhất
    "revenue_lag_1", "revenue_lag_2", "revenue_lag_6",
    "revenue_lag_7", "revenue_lag_14", "revenue_lag_30",
    "revenue_lag_90", "revenue_lag_365",
    "cogs_lag_1", "cogs_lag_7", "cogs_lag_30", "cogs_lag_365",

    # Rolling window — làm mượt xu hướng ngắn hạn
    "revenue_roll_mean_7",  "revenue_roll_std_7",
    "revenue_roll_mean_14", "revenue_roll_std_14",
    "revenue_roll_mean_30", "revenue_roll_std_30",
    "revenue_roll_mean_90", "revenue_roll_std_90",
    "revenue_ewm_7", "revenue_ewm_30",

    # Xu hướng & biến động
    "revenue_diff_1", "revenue_diff_7",
    "revenue_pct_change_7",
    "cogs_roll_mean_7", "cogs_roll_mean_30",
]

TARGETS: list[str] = ["Revenue", "COGS"]


# ── Model factory ─────────────────────────────────────────────────────────────

def build_lgb(params: dict) -> LGBMRegressor:
    """Tạo LGBMRegressor với params đã tune. verbose=-1 để tắt log của LGB."""
    return LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)


def build_xgb(params: dict) -> XGBRegressor:
    """Tạo XGBRegressor với params đã tune. tree_method='hist' cho tốc độ."""
    return XGBRegressor(**params, random_state=42, n_jobs=-1, tree_method="hist")


def build_ridge(alpha: float = 100.0) -> Ridge:
    """
    Tạo Ridge regression (Level 0 base model).
    Alpha cao giúp tránh overfitting trên scaled features.
    """
    return Ridge(alpha=alpha)


def build_meta() -> Ridge:
    """Meta model (Level 1) — Ridge nhẹ để blend 3 predictions từ Level 0."""
    return Ridge(alpha=1.0)


# ── Inference ─────────────────────────────────────────────────────────────────

def stack_predict(
    X,
    lgb_m:   LGBMRegressor,
    xgb_m:   XGBRegressor,
    ridge_m: Ridge,
    meta_m:  Ridge,
    scaler:  StandardScaler,
) -> np.ndarray:
    """
    Stacking inference: tạo predictions từ 3 base models → đưa vào meta model.

    Args:
        X       : Feature DataFrame (chưa scale).
        lgb_m   : LightGBM đã train.
        xgb_m   : XGBoost đã train.
        ridge_m : Ridge đã train (cần scaled features).
        meta_m  : Meta Ridge đã train.
        scaler  : Scaler đã fit trên training set.

    Returns:
        ndarray dự báo cuối cùng.
    """
    X_sc    = scaler.transform(X)
    meta_in = np.column_stack([
        lgb_m.predict(X),
        xgb_m.predict(X),
        ridge_m.predict(X_sc),
    ])
    return meta_m.predict(meta_in)
