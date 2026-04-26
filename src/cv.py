"""
src/cv.py
=========
Cross-validation và Hyperparameter Tuning cho Stacking Ensemble.

Module này thực hiện:
  1. objective_lgb / objective_xgb:
       Hàm mục tiêu Optuna — dùng 3-fold TimeSeriesSplit nội bộ trên
       tập Train để tính CV MAE. *Không* dùng Val set → không leakage.

  2. run_tuning:
       Khởi tạo Optuna study và chạy n_trials với TPESampler.

  3. generate_oof:
       Sau khi đã tìm được best_params, sinh Out-of-Fold predictions
       cho cả 3 model (LGB, XGB, Ridge) dùng 5-fold TimeSeriesSplit.
       Scaler được fit riêng trong từng fold (fold-aware scaling).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
import mlflow

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

log = logging.getLogger(__name__)


# ── Optuna Objectives ─────────────────────────────────────────────────────────

def objective_lgb(trial, X, y, parent_run_id: str) -> float:
    """
    Hàm mục tiêu cho Optuna (LightGBM).
    Sử dụng TimeSeriesSplit nội bộ trên tập Train để tính CV MAE.
    Mỗi trial được log vào MLflow dưới dạng nested run.
    """
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 300, 1500),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 20, 255),
        max_depth         = trial.suggest_int("max_depth", 3, 12),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )

    # 3-fold CV — đủ để estimate tốt mà không quá chậm
    tscv   = TimeSeriesSplit(n_splits=3)
    scores = []
    for tr_idx, val_idx in tscv.split(X):
        m = LGBMRegressor(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = m.predict(X.iloc[val_idx])
        scores.append(mean_absolute_error(y.iloc[val_idx], preds))

    cv_mae = float(np.mean(scores))

    # Log từng trial vào MLflow dưới dạng nested run
    with mlflow.start_run(run_name=f"lgb_trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("cv_mae", cv_mae)

    return cv_mae


def objective_xgb(trial, X, y, parent_run_id: str) -> float:
    """
    Hàm mục tiêu cho Optuna (XGBoost).
    """
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 300, 1500),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth        = trial.suggest_int("max_depth", 3, 10),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        random_state     = 42,
        n_jobs           = -1,
        tree_method      = "hist",
    )

    tscv   = TimeSeriesSplit(n_splits=3)
    scores = []
    for tr_idx, val_idx in tscv.split(X):
        m = XGBRegressor(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = m.predict(X.iloc[val_idx])
        scores.append(mean_absolute_error(y.iloc[val_idx], preds))

    cv_mae = float(np.mean(scores))

    with mlflow.start_run(run_name=f"xgb_trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("cv_mae", cv_mae)

    return cv_mae


# ── Tuning Helper ──────────────────────────────────────────────────────────────

def run_tuning(
    objective_fn,
    X,
    y,
    name: str,
    n_trials: int = 30,
    run_id: str = "",
) -> tuple[dict, float]:
    """
    Khởi tạo và chạy một Optuna study.

    Args:
        objective_fn : objective_lgb hoặc objective_xgb.
        X, y         : Feature matrix và target.
        name         : Tên human-readable cho log.
        n_trials     : Số lần thử.
        run_id       : MLflow parent run id (để nested run).

    Returns:
        (best_params, best_cv_mae)
    """
    log.info(f"[Optuna] Tuning {name} — {n_trials} trials")
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(lambda t: objective_fn(t, X, y, run_id), n_trials=n_trials)
    log.info(f"[{name}] Best CV MAE: {study.best_value:,.2f}")
    return study.best_params, study.best_value


# ── OOF Generation ─────────────────────────────────────────────────────────────

def generate_oof(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    best_lgb_rev:  dict,
    best_xgb_rev:  dict,
    best_lgb_cogs: dict,
    best_xgb_cogs: dict,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sinh Out-of-Fold predictions dùng TimeSeriesSplit 5-fold.

    Fold-aware scaling: StandardScaler được fit riêng trên training fold,
    sau đó transform val fold → không bị leakage từ tương lai vào quá khứ.

    Level 0 base models: LightGBM, XGBoost, Ridge (mỗi model 1 cột trong OOF).

    Args:
        X_train       : Feature matrix của tập train.
        y_train       : Target DataFrame (Revenue, COGS).
        best_lgb_rev  : Best params LightGBM cho Revenue.
        best_xgb_rev  : Best params XGBoost cho Revenue.
        best_lgb_cogs : Best params LightGBM cho COGS.
        best_xgb_cogs : Best params XGBoost cho COGS.
        n_splits      : Số folds (default 5).

    Returns:
        (oof_rev, oof_cogs) — DataFrames shape (n_train, 3), columns ['lgb','xgb','ridge'].
        Các rows không thuộc bất kỳ val fold nào sẽ là NaN.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_rev  = pd.DataFrame(index=X_train.index, columns=["lgb", "xgb", "ridge"], dtype=float)
    oof_cogs = pd.DataFrame(index=X_train.index, columns=["lgb", "xgb", "ridge"], dtype=float)

    for fold, (tr_i, val_i) in enumerate(tscv.split(X_train)):
        Xt, Xv   = X_train.iloc[tr_i], X_train.iloc[val_i]
        yt_r, _  = y_train["Revenue"].iloc[tr_i], y_train["Revenue"].iloc[val_i]
        yt_c, _  = y_train["COGS"].iloc[tr_i],    y_train["COGS"].iloc[val_i]

        # Fold-aware scaling — fit chỉ trên training fold
        scaler = StandardScaler()
        Xt_sc  = scaler.fit_transform(Xt)
        Xv_sc  = scaler.transform(Xv)

        # Revenue — Level 0
        m_lgb = LGBMRegressor(**best_lgb_rev, random_state=42, n_jobs=-1, verbose=-1).fit(Xt, yt_r)
        m_xgb = XGBRegressor(**best_xgb_rev, random_state=42, n_jobs=-1, tree_method="hist").fit(Xt, yt_r)
        m_rid = Ridge(alpha=100.0).fit(Xt_sc, yt_r)
        oof_rev.iloc[val_i, 0] = m_lgb.predict(Xv)
        oof_rev.iloc[val_i, 1] = m_xgb.predict(Xv)
        oof_rev.iloc[val_i, 2] = m_rid.predict(Xv_sc)

        # COGS — Level 0
        m_lgb = LGBMRegressor(**best_lgb_cogs, random_state=42, n_jobs=-1, verbose=-1).fit(Xt, yt_c)
        m_xgb = XGBRegressor(**best_xgb_cogs, random_state=42, n_jobs=-1, tree_method="hist").fit(Xt, yt_c)
        m_rid = Ridge(alpha=100.0).fit(Xt_sc, yt_c)
        oof_cogs.iloc[val_i, 0] = m_lgb.predict(Xv)
        oof_cogs.iloc[val_i, 1] = m_xgb.predict(Xv)
        oof_cogs.iloc[val_i, 2] = m_rid.predict(Xv_sc)

        log.info(f"  [OOF] Fold {fold + 1}/{n_splits} done")

    return oof_rev, oof_cogs
