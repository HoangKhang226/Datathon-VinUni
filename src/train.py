"""
src/train.py
============
Orchestrator chính của pipeline huấn luyện.

Quy trình (phản ánh chính xác notebook train.ipynb):
  1. Load & build features từ DataLoader / FeatureEngineer
  2. Chia Train / Validation theo mốc thời gian
  3. Optuna tuning (CV-based, không dùng val set)
  4. Sinh OOF predictions (5-fold, fold-aware scaling)
  5. Train Meta model (Level 1) trên OOF
  6. Đánh giá trên Validation 2022
  7. Retrain Final models trên toàn bộ dữ liệu lịch sử
  8. Giải thích mô hình bằng SHAP
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import joblib
import os

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Imports từ các module nội bộ ─────────────────────────────────────────────
from src.models import FEATURES, TARGETS, build_lgb, build_xgb, stack_predict
from src.cv     import objective_lgb, objective_xgb, run_tuning, generate_oof
from src.utils  import evaluate, log_shap_summary, setup_logging
from src.feature_engineering import DataLoader, FeatureEngineer


# ── Config defaults ────────────────────────────────────────────────────────────

TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END   = "2022-12-31"
FULL_END  = "2022-12-31"

N_TRIALS  = 30   # số Optuna trials mỗi model
OOF_FOLDS = 5    # số folds sinh OOF predictions

MLFLOW_EXPERIMENT = "revenue-cogs-stacking"


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(log_file: str = "logs/train.log") -> None:
    """
    Entry point của pipeline huấn luyện.
    Gọi hàm này từ main.py.
    """
    log = setup_logging(log_file)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    log.info(f"MLflow experiment: '{MLFLOW_EXPERIMENT}'")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("=== 1. Loading & Building Features ===")
    data     = DataLoader()
    features = FeatureEngineer(data)
    df       = features.build()
    log.info(f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    log.info(f"=== 2. Time Split — train ≤{TRAIN_END} | val {VAL_START}→{VAL_END} ===")
    df_train = df[df.Date <= TRAIN_END]
    df_val   = df[(df.Date >= VAL_START) & (df.Date <= VAL_END)]
    log.info(f"df_train: {len(df_train):,}  |  df_val: {len(df_val):,}")

    X_train = df_train[FEATURES].copy()
    y_train = df_train[TARGETS].copy()
    X_val   = df_val[FEATURES].copy()
    y_val   = df_val[TARGETS].copy()

    # ── Main MLflow run ───────────────────────────────────────────────────────
    with mlflow.start_run(run_name="stacking_cv_pipeline") as main_run:
        run_id = main_run.info.run_id
        mlflow.set_tags({"model_type": "stacking_diversity_oof", "cv": "3-fold-tscv"})

        # ── 3. Optuna Tuning (CV-based, no val leakage) ───────────────────────
        log.info("=== 3. Hyperparameter Tuning (Optuna CV) ===")
        best_lgb_rev,  _ = run_tuning(objective_lgb, X_train, y_train["Revenue"], "LGB-Rev",  N_TRIALS, run_id)
        best_xgb_rev,  _ = run_tuning(objective_xgb, X_train, y_train["Revenue"], "XGB-Rev",  N_TRIALS, run_id)
        best_lgb_cogs, _ = run_tuning(objective_lgb, X_train, y_train["COGS"],    "LGB-Cogs", N_TRIALS, run_id)
        best_xgb_cogs, _ = run_tuning(objective_xgb, X_train, y_train["COGS"],    "XGB-Cogs", N_TRIALS, run_id)

        # ── 4. OOF Generation (Level 0, fold-aware scaling) ───────────────────
        log.info("=== 4. Generating OOF Predictions ===")
        oof_rev, oof_cogs = generate_oof(
            X_train, y_train,
            best_lgb_rev, best_xgb_rev,
            best_lgb_cogs, best_xgb_cogs,
            n_splits=OOF_FOLDS,
        )

        # ── 5. Train Meta Models (Level 1) ────────────────────────────────────
        log.info("=== 5. Training Meta Models on OOF ===")
        mask_rev  = oof_rev.notna().all(axis=1)
        meta_rev  = Ridge(alpha=1.0).fit(oof_rev[mask_rev],  y_train["Revenue"][mask_rev])

        mask_cogs = oof_cogs.notna().all(axis=1)
        meta_cogs = Ridge(alpha=1.0).fit(oof_cogs[mask_cogs], y_train["COGS"][mask_cogs])

        log.info(f"Meta Weights Rev:  {meta_rev.coef_.round(3).tolist()}")
        log.info(f"Meta Weights COGS: {meta_cogs.coef_.round(3).tolist()}")

        # ── 6. Evaluation on Val 2022 ────────────────────────────────────────
        log.info("=== 6. Evaluation — Validation 2022 ===")
        scaler_val = StandardScaler()
        X_tr_sc    = scaler_val.fit_transform(X_train)
        X_val_sc   = scaler_val.transform(X_val)

        lgb_r = build_lgb(best_lgb_rev).fit(X_train, y_train["Revenue"])
        xgb_r = build_xgb(best_xgb_rev).fit(X_train, y_train["Revenue"])
        rid_r = Ridge(alpha=100.0).fit(X_tr_sc, y_train["Revenue"])

        lgb_c = build_lgb(best_lgb_cogs).fit(X_train, y_train["COGS"])
        xgb_c = build_xgb(best_xgb_cogs).fit(X_train, y_train["COGS"])
        rid_c = Ridge(alpha=100.0).fit(X_tr_sc, y_train["COGS"])

        val_meta_rev  = np.column_stack([lgb_r.predict(X_val), xgb_r.predict(X_val), rid_r.predict(X_val_sc)])
        val_meta_cogs = np.column_stack([lgb_c.predict(X_val), xgb_c.predict(X_val), rid_c.predict(X_val_sc)])

        stack_rev_pred  = meta_rev.predict(val_meta_rev)
        stack_cogs_pred = meta_cogs.predict(val_meta_cogs)

        log.info("--- Revenue ---")
        evaluate(y_val["Revenue"], val_meta_rev[:, 0], "LGB",   log)
        r_stack = evaluate(y_val["Revenue"], stack_rev_pred, "STACK", log)

        log.info("--- COGS ---")
        evaluate(y_val["COGS"], val_meta_cogs[:, 0], "LGB",   log)
        c_stack = evaluate(y_val["COGS"], stack_cogs_pred, "STACK", log)

        mlflow.log_metrics({"val_mae_rev": r_stack["mae"], "val_mae_cogs": c_stack["mae"]})
        log.info(f"MLflow run finished → {run_id}")

    # ── 7. Final Retrain (Full data 2013–2022) ────────────────────────────────
    log.info("=== 7. Final Retrain on Full Data ===")
    df_full = df[df.Date <= FULL_END].dropna(subset=["revenue_lag_365"])
    X_full  = df_full[FEATURES].copy()
    y_full  = df_full[TARGETS].copy()

    scaler_f = StandardScaler()
    X_full_sc = scaler_f.fit_transform(X_full)

    with mlflow.start_run(run_name="final_retrain"):
        lgb_rf = build_lgb(best_lgb_rev).fit(X_full, y_full["Revenue"])
        xgb_rf = build_xgb(best_xgb_rev).fit(X_full, y_full["Revenue"])
        rid_rf = Ridge(alpha=100.0).fit(X_full_sc, y_full["Revenue"])

        lgb_cf = build_lgb(best_lgb_cogs).fit(X_full, y_full["COGS"])
        xgb_cf = build_xgb(best_xgb_cogs).fit(X_full, y_full["COGS"])
        rid_cf = Ridge(alpha=100.0).fit(X_full_sc, y_full["COGS"])

        meta_X_f_rev  = np.column_stack([lgb_rf.predict(X_full), xgb_rf.predict(X_full), rid_rf.predict(X_full_sc)])
        meta_X_f_cogs = np.column_stack([lgb_cf.predict(X_full), xgb_cf.predict(X_full), rid_cf.predict(X_full_sc)])

        meta_final_rev  = Ridge(alpha=1.0).fit(meta_X_f_rev,  y_full["Revenue"])
        meta_final_cogs = Ridge(alpha=1.0).fit(meta_X_f_cogs, y_full["COGS"])

        os.makedirs("models", exist_ok=True)
        for name, obj in {
            "lgb_rev": lgb_rf, "xgb_rev": xgb_rf, "rid_rev": rid_rf,
            "lgb_cogs": lgb_cf, "xgb_cogs": xgb_cf, "rid_cogs": rid_cf,
            "meta_rev": meta_final_rev, "meta_cogs": meta_final_cogs,
            "scaler": scaler_f,
        }.items():
            path = f"models/{name}.pkl"
            joblib.dump(obj, path)
            mlflow.log_artifact(path, artifact_path="models")

        log.info("Final models saved to models/")

    # ── 8. SHAP Interpretation ─────────────────────────────────────────────────
    log.info("=== 8. SHAP Model Interpretation ===")
    with mlflow.start_run(run_name="model_interpretation"):
        log_shap_summary(lgb_rf, X_full, "Revenue", "LGBM", mlflow, log)
        log_shap_summary(lgb_cf, X_full, "COGS",    "LGBM", mlflow, log)

    log.info("Pipeline complete ✓")

    # ── 9. Recursive Forecasting & Submission ──────────────────────────────────
    log.info("=== 9. Recursive Forecasting & Submission ===")

    sub_path = "Data/sample_submission.csv"
    if not os.path.exists(sub_path):
        log.warning(f"File {sub_path} không tồn tại. Bỏ qua bước submission.")
        return

    # 5.1 Load test dates từ sample_submission
    df_sub = pd.read_csv(sub_path, parse_dates=["Date"])
    test_dates = df_sub["Date"].values  # 2023-01-01 → 2024-07-01

    # Bắt đầu từ full dataframe (đã có đủ lịch sử đến 2022-12-31)
    df_extended = df_full.copy()

    predictions_rev  = []
    predictions_cogs = []

    for i, date in enumerate(test_dates):
        # 1. Append dummy row rồi build features cho ngày 'date'
        new_row_init = pd.DataFrame({"Date": [date], "Revenue": [0.0], "COGS": [0.0]})
        df_extended = pd.concat([df_extended, new_row_init], ignore_index=True)

        row_df = features.prepare_recursive_features(df_extended)
        X_row  = row_df[FEATURES]

        # 2. Predict bằng ensemble (Revenue)
        pred_lgb_r   = lgb_rf.predict(X_row)[0]
        pred_xgb_r   = xgb_rf.predict(X_row)[0]
        pred_ridge_r = rid_rf.predict(scaler_f.transform(X_row))[0]
        pred_rev     = meta_final_rev.predict([[pred_lgb_r, pred_xgb_r, pred_ridge_r]])[0]

        # Tương tự cho COGS
        pred_lgb_c   = lgb_cf.predict(X_row)[0]
        pred_xgb_c   = xgb_cf.predict(X_row)[0]
        pred_ridge_c = rid_cf.predict(scaler_f.transform(X_row))[0]
        pred_cogs    = meta_final_cogs.predict([[pred_lgb_c, pred_xgb_c, pred_ridge_c]])[0]

        predictions_rev.append(pred_rev)
        predictions_cogs.append(pred_cogs)

        # 3. Cập nhật kết quả vào df_extended để dùng làm lag cho ngày tiếp theo
        df_extended.loc[df_extended.index[-1], "Revenue"] = pred_rev
        df_extended.loc[df_extended.index[-1], "COGS"]    = pred_cogs

        if (i + 1) % 50 == 0:
            log.info(f"  Forecasted {i+1}/{len(test_dates)} days...")

    # 5.2 Xuất submission.csv
    df_submission = pd.DataFrame({
        "Date":    [pd.Timestamp(d).strftime("%Y-%m-%d") for d in test_dates],
        "Revenue": predictions_rev,
        "COGS":    predictions_cogs,
    })

    assert len(df_submission) == len(df_sub), "Số dòng không khớp!"
    assert list(df_submission["Date"]) == list(df_sub["Date"].dt.strftime("%Y-%m-%d")), "Thứ tự ngày không khớp!"

    df_submission.to_csv("submission.csv", index=False)
    log.info("✅ Đã xuất submission.csv")
    log.info("\n" + str(df_submission.head()))
