"""
submit.py
=========
Chạy nhanh Step 9: Recursive Forecasting & Submission
mà KHÔNG cần huấn luyện lại toàn bộ pipeline.

Điều kiện tiên quyết:
  - Đã chạy `python main.py` ít nhất 1 lần (đã có models/ đầy đủ)
  - File Data/sample_submission.csv tồn tại

Cách chạy:
    python submit.py
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd

from src.feature_engineering import DataLoader, FeatureEngineer
from src.models import FEATURES
from src.utils import setup_logging


def main(log_file: str = "logs/submit.log") -> None:
    log = setup_logging(log_file)

    # --------- 1. Loading pre-trained models ---------

    required = ["lgb_rev", "xgb_rev", "rid_rev", "lgb_cogs", "xgb_cogs",
                "rid_cogs", "meta_rev", "meta_cogs", "scaler"]
    for name in required:
        path = f"models/{name}.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Không tìm thấy {path}. Hãy chạy `python main.py` trước."
            )

    lgb_rf          = joblib.load("models/lgb_rev.pkl")
    xgb_rf          = joblib.load("models/xgb_rev.pkl")
    rid_rf          = joblib.load("models/rid_rev.pkl")
    lgb_cf          = joblib.load("models/lgb_cogs.pkl")
    xgb_cf          = joblib.load("models/xgb_cogs.pkl")
    rid_cf          = joblib.load("models/rid_cogs.pkl")
    meta_final_rev  = joblib.load("models/meta_rev.pkl")
    meta_final_cogs = joblib.load("models/meta_cogs.pkl")
    scaler_f        = joblib.load("models/scaler.pkl")
    log.info("--------- loaded all models ---------")


    # --------- 2. Load data & build features ---------

    log.info("--------- loading data & building features ---------")

    data     = DataLoader()
    features = FeatureEngineer(data)
    df       = features.build()
    df_full  = df[df.Date <= "2022-12-31"].dropna(subset=["revenue_lag_365"])
    log.info(f"df_full: {len(df_full):,} rows (lịch sử đến 2022-12-31)")

    # --------- 3. Load test dates ---------

    sub_path = "Data/sample_submission.csv"
    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Không tìm thấy {sub_path}.")

    df_sub = pd.read_csv(sub_path, parse_dates=["Date"])
    test_dates = df_sub["Date"].values
    log.info(f"Dự báo cho {len(test_dates)} ngày "
             f"(từ {test_dates[0]} đến {test_dates[-1]})")

    # --------- 4. Recursive Forecasting Loop ---------

    df_extended      = df_full[["Date", "Revenue", "COGS"]].copy()
    predictions_rev  = []
    predictions_cogs = []

    for i, date in enumerate(test_dates):
        # Bước 5.1: thêm dummy row → build features
        new_row = pd.DataFrame({"Date": [date], "Revenue": [0.0], "COGS": [0.0]})
        df_extended = pd.concat([df_extended, new_row], ignore_index=True)

        row_df = features.prepare_recursive_features(df_extended)
        X_row  = row_df[FEATURES]

        # Bước 5.2: dự báo bằng ensemble
        pred_lgb_r   = lgb_rf.predict(X_row)[0]
        pred_xgb_r   = xgb_rf.predict(X_row)[0]
        pred_ridge_r = rid_rf.predict(scaler_f.transform(X_row))[0]
        pred_rev     = meta_final_rev.predict([[pred_lgb_r, pred_xgb_r, pred_ridge_r]])[0]

        pred_lgb_c   = lgb_cf.predict(X_row)[0]
        pred_xgb_c   = xgb_cf.predict(X_row)[0]
        pred_ridge_c = rid_cf.predict(scaler_f.transform(X_row))[0]
        pred_cogs    = meta_final_cogs.predict([[pred_lgb_c, pred_xgb_c, pred_ridge_c]])[0]

        predictions_rev.append(pred_rev)
        predictions_cogs.append(pred_cogs)

        # Bước 5.3: cập nhật lại df_extended để làm lag cho ngày kế tiếp
        df_extended.loc[df_extended.index[-1], "Revenue"] = pred_rev
        df_extended.loc[df_extended.index[-1], "COGS"]    = pred_cogs

        if (i + 1) % 50 == 0:
            log.info(f"  Forecasted {i+1}/{len(test_dates)} days...")

    # --------- 5. Final Submission ---------

    df_submission = pd.DataFrame({
        "Date":    [pd.Timestamp(d).strftime("%Y-%m-%d") for d in test_dates],
        "Revenue": predictions_rev,
        "COGS":    predictions_cogs,
    })

    # Assert đúng thứ tự
    assert len(df_submission) == len(df_sub), "Số dòng không khớp!"
    assert (list(df_submission["Date"])
            == list(df_sub["Date"].dt.strftime("%Y-%m-%d"))), "Thứ tự ngày không khớp!"

    df_submission.to_csv("submission.csv", index=False)
    log.info("--------- submission.csv generated ---------")

    log.info("\n" + str(df_submission.head()))


if __name__ == "__main__":
    main()
