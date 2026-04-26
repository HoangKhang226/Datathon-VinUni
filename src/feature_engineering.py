import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path("Data")

from src.data_loader import DataLoader


# ==================================================================
class FeatureEngineer:
    """
    Xây dựng feature matrix từ DataLoader cho bài toán dự báo Revenue & COGS.

    Dựa trên kết quả EDA:
    - Log1p transform: skew 1.67 -> -0.16 (dùng cho Ridge, optional)
    - Lag ưu tiên: lag_1, lag_2, lag_6, lag_14, lag_365
    - Web traffic: chỉ giữ sessions, unique_visitors, page_views (r~0.32)
    - Đỉnh mùa vụ: T3-T5; Đáy: T11-T12 -> dùng Fourier + holiday flags
    - promo_id_2: bỏ hẳn (100% missing)
    """

    TARGETS   = ["Revenue", "COGS"]

    def __init__(self, data: DataLoader):
        self.data = data
        self.df   = None   # feature matrix cuối cùng

    # ── Public API ─────────────────────────────────────────────────
    def build(self) -> pd.DataFrame:
        """Chạy toàn bộ pipeline và trả về feature matrix."""
        log.info("Building features...")
        self.df = self.data.sales.copy().sort_values("Date").reset_index(drop=True)

        self._add_time_features()
        self._add_lag_rolling_features()
        self._add_seasonal_features()
        self._add_transaction_features()
        self._add_web_traffic_features()
        self._add_inventory_features()
        self._add_promotion_features()
        self._drop_na_from_max_lag()

        log.info(f"Feature matrix: {self.df.shape[0]:,} rows x {self.df.shape[1]} cols")
        log.info(f"  Targets : {self.TARGETS}")
        log.info(f"  Features: {len(self.get_feature_cols())} columns")
        return self.df

    def get_feature_cols(self) -> list:
        """Trả về danh sách tên các cột feature (loại Date và targets)."""
        exclude = {"Date"} | set(self.TARGETS)
        return [c for c in self.df.columns if c not in exclude]

    # ── Nhóm A: Time Features ───────────────────────────────────────
    def _add_time_features(self):
        df = self.df
        df["year"]          = df["Date"].dt.year
        df["month"]         = df["Date"].dt.month
        df["day"]           = df["Date"].dt.day
        df["day_of_week"]   = df["Date"].dt.dayofweek
        df["day_of_year"]   = df["Date"].dt.dayofyear
        df["week_of_year"]  = df["Date"].dt.isocalendar().week.astype(int)
        df["quarter"]       = df["Date"].dt.quarter
        df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
        df["is_month_end"]  = df["Date"].dt.is_month_end.astype(int)
        df["is_month_start"]= df["Date"].dt.is_month_start.astype(int)
        df["is_year_end"]   = ((df["month"] == 12) & (df["day"] == 31)).astype(int)
        df["is_year_start"] = ((df["month"] == 1)  & (df["day"] == 1)).astype(int)
        # Fourier encoding
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
        df["sin_dow"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["cos_dow"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
        log.info("  Time features done")

    # ── Nhóm B: Lag & Rolling Features ─────────────────────────────
    def _add_lag_rolling_features(self):
        df = self.df
        # Lag features
        for lag in [1, 2, 6, 7, 14, 30, 90, 365]:
            df[f"revenue_lag_{lag}"] = df["Revenue"].shift(lag)
        for lag in [1, 7, 30, 365]:
            df[f"cogs_lag_{lag}"] = df["COGS"].shift(lag)

        # Rolling (shift(1) trước để không leak ngày T)
        rev_shifted = df["Revenue"].shift(1)
        for window in [7, 14, 30, 90]:
            df[f"revenue_roll_mean_{window}"] = rev_shifted.rolling(window).mean()
            df[f"revenue_roll_std_{window}"]  = rev_shifted.rolling(window).std()
        for window in [7, 30]:
            df[f"cogs_roll_mean_{window}"] = df["COGS"].shift(1).rolling(window).mean()

        # Exponential weighted mean
        for span in [7, 30]:
            df[f"revenue_ewm_{span}"] = rev_shifted.ewm(span=span).mean()

        # Difference features
        df["revenue_diff_1"]       = df["Revenue"].shift(1) - df["Revenue"].shift(2)
        df["revenue_diff_7"]       = df["Revenue"].shift(1) - df["Revenue"].shift(8)
        df["revenue_pct_change_7"] = (
            df["Revenue"].shift(1) / df["Revenue"].shift(8) - 1
        ).replace([np.inf, -np.inf], np.nan)
        log.info("  Lag & Rolling features done")

    # ── Nhóm C: Seasonal / Holiday Features ──────────────────
    def _add_seasonal_features(self):
        df  = self.df
        m   = df["Date"].dt.month
        day = df["Date"].dt.day
        dow = df["Date"].dt.dayofweek

        df["is_1111"]         = ((m == 11) & (day == 11)).astype(int)
        df["is_1212"]         = ((m == 12) & (day == 12)).astype(int)
        df["is_christmas"]    = ((m == 12) & (day >= 23)).astype(int)
        df["is_new_year"]     = ((m == 1)  & (day <= 3)).astype(int)
        df["is_womens_day"]   = ((m == 3)  & (day == 8)).astype(int)
        df["is_valentines"]   = ((m == 2)  & (day.between(10, 14))).astype(int)
        df["is_mid_year_sale"]= (((m == 6) & (day >= 25)) | ((m == 7) & (day <= 5))).astype(int)
        df["is_tet_period"]   = (((m == 1) & (day.between(15, 31))) | ((m == 2) & (day <= 15))).astype(int)
        df["is_black_friday"]   = ((m == 11) & (dow == 4) & (day.between(22, 28))).astype(int)
        df["is_back_to_school"] = (m == 8).astype(int)

        # days_to_tet: số ngày đến Tết âm lịch gần nhất (2012-2024)
        TET_DATES = [
            pd.Timestamp("2012-01-23"), pd.Timestamp("2013-02-10"),
            pd.Timestamp("2014-01-31"), pd.Timestamp("2015-02-19"),
            pd.Timestamp("2016-02-08"), pd.Timestamp("2017-01-28"),
            pd.Timestamp("2018-02-16"), pd.Timestamp("2019-02-05"),
            pd.Timestamp("2020-01-25"), pd.Timestamp("2021-02-12"),
            pd.Timestamp("2022-02-01"), pd.Timestamp("2023-01-22"),
            pd.Timestamp("2024-02-10"),
        ]
        def days_to_next_tet(date):
            future = [t for t in TET_DATES if t >= date]
            return (future[0] - date).days if future else 365

        df["days_to_tet"] = df["Date"].apply(days_to_next_tet)
        log.info("  Seasonal & Holiday features done")


    # ── Nhóm D: Transaction Aggregates ─────────────────────────────
    def _add_transaction_features(self):
        orders = self.data.orders
        items  = self.data.order_items
        pays   = self.data.payments
        rets   = self.data.returns
        revs   = self.data.reviews

        daily_ord = orders.groupby("order_date").agg(
            daily_order_count  = ("order_id",     "count"),
            daily_cancel_count = ("order_status", lambda x: (x == "cancelled").sum()),
        ).reset_index().rename(columns={"order_date": "Date"})
        daily_ord["daily_cancel_rate"] = (
            daily_ord["daily_cancel_count"] / daily_ord["daily_order_count"]
        )
        # Bỏ daily_cancel_count — trung gian, redundant với daily_cancel_rate
        daily_ord = daily_ord.drop(columns=["daily_cancel_count"])

        items_d = items.merge(orders[["order_id", "order_date"]], on="order_id")
        daily_items = items_d.groupby("order_date").agg(
            daily_items_sold     = ("quantity",        "sum"),
            daily_discount_total = ("discount_amount", "sum"),
            daily_promo_rate     = ("promo_id",        lambda x: x.notna().mean()),
        ).reset_index().rename(columns={"order_date": "Date"})

        pays_d   = pays.merge(orders[["order_id", "order_date"]], on="order_id")
        daily_pay = pays_d.groupby("order_date").agg(
            daily_avg_payment = ("payment_value", "mean"),
        ).reset_index().rename(columns={"order_date": "Date"})

        daily_ret = rets.groupby("return_date").agg(
            daily_return_count = ("return_id",     "count"),
            daily_refund_total = ("refund_amount",  "sum"),
        ).reset_index().rename(columns={"return_date": "Date"})

        daily_rev = revs.groupby("review_date").agg(
            daily_avg_rating   = ("rating",    "mean"),
            daily_review_count = ("review_id", "count"),
        ).reset_index().rename(columns={"review_date": "Date"})

        for dft in [daily_ord, daily_items, daily_pay, daily_ret, daily_rev]:
            self.df = self.df.merge(dft, on="Date", how="left")

        trans_cols = [
            "daily_order_count", "daily_cancel_rate", "daily_items_sold",
            "daily_discount_total", "daily_promo_rate", "daily_avg_payment",
            "daily_return_count", "daily_refund_total",
            "daily_avg_rating", "daily_review_count",
        ]
        for col in trans_cols:
            self.df[col] = self.df[col].fillna(self.df[col].shift(365))
        log.info("  Transaction aggregate features done")

    # ── Nhóm E: Web Traffic Features ────────────────────────────────
    def _add_web_traffic_features(self):
        # Chỉ giữ 3 features có r~0.32 (bỏ bounce_rate, session_dur)
        daily_web = self.data.web_traffic.groupby("date").agg(
            daily_sessions        = ("sessions",       "sum"),
            daily_unique_visitors = ("unique_visitors", "sum"),
            daily_page_views      = ("page_views",      "sum"),
        ).reset_index().rename(columns={"date": "Date"})

        self.df = self.df.merge(daily_web, on="Date", how="left")

        for col in ["daily_sessions", "daily_unique_visitors", "daily_page_views"]:
            self.df[col] = self.df[col].fillna(self.df[col].shift(365))
            self.df[col] = self.df[col].fillna(self.df[col].median())
        log.info("  Web traffic features done")

    # ── Nhóm F: Inventory Features (monthly) ────────────────────────
    def _add_inventory_features(self):
        inv = self.data.inventory.copy()
        inv["year"]  = inv["snapshot_date"].dt.year
        inv["month"] = inv["snapshot_date"].dt.month

        monthly_inv = inv.groupby(["year", "month"]).agg(
            monthly_total_stock      = ("stock_on_hand",    "sum"),
            monthly_stockout_count   = ("stockout_flag",    "sum"),
            monthly_avg_fill_rate    = ("fill_rate",        "mean"),
            monthly_avg_sell_through = ("sell_through_rate","mean"),
            monthly_total_units_sold = ("units_sold",       "sum"),
        ).reset_index()

        self.df = self.df.merge(monthly_inv, on=["year", "month"], how="left")
        log.info("  Inventory features done")

    # ── Nhóm G: Promotion Features ───────────────────────────────────
    def _add_promotion_features(self):
        promos = self.data.promotions

        def count_active(date):
            mask   = (promos["start_date"] <= date) & (promos["end_date"] >= date)
            active = promos[mask]
            return pd.Series({
                "active_promo_count": len(active),
                "avg_discount_value": active["discount_value"].mean() if len(active) > 0 else 0.0,
                "has_pct_promo":      int((active["promo_type"] == "percentage").any()),
                "has_fixed_promo":    int((active["promo_type"] == "fixed").any()),
            })

        promo_feats = self.df["Date"].apply(count_active)
        # Cast về int để tránh float64 do apply()
        for col in ["active_promo_count", "has_pct_promo", "has_fixed_promo"]:
            promo_feats[col] = promo_feats[col].astype(int)
        self.df = pd.concat([self.df, promo_feats], axis=1)
        log.info("  Promotion features done")

    # ── Utility ──────────────────────────────────────────────────────
    def _drop_na_from_max_lag(self):
        """Bỏ các dòng đầu bị NaN do lag lớn nhất (lag_365)."""
        before   = len(self.df)
        self.df  = self.df.dropna(subset=["revenue_lag_365"]).reset_index(drop=True)
        dropped  = before - len(self.df)
        log.info(f"  Dropped {dropped} rows from lag_365 NaN")

    def prepare_recursive_features(self, df_extended: pd.DataFrame) -> pd.DataFrame:
        """
        Xây dựng features cho dòng cuối cùng của df_extended.
        Dùng cho vòng lặp dự báo đệ quy (Step 9).

        df_extended phải chứa đủ lịch sử (Revenue, COGS) đến ngày trước đó,
        cộng thêm 1 dummy row cho ngày cần dự báo (Revenue=0, COGS=0).
        """
        original_df = self.df          # lưu lại để rollback
        # Đảm bảo chỉ giữ lại các cột lõi để tránh trùng tên khi build lại
        self.df = df_extended[["Date", "Revenue", "COGS"]].copy()

        # Tính lại toàn bộ features trên df_extended
        self._add_time_features()
        self._add_lag_rolling_features()
        self._add_seasonal_features()
        self._add_transaction_features()
        self._add_web_traffic_features()
        self._add_inventory_features()
        self._add_promotion_features()

        # Lấy dòng cuối (ngày cần dự báo)
        row_feat = self.df.iloc[[-1]].copy()

        self.df = original_df          # rollback
        return row_feat


def fast_check_features(df):
    feat_cols = FeatureEngineer(DataLoader()).get_feature_cols() # Dummy check
    log.info(f"Feature: {'Dtype':<10} {'Unique':>7} {'Missing%':>10}  Top values / Range")
    log.info("─" * 100)

    for col in feat_cols:
        series  = df[col]
        dtype   = str(series.dtype)
        n_uniq  = series.nunique()
        miss_pct = series.isna().mean() * 100

        if n_uniq <= 10:
            val_counts = series.value_counts(normalize=True, dropna=False)
            top = "  |  ".join([f"{v}: {p*100:.1f}%" for v, p in val_counts.items()])
        else:
            top = f"min={series.min():,.2f}  mean={series.mean():,.2f}  max={series.max():,.2f}"

        log.info(f"{col:<35} {dtype:<10} {n_uniq:>7,} {miss_pct:>9.1f}%  {top}")


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    data = DataLoader()
    features   = FeatureEngineer(data)

    df   = features.build()
    fast_check_features(df)

