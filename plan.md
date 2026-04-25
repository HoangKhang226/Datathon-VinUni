# 🗺️ Kế hoạch Chi tiết — Dự báo Doanh thu (Revenue & COGS)

## Tổng quan Pipeline

```
[0] EDA & Phân tích Dữ liệu
        ↓
[1] Feature Engineering  →  [2] Train/Val Split  →  [3] Train Ensemble
        ↓                                                   ↓
[5] Xuất submission.csv  ←  [4] Retrain Full Data  ←  Tinh chỉnh tham số
```

---

## Bước 0 — EDA & Phân tích Dữ liệu (60 điểm — Phần 2 đề thi)

### Kết quả EDA thực tế

---

#### 0.A Outlier & Phân phối Revenue

| Chỉ số | Giá trị |
|--------|---------|
| Q1 | 2.47M VNĐ |
| Q3 | 5.35M VNĐ |
| IQR | 2.88M VNĐ |
| Ngưỡng dưới (Q1 - 1.5×IQR) | **-1.85M** → không có outlier thấp thực sự |
| Ngưỡng trên (Q3 + 1.5×IQR) | **9.67M** |
| Số ngày outlier (vượt ngưỡng trên) | **169 / 3833 ngày (4.4%)** |

> ⚠️ **Lưu ý:** Ngưỡng dưới âm (-1.85M) có nghĩa là **không tồn tại outlier thấp** — mọi ngày Revenue đều dương và hợp lý. Các outlier 4.4% đều là **spike cao** (ngày lễ lớn), không phải lỗi dữ liệu.

**→ Quyết định:** Dùng **cắm cờ** (không winsorize) — tạo feature `is_high_revenue_day` để model nhận biết các ngày đặc biệt.

---

#### 0.B Log-transform

| Transform | Skewness | Kurtosis | Shapiro-W p | Đánh giá |
|-----------|----------|----------|-------------|---------|
| Original | 1.6700 | 4.0303 | 0.0000 | ❌ Lệch nhiều |
| **log1p** | **-0.1594** | **0.1891** | **0.1999** | ✅ Gần chuẩn nhất |
| sqrt | 0.7404 | 0.7572 | 0.0000 | 🟡 Cải thiện nhưng chưa đủ |

> ✅ **Kết luận: Dùng `log1p(Revenue)` làm target khi train Ridge Regression.**
> - Skewness giảm từ 1.67 → -0.16 (gần chuẩn)
> - Shapiro-W p = 0.1999 > 0.05 → không bác bỏ chuẩn
> - Sau khi predict: `np.expm1(pred)` để inverse lại
> - LGB và XGBoost: có thể dùng target gốc hoặc log1p đều được

---

#### 0.C Lag quan trọng từ ACF/PACF

Tổng số lag có ý nghĩa: **ACF = 198 lag, PACF = 109 lag**

| Lag | ACF | PACF | Kết luận |
|-----|-----|------|---------|
| **lag_1** | +0.8654 ✅ | +0.8654 ✅ | 🔴 **Ưu tiên cao nhất** — tương quan trực tiếp rất mạnh |
| **lag_2** | +0.7350 ✅ | -0.0556 ✅ | 🟡 ACF mạnh nhưng PACF gián tiếp |
| lag_3 | +0.6214 ✅ | -0.0076 ❌ | Bỏ qua PACF |
| **lag_6** | +0.4673 ✅ | +0.3778 ✅ | 🟡 Cả hai có ý nghĩa |
| lag_7 | +0.4917 ✅ | +0.0186 ❌ | ACF có nhưng PACF không — ảnh hưởng gián tiếp qua lag_1 |
| **lag_14** | +0.4956 ✅ | -0.0582 ✅ | 🟡 Giữ lại |
| lag_21 | +0.4356 ✅ | +0.0204 ❌ | Bỏ qua |
| **lag_365** | +0.7380 ✅ | +0.0431 ✅ | 🔴 **Ưu tiên cao** — mùa vụ năm trước |

> ✅ **Lag cần ưu tiên trong feature engineering:** `lag_1`, `lag_2`, `lag_6`, `lag_14`, `lag_365`
> Lag_7 ACF cao nhưng PACF không có ý nghĩa → ảnh hưởng gián tiếp qua lag_1, vẫn nên giữ vì phổ biến trong time-series tuần.

---

#### 0.D Missing Rate

| Bảng | Kết quả | Quyết định |
|------|---------|-----------|
| sales, orders, payments, customers, products, inventory, web_traffic, returns, reviews, shipments, geography | ✅ Không missing | Merge bình thường |
| **order_items** | 🔴 `promo_id`: 61.3%, `promo_id_2`: 100% | `promo_id` NaN = không dùng promo (bình thường). `promo_id_2` bỏ hẳn |
| **promotions** | 🔴 `applicable_category`: 80% | NaN = áp dụng tất cả categories (theo đề bài) |

> ✅ **Quyết định:**
> - `promo_id` NaN → fill bằng `"no_promo"`, tạo feature `has_promo = (promo_id != "no_promo")`
> - `promo_id_2` → **bỏ hoàn toàn** (100% missing)
> - `applicable_category` NaN → fill `"all"` theo đúng ý nghĩa nghiệp vụ

---

#### 0.E Web Traffic Coverage

| Chỉ số | Giá trị |
|--------|---------|
| Số ngày trong sales | 3,833 ngày |
| Số ngày trong web_traffic | 3,652 ngày |
| Missing sessions sau merge | **181 ngày (4.7%)** |

> ⚠️ **181 ngày không có web_traffic** (web_traffic bắt đầu muộn hơn sales hoặc có ngày bị thiếu).
> **→ Xử lý:** Fill NaN bằng `lag_365` (cùng kỳ năm trước). Nếu vẫn NaN, fill bằng median.

---

#### 0.F Tính mùa vụ (từ biểu đồ)

Từ biểu đồ **"Revenue theo Tháng — So sánh từng Năm"**:

| Quan sát | Chi tiết |
|---------|---------|
| **Đỉnh rõ ràng** | T3–T5 (Quý 1 cuối + Quý 2 đầu) — spike đồng nhất qua các năm |
| **Đáy rõ ràng** | T11–T12 — Revenue thấp nhất (ngược với kỳ vọng về 11.11/12.12) |
| **Tính mùa vụ** | Có, pattern lặp lại rõ qua các năm → cần Fourier features |
| **Xu hướng tăng** | Revenue tổng thể tăng từ 2012 → 2019, ổn định 2019–2022 |

> ✅ **Lưu ý đặc biệt:** T11–T12 thấp trái với kỳ vọng — có thể do dữ liệu aggregate theo tháng làm "loãng" spike 11.11, 12.12. Cần tạo feature ngày cụ thể (`is_1111`, `is_1212`) thay vì chỉ dùng tháng.

#### 0.G Cross-Correlation Web Traffic → Revenue

```
                  lag_0   lag_1   lag_2   lag_3   lag_5   lag_7  lag_14  lag_30
avg_bounce_rate -0.0206 -0.0173 -0.0016  0.0030  0.0130 -0.0146 -0.0235 -0.0175
avg_session_dur -0.0256 -0.0214 -0.0262 -0.0087 -0.0004 -0.0069 -0.0035 -0.0150
page_views       0.3016  0.3055  0.2949  0.2846  0.2863  0.2865  0.2643  0.1994
sessions         0.3211  0.3216  0.3159  0.3119  0.3129  0.3092  0.2885  0.2132
unique_visitors  0.3188  0.3185  0.3113  0.3046  0.3093  0.3073  0.2870  0.2149
```

**Phân tích:**

| Feature | Correlation | Nhận xét | Quyết định |
|---------|------------|---------|-----------|
| `sessions` | ~0.32 | Tương quan vừa, ổn định qua các lag | ✅ Giữ làm feature |
| `unique_visitors` | ~0.32 | Tương tự sessions, không cần cả hai | ✅ Giữ 1 trong 2 |
| `page_views` | ~0.30 | Tương quan vừa | ✅ Giữ làm feature |
| `avg_bounce_rate` | ~-0.02 | Gần bằng 0 | ❌ Bỏ |
| `avg_session_dur` | ~-0.02 | Gần bằng 0 | ❌ Bỏ |

> ⚠️ **Lưu ý quan trọng — Không phải leading indicator:**
> Correlation hầu như không giảm từ lag_0 (0.321) đến lag_7 (0.309) → `sessions` hôm nay **không dự báo revenue ngày mai tốt hơn cùng ngày**. Điều này cho thấy cả hai cùng bị drive bởi **seasonality chung** (tháng cao điểm → traffic & revenue đều tăng).
>
> **→ Hệ quả thực tế:** Không thể dùng web traffic như một "tín hiệu dẫn trước" (leading signal). Chỉ dùng làm feature bổ trợ cùng ngày — và với tập test 2023-2024 sẽ phải fill bằng lag_365.

---

### Checklist EDA ✅

- [x] Xác nhận Revenue có tính mùa vụ — **đỉnh T3-T5, đáy T11-T12**
- [x] Outlier: 169 ngày (4.4%) đều là spike cao → **cắm cờ, không winsorize**
- [x] Log-transform: **log1p cải thiện mạnh** (skew 1.67 → -0.16), dùng cho Ridge
- [x] Lag ưu tiên: **lag_1, lag_2, lag_6, lag_14, lag_365**
- [x] Missing rate: chỉ `order_items.promo_id_2` (100%) cần bỏ; còn lại xử lý được
- [x] Web traffic cross-correlation: sessions/visitors/page_views r≈0.32 (**giữ**); bounce_rate/session_dur r≈0 (**bỏ**); không phải leading indicator → dùng cùng ngày + fill lag_365

---

## Bước 1 — Feature Engineering

### 1.1 Load dữ liệu gốc

```python
# File chính (target)
df = pd.read_csv("Data/sales.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Bảng bổ trợ
df_orders      = pd.read_csv("Data/orders.csv",      parse_dates=["order_date"])
df_items       = pd.read_csv("Data/order_items.csv")
df_payments    = pd.read_csv("Data/payments.csv")
df_web         = pd.read_csv("Data/web_traffic.csv", parse_dates=["date"])
df_inventory   = pd.read_csv("Data/inventory.csv",   parse_dates=["snapshot_date"])
df_promotions  = pd.read_csv("Data/promotions.csv",  parse_dates=["start_date","end_date"])
df_returns     = pd.read_csv("Data/returns.csv",     parse_dates=["return_date"])
df_reviews     = pd.read_csv("Data/reviews.csv",     parse_dates=["review_date"])
```

---

### 1.2 Nhóm A — Time Features (từ cột `Date`)

> Nguồn: `sales.csv` — tự tính từ cột Date, không có nguy cơ data leakage

| Feature | Code | Ghi chú |
|---------|------|---------|
| `year` | `df.Date.dt.year` | |
| `month` | `df.Date.dt.month` | |
| `day` | `df.Date.dt.day` | |
| `day_of_week` | `df.Date.dt.dayofweek` | 0=Thứ 2, 6=Chủ nhật |
| `day_of_year` | `df.Date.dt.dayofyear` | |
| `week_of_year` | `df.Date.dt.isocalendar().week` | |
| `quarter` | `df.Date.dt.quarter` | |
| `is_weekend` | `dow >= 5` | |
| `is_month_end` | `df.Date.dt.is_month_end` | |
| `is_month_start` | `df.Date.dt.is_month_start` | |
| `is_year_end` | `month==12 & day==31` | |
| `is_year_start` | `month==1 & day==1` | |
| `sin_month` | `sin(2π × month / 12)` | Fourier encoding mùa vụ |
| `cos_month` | `cos(2π × month / 12)` | Fourier encoding mùa vụ |
| `sin_dow` | `sin(2π × dow / 7)` | Fourier encoding tuần |
| `cos_dow` | `cos(2π × dow / 7)` | Fourier encoding tuần |

---

### 1.3 Nhóm B — Lag & Rolling Features (từ `sales.csv`)

> Nguồn: `Revenue` và `COGS` của các ngày **trước** ngày hiện tại — an toàn, không leakage

**Lag Features:**

| Feature | Lag | Ý nghĩa |
|---------|-----|---------|
| `revenue_lag_1` | 1 ngày | Doanh thu hôm qua |
| `revenue_lag_7` | 7 ngày | Cùng thứ tuần trước |
| `revenue_lag_14` | 14 ngày | Cùng thứ 2 tuần trước |
| `revenue_lag_30` | 30 ngày | Cùng kỳ tháng trước |
| `revenue_lag_90` | 90 ngày | Cùng kỳ quý trước |
| `revenue_lag_365` | 365 ngày | Cùng ngày năm trước |
| `cogs_lag_1` | 1 ngày | |
| `cogs_lag_7` | 7 ngày | |
| `cogs_lag_30` | 30 ngày | |
| `cogs_lag_365` | 365 ngày | |

**Rolling Mean / Std Features:**

| Feature | Window | Ý nghĩa |
|---------|--------|---------|
| `revenue_roll_mean_7` | 7 ngày | Xu hướng ngắn hạn |
| `revenue_roll_mean_14` | 14 ngày | |
| `revenue_roll_mean_30` | 30 ngày | Xu hướng trung hạn |
| `revenue_roll_mean_90` | 90 ngày | Xu hướng dài hạn |
| `revenue_roll_std_7` | 7 ngày | Độ biến động |
| `revenue_roll_std_30` | 30 ngày | |
| `cogs_roll_mean_7` | 7 ngày | |
| `cogs_roll_mean_30` | 30 ngày | |

> ⚠️ **Quan trọng:** Phải dùng `.shift(1)` trước khi rolling để đảm bảo không dùng giá trị ngày T khi tính feature cho ngày T.
> ```python
> df["revenue_roll_mean_7"] = df["Revenue"].shift(1).rolling(7).mean()
> ```

**Exponential Weighted Mean:**

| Feature | Span | Ý nghĩa |
|---------|------|---------|
| `revenue_ewm_7` | 7 | Trung bình mũ — nhạy hơn với biến động gần đây |
| `revenue_ewm_30` | 30 | |

**Difference Features:**

| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `revenue_diff_1` | `Revenue - lag_1` | Biến động so với hôm qua |
| `revenue_diff_7` | `Revenue - lag_7` | Biến động so với tuần trước |
| `revenue_pct_change_7` | `(lag_1 - lag_7) / lag_7` | % thay đổi |

---

### 1.4 Nhóm C — Seasonal / Holiday Features

> Nguồn: Tự tính từ `Date` — có sẵn cho cả tập test 2023-2024

**Sự kiện thương mại điện tử Việt Nam:**

| Feature | Điều kiện | Ghi chú |
|---------|-----------|---------|
| `is_tet_period` | Khoảng -15 đến +5 ngày so với Tết âm lịch | Dùng thư viện `holidays` hoặc hard-code |
| `days_to_tet` | Số ngày đến Tết gần nhất | Âm nếu đã qua Tết |
| `is_1111` | `month==11 & day==11` | Ngày hội mua sắm |
| `is_1212` | `month==12 & day==12` | |
| `is_black_friday` | Thứ 6 tuần 4 tháng 11 | |
| `is_mid_year_sale` | `month==6 & day >= 25` hoặc `month==7 & day <= 5` | |
| `is_back_to_school` | `month==8` | |
| `is_christmas` | `month==12 & day >= 23` | |
| `is_valentines` | `month==2 & day >= 10 & day <= 14` | |
| `is_womens_day` | `month==3 & day==8` | |

---

### 1.5 Nhóm D — Transaction Aggregates (chỉ dùng cho tập train 2012-2022)

> Nguồn: `orders.csv`, `order_items.csv`, `payments.csv`, `returns.csv`, `reviews.csv`
> Merge vào `df` theo ngày — sẽ bị NaN với tập test, cần fill bằng lag_365

**Tạo daily_transaction_features:**

```python
# Từ orders.csv
daily_orders = df_orders.groupby("order_date").agg(
    daily_order_count  = ("order_id", "count"),
    daily_cancel_count = ("order_status", lambda x: (x == "cancelled").sum()),
).reset_index().rename(columns={"order_date": "Date"})

daily_orders["daily_cancel_rate"] = daily_orders["daily_cancel_count"] / daily_orders["daily_order_count"]

# Từ order_items.csv (join với orders để lấy order_date)
items_with_date = df_items.merge(df_orders[["order_id","order_date"]], on="order_id")
daily_items = items_with_date.groupby("order_date").agg(
    daily_items_sold     = ("quantity", "sum"),
    daily_discount_total = ("discount_amount", "sum"),
    daily_promo_rate     = ("promo_id", lambda x: x.notna().mean()),
).reset_index().rename(columns={"order_date": "Date"})

# Từ payments.csv (join với orders)
pay_with_date = df_payments.merge(df_orders[["order_id","order_date"]], on="order_id")
daily_pay = pay_with_date.groupby("order_date").agg(
    daily_avg_payment = ("payment_value", "mean"),
).reset_index().rename(columns={"order_date": "Date"})

# Từ returns.csv
daily_returns = df_returns.groupby("return_date").agg(
    daily_return_count  = ("return_id", "count"),
    daily_refund_total  = ("refund_amount", "sum"),
).reset_index().rename(columns={"return_date": "Date"})

# Từ reviews.csv
daily_reviews = df_reviews.groupby("review_date").agg(
    daily_avg_rating = ("rating", "mean"),
    daily_review_count = ("review_id", "count"),
).reset_index().rename(columns={"review_date": "Date"})
```

**Merge tất cả vào df chính:**
```python
for dft in [daily_orders, daily_items, daily_pay, daily_returns, daily_reviews]:
    df = df.merge(dft, on="Date", how="left")
```

**Fill NaN cho tập test (lag_365):**
```python
transaction_cols = ["daily_order_count", "daily_cancel_rate", "daily_items_sold", ...]
for col in transaction_cols:
    df[col] = df[col].fillna(df[col].shift(365))
```

---

### 1.6 Nhóm E — Web Traffic Features

> Nguồn: `web_traffic.csv` — aggregate theo ngày

```python
daily_web = df_web.groupby("date").agg(
    daily_sessions          = ("sessions", "sum"),
    daily_unique_visitors   = ("unique_visitors", "sum"),
    daily_page_views        = ("page_views", "sum"),
    daily_avg_bounce_rate   = ("bounce_rate", "mean"),
    daily_avg_session_dur   = ("avg_session_duration_sec", "mean"),
).reset_index().rename(columns={"date": "Date"})

df = df.merge(daily_web, on="Date", how="left")
# Fill NaN bằng lag_365
```

---

### 1.7 Nhóm F — Inventory Features (theo tháng)

> Nguồn: `inventory.csv` — snapshot cuối tháng, forward fill sang các ngày trong tháng

```python
monthly_inv = df_inventory.groupby(["year","month"]).agg(
    monthly_total_stock      = ("stock_on_hand", "sum"),
    monthly_stockout_count   = ("stockout_flag", "sum"),
    monthly_avg_fill_rate    = ("fill_rate", "mean"),
    monthly_avg_sell_through = ("sell_through_rate", "mean"),
    monthly_total_units_sold = ("units_sold", "sum"),
).reset_index()

df["year"]  = df.Date.dt.year
df["month"] = df.Date.dt.month
df = df.merge(monthly_inv, on=["year","month"], how="left")
```

---

### 1.8 Nhóm G — Promotion Features

> Nguồn: `promotions.csv` — tính theo từng ngày

```python
def get_active_promos(date, df_promos):
    active = df_promos[(df_promos.start_date <= date) & (df_promos.end_date >= date)]
    return pd.Series({
        "active_promo_count":    len(active),
        "avg_discount_value":    active.discount_value.mean() if len(active) > 0 else 0,
        "has_pct_promo":         int((active.promo_type == "percentage").any()),
        "has_fixed_promo":       int((active.promo_type == "fixed").any()),
    })

promo_features = df.Date.apply(lambda d: get_active_promos(d, df_promotions))
df = pd.concat([df, promo_features], axis=1)
```

---

## Bước 2 — Chia Train / Validation

```python
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END   = "2022-12-31"

# Drop NaN từ các lag features đầu chuỗi
df = df.dropna(subset=["revenue_lag_365"])  # lag lớn nhất → đảm bảo tất cả lag đều có

FEATURES = [
    # Time
    "year","month","day","day_of_week","day_of_year","week_of_year","quarter",
    "is_weekend","is_month_end","is_month_start","is_year_end","is_year_start",
    "sin_month","cos_month","sin_dow","cos_dow",
    # Lag
    "revenue_lag_1","revenue_lag_7","revenue_lag_14","revenue_lag_30",
    "revenue_lag_90","revenue_lag_365",
    "cogs_lag_1","cogs_lag_7","cogs_lag_30","cogs_lag_365",
    # Rolling
    "revenue_roll_mean_7","revenue_roll_mean_14","revenue_roll_mean_30","revenue_roll_mean_90",
    "revenue_roll_std_7","revenue_roll_std_30",
    "cogs_roll_mean_7","cogs_roll_mean_30",
    "revenue_ewm_7","revenue_ewm_30",
    "revenue_diff_1","revenue_diff_7","revenue_pct_change_7",
    # Seasonal
    "is_tet_period","days_to_tet","is_1111","is_1212","is_black_friday","is_christmas",
    # Transaction
    "daily_order_count","daily_cancel_rate","daily_items_sold",
    "daily_discount_total","daily_promo_rate","daily_avg_payment",
    "daily_return_count","daily_avg_rating",
    # Web
    "daily_sessions","daily_unique_visitors","daily_avg_bounce_rate",
    # Inventory
    "monthly_total_stock","monthly_avg_fill_rate","monthly_avg_sell_through",
    # Promo
    "active_promo_count","avg_discount_value","has_pct_promo","has_fixed_promo",
]

TARGETS = ["Revenue", "COGS"]

df_train = df[df.Date <= TRAIN_END]
df_val   = df[(df.Date >= VAL_START) & (df.Date <= VAL_END)]

X_train, y_train = df_train[FEATURES], df_train[TARGETS]
X_val,   y_val   = df_val[FEATURES],   df_val[TARGETS]
```

---

## Bước 3 — Train Ensemble Model

Dùng kiến trúc **Stacking Ensemble** gồm 3 base models + 1 meta-model:

```
Base Models:
├── LightGBM (LGB)    ← mạnh nhất với tabular + time-series
├── XGBoost           ← bổ trợ LGB, bắt được pattern khác
└── Ridge Regression  ← tuyến tính, ổn định, không overfit

Meta-model (Level 2):
└── Ridge Regression  ← kết hợp predictions của 3 base models
```

### 3.1 Tinh chỉnh tham số với Optuna

> Train riêng cho **Revenue** và **COGS** (2 target riêng biệt)

```python
import optuna
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def objective_lgb(trial, X_tr, y_tr, X_val, y_val):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
        "max_depth":        trial.suggest_int("max_depth", 4, 12),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "random_state": 42,
        "verbosity": -1,
    }
    model = LGBMRegressor(**params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)

# Chạy Optuna (Revenue)
study_lgb_rev = optuna.create_study(direction="minimize")
study_lgb_rev.optimize(
    lambda trial: objective_lgb(trial, X_train["Revenue"], y_train["Revenue"], ...),
    n_trials=100,
    show_progress_bar=True
)
best_lgb_rev_params = study_lgb_rev.best_params

# Tương tự cho XGBoost và COGS
```

**Tham số Optuna cho XGBoost:**
```python
params_xgb = {
    "n_estimators":      trial.suggest_int("n_estimators", 500, 3000),
    "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
    "max_depth":         trial.suggest_int("max_depth", 4, 10),
    "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
    "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
    "gamma":             trial.suggest_float("gamma", 0, 5),
    "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
    "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
    "random_state": 42,
}
```

**Tham số Optuna cho Ridge:**
```python
params_ridge = {
    "alpha": trial.suggest_float("alpha", 0.01, 1000, log=True),
}
```

---

### 3.2 Cross-Validation Time Series

> Dùng `TimeSeriesSplit` thay vì KFold thông thường để tránh data leakage

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=30)  # gap 30 ngày giữa train và val mỗi fold
```

```
Fold 1: [2012-2016 train] [gap 30d] [2017 val]
Fold 2: [2012-2017 train] [gap 30d] [2018 val]
Fold 3: [2012-2018 train] [gap 30d] [2019 val]
Fold 4: [2012-2019 train] [gap 30d] [2020 val]
Fold 5: [2012-2020 train] [gap 30d] [2021 val]
→ Final eval: val trên năm 2022
```

---

### 3.3 Stacking Ensemble

```python
from sklearn.linear_model import Ridge
import numpy as np

# Level 1: predictions trên val set
pred_lgb_val   = lgb_model.predict(X_val)    # LGB
pred_xgb_val   = xgb_model.predict(X_val)    # XGBoost
pred_ridge_val = ridge_model.predict(X_val)  # Ridge

# Stack predictions làm input cho meta-model
meta_X_val = np.column_stack([pred_lgb_val, pred_xgb_val, pred_ridge_val])

# Level 2: Ridge meta-model
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X_val, y_val)

# Final prediction
final_pred_val = meta_model.predict(meta_X_val)
```

---

### 3.4 Đánh giá trên Validation Set (2022)

```python
def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2   = r2_score(y_true, y_pred)
    print(f"[{label}] MAE={mae:,.0f} | RMSE={rmse:,.0f} | R²={r2:.4f}")
    return mae, rmse, r2

evaluate(y_val["Revenue"], final_pred_val_revenue, "Revenue - Ensemble")
evaluate(y_val["COGS"],    final_pred_val_cogs,    "COGS - Ensemble")
```

**Ngưỡng chấp nhận (tham khảo):**
| Chỉ số | Mục tiêu |
|--------|---------|
| MAE | < 5% của mean(Revenue) |
| RMSE | < 10% của mean(Revenue) |
| R² | > 0.90 |

---

## Bước 4 — Retrain trên Toàn bộ 2012–2022

```python
FULL_END = "2022-12-31"
df_full  = df[df.Date <= FULL_END].dropna(subset=["revenue_lag_365"])

X_full = df_full[FEATURES]
y_full = df_full[TARGETS]

# Retrain với best params tìm được ở bước 3
lgb_final_rev   = LGBMRegressor(**best_lgb_rev_params, random_state=42)
xgb_final_rev   = XGBRegressor(**best_xgb_rev_params, random_state=42)
ridge_final_rev = Ridge(**best_ridge_rev_params)

lgb_final_rev.fit(X_full, y_full["Revenue"])
xgb_final_rev.fit(X_full, y_full["Revenue"])
ridge_final_rev.fit(X_full, y_full["Revenue"])

# Tương tự cho COGS
```

---

## Bước 5 — Dự báo và Xuất submission.csv

### 5.1 Tạo feature cho tập test (Recursive Forecasting)

```python
# Load sample_submission để lấy đúng thứ tự ngày
df_sub = pd.read_csv("Data/sample_submission.csv", parse_dates=["Date"])
test_dates = df_sub["Date"].values  # 2023-01-01 → 2024-07-01

# Bắt đầu từ full dataframe (đã có đủ lịch sử đến 2022-12-31)
df_extended = df_full.copy()

predictions_rev  = []
predictions_cogs = []

for date in test_dates:
    # 1. Build features cho ngày 'date' từ df_extended (chỉ dùng dữ liệu < date)
    row_features = build_features_for_date(date, df_extended)

    # 2. Predict bằng ensemble
    pred_lgb   = lgb_final_rev.predict([row_features])[0]
    pred_xgb   = xgb_final_rev.predict([row_features])[0]
    pred_ridge = ridge_final_rev.predict([row_features])[0]

    pred_rev = meta_model_rev.predict([[pred_lgb, pred_xgb, pred_ridge]])[0]

    # Tương tự cho COGS
    pred_cogs = ...

    predictions_rev.append(pred_rev)
    predictions_cogs.append(pred_cogs)

    # 3. Append kết quả dự báo vào df_extended để dùng làm lag cho ngày tiếp theo
    new_row = {"Date": date, "Revenue": pred_rev, "COGS": pred_cogs, ...}
    df_extended = pd.concat([df_extended, pd.DataFrame([new_row])], ignore_index=True)
```

### 5.2 Xuất submission.csv

```python
df_submission = pd.DataFrame({
    "Date":    [d.strftime("%Y-%m-%d") for d in test_dates],
    "Revenue": predictions_rev,
    "COGS":    predictions_cogs,
})

# Đảm bảo đúng thứ tự như sample_submission
assert len(df_submission) == len(df_sub), "Số dòng không khớp!"
assert list(df_submission["Date"]) == list(df_sub["Date"].dt.strftime("%Y-%m-%d")), "Thứ tự ngày không khớp!"

df_submission.to_csv("submission.csv", index=False)
print("✅ Đã xuất submission.csv")
print(df_submission.head())
```

---

## Tóm tắt Files cần tạo

| File | Nội dung |
|------|---------|
| `feature_engineering.py` | Toàn bộ logic build features (Nhóm A–G) |
| `model.py` | Train base models, Optuna tuning, stacking ensemble |
| `predict.py` | Recursive forecasting + xuất submission.csv |
| `submission.csv` | File nộp Kaggle cuối cùng |

---

## Checklist trước khi nộp

- [ ] Tất cả lag features dùng `.shift(1)` trước khi rolling
- [ ] Không có dữ liệu 2023-2024 trong bất kỳ feature nào của tập train
- [ ] `submission.csv` có đúng 549 dòng (1 header + 548 ngày)
- [ ] Thứ tự ngày trong submission.csv khớp với sample_submission.csv
- [ ] `random_state=42` được đặt cho tất cả model
- [ ] Đã test lại toàn bộ pipeline từ đầu bằng cách chạy lại trên máy sạch
