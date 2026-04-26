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

| Chỉ số                             | Giá trị                                    |
| ---------------------------------- | ------------------------------------------ |
| Q1                                 | 2.47M VNĐ                                  |
| Q3                                 | 5.35M VNĐ                                  |
| IQR                                | 2.88M VNĐ                                  |
| Ngưỡng dưới (Q1 - 1.5×IQR)         | **-1.85M** → không có outlier thấp thực sự |
| Ngưỡng trên (Q3 + 1.5×IQR)         | **9.67M**                                  |
| Số ngày outlier (vượt ngưỡng trên) | **169 / 3833 ngày (4.4%)**                 |

> ⚠️ **Lưu ý:** Ngưỡng dưới âm (-1.85M) có nghĩa là **không tồn tại outlier thấp** — mọi ngày Revenue đều dương và hợp lý. Các outlier 4.4% đều là **spike cao** (ngày lễ lớn), không phải lỗi dữ liệu.

**→ Quyết định:** Dùng **cắm cờ** (không winsorize) — tạo feature `is_high_revenue_day` để model nhận biết các ngày đặc biệt.

---

#### 0.B Log-transform

| Transform | Skewness    | Kurtosis   | Shapiro-W p | Đánh giá                   |
| --------- | ----------- | ---------- | ----------- | -------------------------- |
| Original  | 1.6700      | 4.0303     | 0.0000      | ❌ Lệch nhiều              |
| **log1p** | **-0.1594** | **0.1891** | **0.1999**  | ✅ Gần chuẩn nhất          |
| sqrt      | 0.7404      | 0.7572     | 0.0000      | 🟡 Cải thiện nhưng chưa đủ |

> ✅ **Kết luận: Dùng `log1p(Revenue)` làm target khi train Ridge Regression.**
>
> - Skewness giảm từ 1.67 → -0.16 (gần chuẩn)
> - Shapiro-W p = 0.1999 > 0.05 → không bác bỏ chuẩn
> - Sau khi predict: `np.expm1(pred)` để inverse lại
> - LGB và XGBoost: có thể dùng target gốc hoặc log1p đều được

---

#### 0.C Lag quan trọng từ ACF/PACF

Tổng số lag có ý nghĩa: **ACF = 198 lag, PACF = 109 lag**

| Lag         | ACF        | PACF       | Kết luận                                                |
| ----------- | ---------- | ---------- | ------------------------------------------------------- |
| **lag_1**   | +0.8654 ✅ | +0.8654 ✅ | 🔴 **Ưu tiên cao nhất** — tương quan trực tiếp rất mạnh |
| **lag_2**   | +0.7350 ✅ | -0.0556 ✅ | 🟡 ACF mạnh nhưng PACF gián tiếp                        |
| lag_3       | +0.6214 ✅ | -0.0076 ❌ | Bỏ qua PACF                                             |
| **lag_6**   | +0.4673 ✅ | +0.3778 ✅ | 🟡 Cả hai có ý nghĩa                                    |
| lag_7       | +0.4917 ✅ | +0.0186 ❌ | ACF có nhưng PACF không — ảnh hưởng gián tiếp qua lag_1 |
| **lag_14**  | +0.4956 ✅ | -0.0582 ✅ | 🟡 Giữ lại                                              |
| lag_21      | +0.4356 ✅ | +0.0204 ❌ | Bỏ qua                                                  |
| **lag_365** | +0.7380 ✅ | +0.0431 ✅ | 🔴 **Ưu tiên cao** — mùa vụ năm trước                   |

> ✅ **Lag cần ưu tiên trong feature engineering:** `lag_1`, `lag_2`, `lag_6`, `lag_14`, `lag_365`
> Lag_7 ACF cao nhưng PACF không có ý nghĩa → ảnh hưởng gián tiếp qua lag_1, vẫn nên giữ vì phổ biến trong time-series tuần.

---

#### 0.D Missing Rate

| Bảng                                                                                                         | Kết quả                                  | Quyết định                                                           |
| ------------------------------------------------------------------------------------------------------------ | ---------------------------------------- | -------------------------------------------------------------------- |
| sales, orders, payments, customers, products, inventory, web_traffic, returns, reviews, shipments, geography | ✅ Không missing                         | Merge bình thường                                                    |
| **order_items**                                                                                              | 🔴 `promo_id`: 61.3%, `promo_id_2`: 100% | `promo_id` NaN = không dùng promo (bình thường). `promo_id_2` bỏ hẳn |
| **promotions**                                                                                               | 🔴 `applicable_category`: 80%            | NaN = áp dụng tất cả categories (theo đề bài)                        |

> ✅ **Quyết định:**
>
> - `promo_id` NaN → fill bằng `"no_promo"`, tạo feature `has_promo = (promo_id != "no_promo")`
> - `promo_id_2` → **bỏ hoàn toàn** (100% missing)
> - `applicable_category` NaN → fill `"all"` theo đúng ý nghĩa nghiệp vụ

---

#### 0.E Web Traffic Coverage

| Chỉ số                     | Giá trị             |
| -------------------------- | ------------------- |
| Số ngày trong sales        | 3,833 ngày          |
| Số ngày trong web_traffic  | 3,652 ngày          |
| Missing sessions sau merge | **181 ngày (4.7%)** |

> ⚠️ **181 ngày không có web_traffic** (web_traffic bắt đầu muộn hơn sales hoặc có ngày bị thiếu).
> **→ Xử lý:** Fill NaN bằng `lag_365` (cùng kỳ năm trước). Nếu vẫn NaN, fill bằng median.

---

#### 0.F Tính mùa vụ (từ biểu đồ)

Từ biểu đồ **"Revenue theo Tháng — So sánh từng Năm"**:

| Quan sát          | Chi tiết                                                       |
| ----------------- | -------------------------------------------------------------- |
| **Đỉnh rõ ràng**  | T3–T5 (Quý 1 cuối + Quý 2 đầu) — spike đồng nhất qua các năm   |
| **Đáy rõ ràng**   | T11–T12 — Revenue thấp nhất (ngược với kỳ vọng về 11.11/12.12) |
| **Tính mùa vụ**   | Có, pattern lặp lại rõ qua các năm → cần Fourier features      |
| **Xu hướng tăng** | Revenue tổng thể tăng từ 2012 → 2019, ổn định 2019–2022        |

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

| Feature           | Correlation | Nhận xét                            | Quyết định         |
| ----------------- | ----------- | ----------------------------------- | ------------------ |
| `sessions`        | ~0.32       | Tương quan vừa, ổn định qua các lag | ✅ Giữ làm feature |
| `unique_visitors` | ~0.32       | Tương tự sessions, không cần cả hai | ✅ Giữ 1 trong 2   |
| `page_views`      | ~0.30       | Tương quan vừa                      | ✅ Giữ làm feature |
| `avg_bounce_rate` | ~-0.02      | Gần bằng 0                          | ❌ Bỏ              |
| `avg_session_dur` | ~-0.02      | Gần bằng 0                          | ❌ Bỏ              |

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

| Feature          | Code                            | Ghi chú                 |
| ---------------- | ------------------------------- | ----------------------- |
| `year`           | `df.Date.dt.year`               |                         |
| `month`          | `df.Date.dt.month`              |                         |
| `day`            | `df.Date.dt.day`                |                         |
| `day_of_week`    | `df.Date.dt.dayofweek`          | 0=Thứ 2, 6=Chủ nhật     |
| `day_of_year`    | `df.Date.dt.dayofyear`          |                         |
| `week_of_year`   | `df.Date.dt.isocalendar().week` |                         |
| `quarter`        | `df.Date.dt.quarter`            |                         |
| `is_weekend`     | `dow >= 5`                      |                         |
| `is_month_end`   | `df.Date.dt.is_month_end`       |                         |
| `is_month_start` | `df.Date.dt.is_month_start`     |                         |
| `is_year_end`    | `month==12 & day==31`           |                         |
| `is_year_start`  | `month==1 & day==1`             |                         |
| `sin_month`      | `sin(2π × month / 12)`          | Fourier encoding mùa vụ |
| `cos_month`      | `cos(2π × month / 12)`          | Fourier encoding mùa vụ |
| `sin_dow`        | `sin(2π × dow / 7)`             | Fourier encoding tuần   |
| `cos_dow`        | `cos(2π × dow / 7)`             | Fourier encoding tuần   |

---

### 1.3 Nhóm B — Lag & Rolling Features (từ `sales.csv`)

> Nguồn: `Revenue` và `COGS` của các ngày **trước** ngày hiện tại — an toàn, không leakage

**Lag Features:**

| Feature           | Lag      | Ý nghĩa               |
| ----------------- | -------- | --------------------- |
| `revenue_lag_1`   | 1 ngày   | Doanh thu hôm qua     |
| `revenue_lag_7`   | 7 ngày   | Cùng thứ tuần trước   |
| `revenue_lag_14`  | 14 ngày  | Cùng thứ 2 tuần trước |
| `revenue_lag_30`  | 30 ngày  | Cùng kỳ tháng trước   |
| `revenue_lag_90`  | 90 ngày  | Cùng kỳ quý trước     |
| `revenue_lag_365` | 365 ngày | Cùng ngày năm trước   |
| `cogs_lag_1`      | 1 ngày   |                       |
| `cogs_lag_7`      | 7 ngày   |                       |
| `cogs_lag_30`     | 30 ngày  |                       |
| `cogs_lag_365`    | 365 ngày |                       |

**Rolling Mean / Std Features:**

| Feature                | Window  | Ý nghĩa            |
| ---------------------- | ------- | ------------------ |
| `revenue_roll_mean_7`  | 7 ngày  | Xu hướng ngắn hạn  |
| `revenue_roll_mean_14` | 14 ngày |                    |
| `revenue_roll_mean_30` | 30 ngày | Xu hướng trung hạn |
| `revenue_roll_mean_90` | 90 ngày | Xu hướng dài hạn   |
| `revenue_roll_std_7`   | 7 ngày  | Độ biến động       |
| `revenue_roll_std_30`  | 30 ngày |                    |
| `cogs_roll_mean_7`     | 7 ngày  |                    |
| `cogs_roll_mean_30`    | 30 ngày |                    |

> ⚠️ **Quan trọng:** Phải dùng `.shift(1)` trước khi rolling để đảm bảo không dùng giá trị ngày T khi tính feature cho ngày T.
>
> ```python
> df["revenue_roll_mean_7"] = df["Revenue"].shift(1).rolling(7).mean()
> ```

**Exponential Weighted Mean:**

| Feature          | Span | Ý nghĩa                                        |
| ---------------- | ---- | ---------------------------------------------- |
| `revenue_ewm_7`  | 7    | Trung bình mũ — nhạy hơn với biến động gần đây |
| `revenue_ewm_30` | 30   |                                                |

**Difference Features:**

| Feature                | Công thức                 | Ý nghĩa                     |
| ---------------------- | ------------------------- | --------------------------- |
| `revenue_diff_1`       | `Revenue - lag_1`         | Biến động so với hôm qua    |
| `revenue_diff_7`       | `Revenue - lag_7`         | Biến động so với tuần trước |
| `revenue_pct_change_7` | `(lag_1 - lag_7) / lag_7` | % thay đổi                  |

---

### 1.4 Nhóm C — Seasonal / Holiday Features

> Nguồn: Tự tính từ `Date` — có sẵn cho cả tập test 2023-2024

**Sự kiện thương mại điện tử Việt Nam:**

| Feature             | Điều kiện                                         | Ghi chú                                 |
| ------------------- | ------------------------------------------------- | --------------------------------------- |
| `is_tet_period`     | Khoảng -15 đến +5 ngày so với Tết âm lịch         | Dùng thư viện `holidays` hoặc hard-code |
| `days_to_tet`       | Số ngày đến Tết gần nhất                          | Âm nếu đã qua Tết                       |
| `is_1111`           | `month==11 & day==11`                             | Ngày hội mua sắm                        |
| `is_1212`           | `month==12 & day==12`                             |                                         |
| `is_black_friday`   | Thứ 6 tuần 4 tháng 11                             |                                         |
| `is_mid_year_sale`  | `month==6 & day >= 25` hoặc `month==7 & day <= 5` |                                         |
| `is_back_to_school` | `month==8`                                        |                                         |
| `is_christmas`      | `month==12 & day >= 23`                           |                                         |
| `is_valentines`     | `month==2 & day >= 10 & day <= 14`                |                                         |
| `is_womens_day`     | `month==3 & day==8`                               |                                         |

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

## BƯỚC 2 — TRAIN / VALIDATION SPLIT (CHUẨN HÓA LẠI)

### Mục tiêu

- **Train:** học quy luật quá khứ (2012–2021)
- **Validation:** mô phỏng “tương lai chưa thấy” (2022)
- **Tuyệt đối không leak** thông tin 2022 vào training

---

### Split theo thời gian

```python
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END   = "2022-12-31"
```

---

### Xử lý dữ liệu (quan trọng để tránh leakage từ lag)

```python
df = df.dropna(subset=["revenue_lag_365"])
```

**Ý nghĩa:**

- Lag 365 = cần ít nhất 1 năm dữ liệu quá khứ
- Nếu không drop → model “học giả” (sai time alignment)

---

### Tách dataset

```python
df_train = df[df.Date <= TRAIN_END]
df_val   = df[(df.Date >= VAL_START) & (df.Date <= VAL_END)]
```

---

### Feature set (time encoded tabular features)

| Feature            | Ý nghĩa                                             |
| ------------------ | --------------------------------------------------- |
| `revenue_lag_1`    | doanh thu ngày trước đó (1-day lag)                 |
| `revenue_lag_7`    | doanh thu cách 7 ngày (tuần trước)                  |
| `revenue_lag_365`  | doanh thu cùng kỳ năm trước (seasonality yearly)    |
| `rolling_mean_7`   | trung bình doanh thu 7 ngày gần nhất (smooth trend) |
| `month`            | tháng trong năm (seasonality theo tháng)            |
| `weekday`          | thứ trong tuần (pattern theo ngày)                  |
| `promo_flag`       | có chạy khuyến mãi hay không (0/1 binary feature)   |
| `traffic_sessions` | số lượt truy cập (proxy cho demand / interest)      |

---

### Ý nghĩa quan trọng

Đây **KHÔNG** phải time-series model thuần. Mà là:

- Time-series → transformed into supervised learning dataset
- Không shuffle
- Validation = future simulation
- Feature engineering phải được “freeze trước split”

---

## BƯỚC 3 — STACKING ENSEMBLE (CHUẨN HÓA LOGIC)

### 3.1 Kiến trúc

- **Level 0 (Base Models)**
  - LightGBM
  - XGBoost
  - Ridge
- **Level 1 (Meta Model)**
  - Ridge (blending)

---

### 3.2 Train Base Models

```python
lgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
```

**Ý nghĩa:**

- LGBM: capture nonlinear interactions
- XGB: học residual patterns khác LGB
- Ridge: baseline linear stabilizer

---

### 3.3 Predict validation

```python
pred_lgb = lgb_model.predict(X_val)
pred_xgb = xgb_model.predict(X_val)
pred_ridge = ridge_model.predict(X_val)
```

---

### 3.4 Build meta dataset

```python
meta_X = np.column_stack([
    pred_lgb,
    pred_xgb,
    pred_ridge
])
```

---

### 3.5 Train meta model

```python
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X, y_val)
```

**Ý nghĩa:**
Meta model học: “model nào đáng tin hơn trong từng pattern”

---

## BƯỚC 3.1 — OPTUNA TUNING

### Nguyên tắc

- Tune từng model riêng
- Không mix target
- Không tune trên full data

### Objective chuẩn

```python
def objective(trial):
    model = LGBMRegressor(
        n_estimators=trial.suggest_int(100, 1000),
        learning_rate=trial.suggest_float(0.01, 0.2),
        num_leaves=trial.suggest_int(20, 200),
        max_depth=trial.suggest_int(3, 12),
        subsample=trial.suggest_float(0.6, 1.0),
        colsample_bytree=trial.suggest_float(0.6, 1.0),
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    return mean_absolute_error(y_val, preds)
```

---

## BƯỚC 3.2 — TIME SERIES CV (CHỈ DÙNG ĐỂ CHECK)

```python
tscv = TimeSeriesSplit(n_splits=5, gap=30)
```

**Ý nghĩa đúng:**

- Không dùng để train final
- Chỉ để:
  - kiểm tra stability
  - detect overfitting theo time slice

---

## BƯỚC 3.3 — STACKING PIPELINE CHUẨN

1. Train base models
2. → Predict validation
3. → Train meta model

- [x] KHÔNG mix train/val
- [x] KHÔNG shuffle
- [x] KHÔNG cross contamination

---

## BƯỚC 3.4 — EVALUATION

- MAE
- RMSE
- R²

**Ngưỡng hợp lý:**

- R² > 0.9 → rất tốt cho revenue forecasting
- MAE giảm ổn định qua CV → model robust

---

## BƯỚC 4 — RETRAIN FINAL MODEL (QUAN TRỌNG NHẤT)

### Mục tiêu

Tận dụng toàn bộ data (2012–2022) để tối đa hóa learning

---

### Full dataset

```python
df_full = df[df.Date <= "2022-12-31"].dropna(subset=["revenue_lag_365"])

X_full = df_full[FEATURES]
y_full = df_full[TARGETS]
```

---

### Train final models

```python
lgb_final = LGBMRegressor(**best_lgb_params)
xgb_final = XGBRegressor(**best_xgb_params)
ridge_final = Ridge(**best_ridge_params)

lgb_final.fit(X_full, y_full["Revenue"])
xgb_final.fit(X_full, y_full["Revenue"])
ridge_final.fit(X_full, y_full["Revenue"])
```

---

### Meta model (có 2 lựa chọn)

- **Option A:** giữ meta model từ validation
- **Option B (chuẩn hơn):** retrain meta model trên toàn history predictions

---

## PIPELINE CUỐI CÙNG

### FULL TRAIN PIPELINE

1. Time split (Train: 2012–2021, Val: 2022)
2. Feature engineering (freeze trước split)
3. Train base models (LGB / XGB / Ridge)
4. Predict validation
5. Train meta model (stacking layer)
6. Hyperparameter tuning (Optuna)
7. Retrain full dataset (2012–2022)
8. Freeze model
9. Inference on test set (2023–2024)

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

| File                     | Nội dung                                            |
| ------------------------ | --------------------------------------------------- |
| `feature_engineering.py` | Toàn bộ logic build features (Nhóm A–G)             |
| `model.py`               | Train base models, Optuna tuning, stacking ensemble |
| `predict.py`             | Recursive forecasting + xuất submission.csv         |
| `submission.csv`         | File nộp Kaggle cuối cùng                           |

---

## Checklist trước khi nộp

- [ ] Tất cả lag features dùng `.shift(1)` trước khi rolling
- [ ] Không có dữ liệu 2023-2024 trong bất kỳ feature nào của tập train
- [ ] `submission.csv` có đúng 549 dòng (1 header + 548 ngày)
- [ ] Thứ tự ngày trong submission.csv khớp với sample_submission.csv
- [ ] `random_state=42` được đặt cho tất cả model
- [ ] Đã test lại toàn bộ pipeline từ đầu bằng cách chạy lại trên máy sạch
