# Datathon 2026 - Revenue Forecasting Pipeline

Hệ thống dự báo Revenue & COGS sử dụng mô hình **Stacking Ensemble** (LGBM, XGBoost, Ridge) được chuẩn hóa cho môi trường production.

## Cấu trúc Dự án

```text
Datathon-VinUni/
 ├── data/                # Chứa các file CSV đầu vào (sales.csv, orders.csv, ...)
 ├── src/
 │   ├── data_loader.py   # Load dữ liệu từ CSV
 │   ├── feature_engineering.py # Tạo đặc trưng (Lags, Window, Seasonal, Tet, ...)
 │   ├── train.py         # Orchestrator điều phối toàn bộ pipeline (8 bước)
 │   ├── models.py        # Định nghĩa các model Level 0 và Level 1
 │   ├── cv.py            # Logic Optuna tuning và OOF predictions (Cross-Validation)
 │   └── utils.py         # Tiện ích (Logging, Evaluation, SHAP plots)
 ├── models/              # Lưu trữ artifact model trained (.pkl)
 ├── logs/                # Lưu trữ file log quá trình huấn luyện
 ├── notebooks/           # Jupyter Notebook dùng cho EDA và thử nghiệm
 ├── main.py              # Entry point duy nhất để chạy toàn bộ hệ thống
 └── requirements.txt     # Danh sách thư viện phụ thuộc
```

## Luồng xử lý (Workflow)

Hệ thống được thiết kế theo 8 bước tự động hóa trong `main.py` -> `src/train.py`:

1.  **Load Data**: Khởi tạo `DataLoader` để đọc các bảng dữ liệu từ `Data/`.
2.  **Feature Engineering**: `FeatureEngineer` xử lý logic tạo lags, rolling, web stats, inventory...
3.  **Split Data**: Chia Train (đến hết 2021) và Validation (năm 2022) theo nguyên tắc chuỗi thời gian.
4.  **Tune Hyperparameters**:
    - Sử dụng **Optuna** để tìm thông số tốt nhất.
    - Áp dụng **5-fold TimeSeriesSplit** nội bộ để đánh giá trial (tránh leakage).
5.  **Generate OOF**:
    - Tạo dự báo Out-of-Fold cho bộ Train bằng bộ tham số tốt nhất.
    - Đảm bảo `StandardScaler` được fit-transform độc lập mỗi fold.
6.  **Train Meta Model**: Huấn luyện Ridge Meta Model trên các dự báo OOF của LGBM, XGB và Ridge Level 0.
7.  **Evaluate**: Đánh giá kết quả cuối cùng (MAE, RMSE, R2) trên bộ Validation 2022.
8.  **SHAP & Save**:
    - Tạo biểu đồ giải thích mô hình bằng **SHAP summary plot**.
    - Log toàn bộ tham số, metrics và biểu đồ vào **MLflow**.
    - Lưu model final vào thư mục `models/`.

## Hướng dẫn Cài đặt & Chạy

### 1. Setup virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Chạy Pipeline

```powershell
python main.py
```

### 3. Xem kết quả bằng MLflow

Sau khi chạy xong, bạn có thể xem chi tiết các trial và biểu đồ giải thích bằng lệnh:

```powershell
mlflow ui
```

Sau đó truy cập `http://localhost:5000` trên trình duyệt.
