"""
src/data_loader.py
==================
Load và quản lý toàn bộ các bảng dữ liệu của Datathon 2026.
(Được trích xuất từ logic trong feature_engineering.py để tuân thủ cấu trúc thư mục)
"""

import pandas as pd
import logging
from pathlib import Path

DATA_DIR = Path("Data")
log = logging.getLogger(__name__)

class DataLoader:
    """
    Load và quản lý toàn bộ các bảng dữ liệu của Datathon 2026.

    Attributes
    ----------
    sales        : pd.DataFrame  — Target chính (Date, Revenue, COGS)
    orders       : pd.DataFrame  — Thông tin đơn hàng
    order_items  : pd.DataFrame  — Chi tiết từng dòng sản phẩm trong đơn
    payments     : pd.DataFrame  — Thông tin thanh toán
    customers    : pd.DataFrame  — Thông tin khách hàng
    products     : pd.DataFrame  — Danh mục sản phẩm
    promotions   : pd.DataFrame  — Chương trình khuyến mãi
    inventory    : pd.DataFrame  — Tồn kho cuối tháng
    web_traffic  : pd.DataFrame  — Lưu lượng truy cập website hàng ngày
    returns      : pd.DataFrame  — Các sản phẩm bị trả lại
    reviews      : pd.DataFrame  — Đánh giá sản phẩm
    shipments    : pd.DataFrame  — Thông tin vận chuyển
    geography    : pd.DataFrame  — Danh sách mã bưu chính các vùng
    """

    # Mapping: attribute_name -> (filename, [date_columns])
    _FILES = {
        "sales":       ("sales.csv",       ["Date"]),
        "orders":      ("orders.csv",      ["order_date"]),
        "order_items": ("order_items.csv", []),
        "payments":    ("payments.csv",    []),
        "customers":   ("customers.csv",   ["signup_date"]),
        "products":    ("products.csv",    []),
        "promotions":  ("promotions.csv",  ["start_date", "end_date"]),
        "inventory":   ("inventory.csv",   ["snapshot_date"]),
        "web_traffic": ("web_traffic.csv", ["date"]),
        "returns":     ("returns.csv",     ["return_date"]),
        "reviews":     ("reviews.csv",     ["review_date"]),
        "shipments":   ("shipments.csv",   ["ship_date", "delivery_date"]),
        "geography":   ("geography.csv",   []),
    }

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self._load_all()

    def _load_all(self):
        """Đọc tất cả file CSV và gắn vào attributes."""
        for attr, (filename, date_cols) in self._FILES.items():
            filepath = self.data_dir / filename
            if not filepath.exists():
                log.warning(f"Không tìm thấy: {filepath}")
                setattr(self, attr, None)
                continue
            kwargs = {"parse_dates": date_cols} if date_cols else {}
            df = pd.read_csv(filepath, **kwargs)
            setattr(self, attr, df)
            log.info(f"Loaded {filename:25s} — {len(df):>8,} dòng | {df.shape[1]} cột")

    def summary(self):
        """In tóm tắt shape và số cột có missing của tất cả bảng."""
        log.info(f"\n{'Bảng':<15} {'Dòng':>8} {'Cột':>5} {'Missing cols':>15}")
        log.info("-" * 50)
        for attr in self._FILES:
            df = getattr(self, attr)
            if df is None:
                log.info(f"{attr:<15} {'N/A':>8}")
                continue
            missing_cols = (df.isnull().mean() > 0).sum()
            log.info(f"{attr:<15} {len(df):>8,} {df.shape[1]:>5} {missing_cols:>15}")

    def __repr__(self):
        loaded = [k for k in self._FILES if getattr(self, k) is not None]
        return f"DataLoader(data_dir='{self.data_dir}', tables={loaded})"
