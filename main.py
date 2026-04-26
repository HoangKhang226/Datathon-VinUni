"""
main.py
=======
Entry point của dự án.

Chạy lệnh:
    python main.py
"""

from src.train import run

if __name__ == "__main__":
    run(log_file="logs/train.log")
