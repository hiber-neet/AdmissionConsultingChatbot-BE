
FROM python:3.12-slim

# 2. Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Thêm đường dẫn hiện tại vào PYTHONPATH để import module dễ dàng
    PYTHONPATH=/code

# 3. Cài đặt thư viện hệ thống cần thiết (cho psycopg2 - Postgres)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Thiết lập thư mục làm việc
WORKDIR /code
# Python 3.12 cần setuptools mới nhất để cài các thư viện cũ (nếu có)
RUN pip install --upgrade pip setuptools wheel
# 5. Copy requirements và cài đặt dependencies trước (để tận dụng cache của Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy toàn bộ source code vào container
COPY . .

# 7. Tạo User Non-root (Bảo mật)
# Tạo user 'appuser', và cấp quyền sở hữu thư mục /code
RUN adduser -u 5678 --disabled-password --gecos "" appuser \
    && chown -R appuser /code

# --- Xử lý riêng cho thư mục uploads ---
# Đảm bảo thư mục uploads tồn tại và appuser có quyền ghi vào đó
RUN mkdir -p /code/uploads && chown -R appuser /code/uploads

# 8. Chuyển sang user thường
USER appuser

# 9. Mở port
EXPOSE 8000

# 10. Lệnh chạy Production (Gunicorn quản lý Uvicorn workers)
# Lưu ý: "app.main:app" dựa trên cấu trúc thư mục của bạn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120", "app.main:app"]