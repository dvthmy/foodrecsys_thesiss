FROM python:3.13-slim

# 1. Cài đặt uv (Lấy trực tiếp từ image chính chủ của Astral)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
# Copy thêm uv.lock nếu bạn muốn cài đặt chính xác version đã lock (Rất khuyến khích)
COPY uv.lock . 

# 2. Cài đặt bằng uv thay vì pip
# --system: Cài vào python hệ thống của Docker (không cần tạo venv ảo)
# --no-cache: Giảm dung lượng image (tùy chọn)
RUN uv pip install --system --no-cache -e .

COPY README.md .
COPY src/ src/
COPY main.py .
COPY *.csv .
COPY *.json .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /tmp/food-recsys/uploads

EXPOSE 8000
EXPOSE 8501