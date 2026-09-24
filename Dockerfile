FROM python:3.10-slim

WORKDIR /app

# onnxruntime / opencv 所需的系统运行库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，充分利用镜像层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制运行所需代码与模型权重
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
