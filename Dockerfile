FROM python:3.11-slim

# 必須パッケージ（ビルド／SSLなどでたまに必要）
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先に requirements だけコピーしてインストール（キャッシュ効かせる）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体をコピー
COPY . .

# FastAPI を立ち上げる
# app.api:app をエントリポイントにしている想定
EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
