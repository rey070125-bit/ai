FROM python:3.11-slim

# Install Tesseract OCR (and English language data)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT automatically
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5000} app:app"]
