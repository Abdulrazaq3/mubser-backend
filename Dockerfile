# Dockerfile for Mubser Backend (Railway)
# Python 3.10 with OpenCV-headless and MediaPipe support

FROM python:3.10-slim

# تثبيت مكتبات النظام الضرورية
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# تعيين مجلد العمل
WORKDIR /app

# نسخ ملفات المتطلبات أولاً (للاستفادة من cache)
COPY requirements.txt .

# تثبيت المكتبات البايثونية
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نسخ كافة ملفات التطبيق
COPY . .

# إنشاء مجلد الرفع
RUN mkdir -p uploads

# تعريض المنفذ
EXPOSE 8000

# أمر التشغيل - Railway يمرر PORT كمتغير بيئة
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
