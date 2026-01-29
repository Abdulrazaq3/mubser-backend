# Dockerfile for Mubser Backend (Railway)
# Python 3.10 with OpenCV-headless and MediaPipe support

FROM python:3.10-slim

# تثبيت مكتبات النظام الضرورية (متوافق مع Debian Trixie)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
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

# إنشاء مجلد الرفع + جعل السكربت قابل للتنفيذ
RUN mkdir -p uploads && chmod +x start.sh

# تعريض المنفذ
EXPOSE 8000

# تعيين PORT افتراضي (Railway سيتجاوزه)
ENV PORT=8000

# أمر التشغيل باستخدام السكربت
CMD ["./start.sh"]
