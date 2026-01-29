import os
import shutil
from typing import List, Optional
from pathlib import Path
import logging

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from PIL import Image

import base64
from io import BytesIO
from app.inference import predict, dummy_extract_text
from .database import Base, engine, get_db
from .models import ImageItem
from .schemas import ImageOut, AnalyzeResponse, AnalyzeBase64Request
from .Model_Word import predict_word, predict_word_from_pil

load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()

app = FastAPI(
    title="مُبصر - نظام لغة الإشارة",
    description="API لتحليل لغة الإشارة العربية (حروف وكلمات)",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إنشاء الجداول
Base.metadata.create_all(bind=engine)

# مجلد الرفع
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_upload_file(upload_file: UploadFile, dest_dir: str) -> str:
    """حفظ ملف مع التحقق"""
    filename = upload_file.filename
    dest_path = os.path.join(dest_dir, filename)

    base, ext = os.path.splitext(filename)
    i = 1
    while os.path.exists(dest_path):
        filename = f"{base}_{i}{ext}"
        dest_path = os.path.join(dest_dir, filename)
        i += 1

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    try:
        with Image.open(dest_path) as im:
            im.verify()
    except Exception:
        try:
            os.remove(dest_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="ملف صورة غير صالح")

    return dest_path


# ==================== ENDPOINTS ====================

@app.get("/")
def root():
    """الصفحة الرئيسية"""
    return {
        "message": "مرحباً بك في مُبصر - نظام لغة الإشارة العربية",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "upload_letter": "POST /images",
            "upload_word": "POST /images_word",
            "analyze_letter": "POST /analyze",
            "analyze_word": "POST /analyze_word",
            "list_images": "GET /images",
            "get_image": "GET /images/{id}",
            "delete_image": "DELETE /images/{id}"
        }
    }


@app.get("/health")
def health():
    """فحص صحة الخادم"""
    return {"status": "ok", "message": "الخادم يعمل بنجاح"}


@app.post("/images", response_model=ImageOut)
async def upload_letter_image(
    file: UploadFile = File(..., description="ملف صورة الحرف"),
    notes: Optional[str] = Form(None, description="ملاحظات اختيارية"),
    db: Session = Depends(get_db),
):
    """رفع صورة حرف وتحليلها (32 صنف)"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="الملف ليس صورة")

    dest_path = save_upload_file(file, UPLOAD_DIR)
    size_bytes = os.path.getsize(dest_path)
    extracted_text = dummy_extract_text(dest_path)

    item = ImageItem(
        filename=os.path.basename(dest_path),
        content_type=file.content_type,
        size_bytes=size_bytes,
        saved_path=dest_path,
        notes=notes,
        extracted_text=extracted_text,
    )
    db.add(item)
    db.commit()
    db.refresh(item)

    return item


@app.post("/images_word", response_model=ImageOut)
async def upload_word_image(
    file: UploadFile = File(..., description="ملف صورة الكلمة"),
    notes: Optional[str] = Form(None, description="ملاحظات اختيارية"),
    db: Session = Depends(get_db),
):
    """رفع صورة كلمة وتحليلها (89 صنف)"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="الملف ليس صورة")

    dest_path = save_upload_file(file, UPLOAD_DIR)
    size_bytes = os.path.getsize(dest_path)
    
    # استخدام نموذج الكلمات (89 صنف)
    label, conf, top_k = predict_word(dest_path, top_k=5)
    extracted_text = f"{label} (ثقة: {conf:.2%})"

    item = ImageItem(
        filename=os.path.basename(dest_path),
        content_type=file.content_type,
        size_bytes=size_bytes,
        saved_path=dest_path,
        notes=notes,
        extracted_text=extracted_text,
    )
    db.add(item)
    db.commit()
    db.refresh(item)

    return item


@app.get("/images", response_model=List[ImageOut])
def list_images(db: Session = Depends(get_db)):
    """عرض جميع الصور المحفوظة"""
    return db.query(ImageItem).all()


@app.get("/images/{image_id}", response_model=ImageOut)
def get_image(image_id: int, db: Session = Depends(get_db)):
    """الحصول على صورة محددة"""
    item = db.get(ImageItem, image_id)
    if not item:
        raise HTTPException(status_code=404, detail="الصورة غير موجودة")
    return item


@app.delete("/images/{image_id}", status_code=204)
def delete_image(image_id: int, db: Session = Depends(get_db)):
    """حذف صورة"""
    item = db.get(ImageItem, image_id)
    if not item:
        raise HTTPException(status_code=404, detail="غير موجود")
    
    try:
        if os.path.exists(item.saved_path):
            os.remove(item.saved_path)
    except Exception:
        pass

    db.delete(item)
    db.commit()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_letter(
    image: UploadFile = File(..., description="صورة حرف"),
    db: Session = Depends(get_db),
):
    """تحليل حرف (32 صنف) - بدون حفظ دائم"""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="الملف ليس صورة")

    saved_path = save_upload_file(image, UPLOAD_DIR)

    try:
        label, conf = predict(saved_path)

        return AnalyzeResponse(
            label=label,
            confidence=conf,
            text=f"{label} (ثقة: {conf:.2%})"
        )

    finally:
        # حذف الصورة بعد التحليل
        try:
            if os.path.exists(saved_path):
                os.remove(saved_path)
        except Exception:
            pass


@app.post("/analyze_word")
async def analyze_word(
    image: UploadFile = File(..., description="صورة كلمة"),
    use_tta: bool = Form(False, description="استخدام TTA للدقة الأعلى (أبطأ)"),
    db: Session = Depends(get_db),
):
    """
    تحليل كلمة (89 صنف) - مع Top 5 وتحسينات متقدمة
    
    **الميزات:**
    - فحص جودة الصورة تلقائياً
    - دعم TTA (Test Time Augmentation) للدقة الأعلى
    - معالجة محسّنة للصور (تباين، سطوع)
    - كشف MediaPipe Holistic لليدين والوجه
    
    **Parameters:**
    - image: صورة الكلمة (JPEG/PNG)
    - use_tta: تفعيل TTA (يزيد الدقة لكن أبطأ ~4x)
    
    **Returns:**
    - label: الكلمة المتوقعة
    - confidence: نسبة الثقة (0-1)
    - text: نص منسق
    - top_5: أعلى 5 احتمالات
    - quality_warning: تحذير إن كانت جودة الصورة منخفضة
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="الملف ليس صورة")

    saved_path = save_upload_file(image, UPLOAD_DIR)

    try:
        # ✅ 1. فتح الصورة وفحص الجودة
        from .Model_Word import predict_word_from_pil, predict_word_with_tta, check_image_quality
        
        img_pil = Image.open(saved_path).convert("RGB")
        
        # فحص الجودة
        quality_ok = check_image_quality(img_pil)
        quality_warning = None if quality_ok else "⚠️ جودة الصورة منخفضة - قد تؤثر على الدقة"

        # ✅ 2. التنبؤ (مع أو بدون TTA)
        if use_tta:
            logger.info("🔄 استخدام TTA - قد يستغرق وقتاً أطول")
            label, conf, top_k = predict_word_with_tta(img_pil, top_k=5, use_tta=True)
        else:
            label, conf, top_k = predict_word_from_pil(img_pil, top_k=5)

        # ✅ 3. تنسيق الاستجابة
        response = {
            "label": label,
            "confidence": float(conf),
            "text": f"{label} (ثقة: {conf:.2%})",
            "top_5": [
                {
                    "label": lbl,
                    "confidence": float(c),
                    "percentage": f"{c:.2%}"
                }
                for lbl, c in top_k
            ],
            "metadata": {
                "used_tta": use_tta,
                "quality_ok": quality_ok,
                "image_size": f"{img_pil.size[0]}x{img_pil.size[1]}"
            }
        }
        
        if quality_warning:
            response["quality_warning"] = quality_warning

        return response

    except Exception as e:
        logger.error(f"❌ خطأ في تحليل الكلمة: {e}")
        raise HTTPException(status_code=500, detail=f"فشل التحليل: {str(e)}")
        
    finally:
        # حذف الصورة بعد التحليل
        try:
            if os.path.exists(saved_path):
                os.remove(saved_path)
        except Exception:
            pass


@app.post("/analyze_word_batch")
async def analyze_word_batch(
    images: List[UploadFile] = File(..., description="قائمة صور الكلمات"),
    use_tta: bool = Form(False),
    db: Session = Depends(get_db),
):
    """
    تحليل دفعة من الكلمات (Batch Processing)
    
    **مفيد لـ:**
    - رفع عدة صور دفعة واحدة
    - معالجة مجموعات كبيرة
    
    **Limits:**
    - حد أقصى 10 صور في الطلب الواحد
    """
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="الحد الأقصى 10 صور")
    
    results = []
    
    for idx, image in enumerate(images):
        if not image.content_type or not image.content_type.startswith("image/"):
            results.append({
                "index": idx,
                "filename": image.filename,
                "error": "ليس ملف صورة"
            })
            continue
        
        saved_path = save_upload_file(image, UPLOAD_DIR)
        
        try:
            from .Model_Word import predict_word_from_pil, predict_word_with_tta, check_image_quality
            
            img_pil = Image.open(saved_path).convert("RGB")
            quality_ok = check_image_quality(img_pil)
            
            if use_tta:
                label, conf, top_k = predict_word_with_tta(img_pil, top_k=3, use_tta=True)
            else:
                label, conf, top_k = predict_word_from_pil(img_pil, top_k=3)
            
            results.append({
                "index": idx,
                "filename": image.filename,
                "label": label,
                "confidence": float(conf),
                "top_3": [
                    {"label": lbl, "confidence": float(c)}
                    for lbl, c in top_k
                ],
                "quality_ok": quality_ok
            })
            
        except Exception as e:
            results.append({
                "index": idx,
                "filename": image.filename,
                "error": str(e)
            })
        finally:
            try:
                if os.path.exists(saved_path):
                    os.remove(saved_path)
            except Exception:
                pass
    
    return {
        "total": len(images),
        "successful": sum(1 for r in results if "error" not in r),
        "failed": sum(1 for r in results if "error" in r),
        "results": results
    }


# ✅ إضافة endpoint لإحصائيات الصور المحفوظة
@app.get("/images/stats")
def get_images_stats(db: Session = Depends(get_db)):
    """
    إحصائيات الصور المحفوظة
    """
    total = db.query(ImageItem).count()
    
    # تجميع حسب التاريخ
    from sqlalchemy import func, cast, Date
    daily_counts = db.query(
        cast(ImageItem.created_at, Date).label('date'),
        func.count(ImageItem.id).label('count')
    ).group_by('date').order_by('date').all()
    
    # حجم إجمالي
    total_size = db.query(func.sum(ImageItem.size_bytes)).scalar() or 0
    
    return {
        "total_images": total,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "daily_stats": [
            {"date": str(date), "count": count}
            for date, count in daily_counts
        ]
    }

