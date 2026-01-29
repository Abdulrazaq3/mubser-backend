# Model_Word.py
"""
تنبؤ كلمات (89 صنف) باستخدام ONNX Runtime + MediaPipe Holistic
- النموذج: Mubser_model_89cls_64.onnx
- الميتاداتا: Mubser_model_89cls_64.meta.json
- إدخال: صورة واحدة (PIL.Image أو مسار)
- إخراج: (label, confidence, top_k_list)
"""

from typing import Tuple, List, Optional, Union
from pathlib import Path
import json
import logging
import atexit

import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
import onnxruntime as ort

# ================= إعداد السجلات =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model_Word")

# ================= إعداد المسارات =================
MODEL_PATH = Path("Mubser_model_89cls_64.onnx")
META_PATH  = Path("Mubser_model_89cls_64.meta.json")

# ================= تحميل الميتاداتا =================
try:
    with META_PATH.open("r", encoding="utf-8") as f:
        _meta = json.load(f)
    CLASSES: List[str] = _meta["classes"]
    IMG_SIZE: int = int(_meta.get("img_size", 64))
    # اختياري: تطبيع (إذا موجود في الميتاداتا)
    _norm = _meta.get("normalize") or {}
    NORM_MEAN: Optional[List[float]] = _norm.get("mean")
    NORM_STD: Optional[List[float]] = _norm.get("std")
    # اختياري: mapping (إن وُجد) لتحويل اللابل لعرض عربي/إنجليزي
    LABEL_MAPPING: Optional[dict] = _meta.get("mapping")
    logger.info(f"✅ Loaded meta: {len(CLASSES)} classes, img_size={IMG_SIZE}")
except Exception as e:
    logger.error(f"❌ Failed to load meta '{META_PATH}': {e}")
    raise


# ================= تهيئة MediaPipe Holistic (وضع fail-safe) =================
mp_holistic = None
_holistic = None

try:
    if hasattr(mp, 'solutions'):
        mp_holistic = mp.solutions.holistic
        _holistic = mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            refine_face_landmarks=False,
            min_detection_confidence=0.6
        )
        logger.info("✅ MediaPipe Holistic initialized successfully")
    else:
        logger.warning("⚠️ MediaPipe solutions not found (Check Python version/Installation)")
except Exception as e:
    logger.warning(f"⚠️ Failed to initialize MediaPipe: {e}")

def _close_holistic():
    try:
        if _holistic:
            _holistic.close()
    except Exception:
        pass
atexit.register(_close_holistic)



# ================= تهيئة ONNX Runtime =================
try:
    _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    _input_name = _session.get_inputs()[0].name
    _output_name = _session.get_outputs()[0].name
    logger.info("✅ ONNX session initialized (CPUExecutionProvider)")
except Exception as e:
    logger.error(f"❌ Failed to initialize ONNX session: {e}")
    raise

# =====================================================
#                وظائف المعالجة/التنبؤ
# =====================================================

def crop_holistic_union_pil(img_pil: Image.Image, pad: int = 20, max_size: int = 640) -> Image.Image:
    """
    يقصّ مستطيلاً واحدًا يضم اليدين + الوجه + الجسم
    """
    if _holistic is None:
        # Fallback مباشر للقص المركزي إذا لم يعمل MediaPipe
        bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_size = min(w, h) * 3 // 4
        x1 = max(0, center_x - crop_size // 2)
        y1 = max(0, center_y - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        crop = bgr[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    try:
        # تصغير الصورة أولاً لتسريع MediaPipe
        w, h = img_pil.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
        
        bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        res = _holistic.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        xs, ys = [], []

        # اليدان
        for hand_lms in [res.left_hand_landmarks, res.right_hand_landmarks]:
            if hand_lms:
                for lm in hand_lms.landmark:
                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))

        # الوجه (نقاط مفتاحية محددة فقط)
        if res.face_landmarks:
            key_face_points = [10, 152, 234, 454, 1]
            lms = res.face_landmarks.landmark
            for k in key_face_points:
                if 0 <= k < len(lms):
                    lm = lms[k]
                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))

        # الجسم (نقاط أساسية)
        if res.pose_landmarks:
            pose_ids = [0, 11, 12, 23, 24]  # رأس، كتفين، وركين
            lms = res.pose_landmarks.landmark
            for k in pose_ids:
                if 0 <= k < len(lms):
                    lm = lms[k]
                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))

        if not xs:
            logger.warning("⚠️ لم يُكتشف إنسان - استخدام crop مركزي")
            # ✅ fallback: قص مركزي بدلاً من الصورة الكاملة
            center_x, center_y = w // 2, h // 2
            crop_size = min(w, h) * 3 // 4
            x1 = max(0, center_x - crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)
            crop = bgr[y1:y2, x1:x2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

        x1, x2 = max(0, min(xs) - pad), min(w, max(xs) + pad)
        y1, y2 = max(0, min(ys) - pad), min(h, max(ys) + pad)
        
        if x2 <= x1 or y2 <= y1:
            logger.warning("⚠️ صندوق غير صالح")
            return img_pil

        crop = bgr[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
        
    except Exception as e:
        logger.error(f"❌ خطأ في القص: {e}")
        # fallback آمن
        return img_pil.resize((IMG_SIZE, IMG_SIZE))

def _apply_optional_normalize(x: np.ndarray) -> np.ndarray:
    """
    يطبّق Normalize (اختياري) إذا تم تعريفه في الميتاداتا
    - x شكلها (1,1,H,W) وقيمها [0..1]
    """
    if NORM_MEAN and NORM_STD and len(NORM_MEAN) >= 1 and len(NORM_STD) >= 1:
        mean = float(NORM_MEAN[0])
        std = float(NORM_STD[0]) if float(NORM_STD[0]) != 0 else 1.0
        x = (x - mean) / std
    return x

def preprocess_pil(img_pil: Image.Image, enhance: bool = True) -> np.ndarray:
    """
    pipeline معالجة محسّن
    """
    # 1. قص
    img_pil = crop_holistic_union_pil(img_pil, pad=20)

    # 2. ✅ تحسين الصورة (اختياري)
    if enhance:
        from PIL import ImageEnhance
        # زيادة التباين قليلاً
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(1.2)
        # ضبط السطوع
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(1.1)

    # 3. رمادي + تغيير الحجم
    img_pil = img_pil.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # 4. إلى مصفوفة
    x = np.array(img_pil, dtype=np.float32) / 255.0

    # 5. ✅ Histogram Equalization (اختياري للإضاءة السيئة)
    # x = cv2.equalizeHist((x * 255).astype(np.uint8)) / 255.0

    # 6. Normalize من الميتاداتا
    x = _apply_optional_normalize(x)

    # 7. إعادة تشكيل
    x = x[None, None, :, :]
    return x

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / np.sum(ez)

def predict_word_from_pil(img_pil: Image.Image, top_k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    تنبؤ من PIL.Image
    يرجّع: (label, confidence, top_k_list)
    """
    # تحضير الإدخال
    x = preprocess_pil(img_pil)

    # تشغيل النموذج
    outputs = _session.run([_output_name], {_input_name: x})
    logits = outputs[0][0]  # (num_classes,)

    probs = _softmax(logits)
    top_idx = int(np.argmax(probs))
    top_label = CLASSES[top_idx]
    top_conf = float(probs[top_idx])

    # أعلى k
    k = int(max(1, min(top_k, len(CLASSES))))
    top_k_indices = np.argsort(probs)[-k:][::-1]
    top_k_list = [(CLASSES[i], float(probs[i])) for i in top_k_indices]

    # إذا عندنا mapping في الميتاداتا، نقدر نرفق التسمية العربية المقابلة
    if LABEL_MAPPING and top_label in LABEL_MAPPING:
        mapped = LABEL_MAPPING[top_label]
        top_label = f"{top_label} | {mapped}"

        # نطبّق نفس الشيء على top_k_list
        new_top = []
        for k_lbl, k_p in top_k_list:
            if k_lbl in LABEL_MAPPING:
                new_top.append((f"{k_lbl} | {LABEL_MAPPING[k_lbl]}", k_p))
            else:
                new_top.append((k_lbl, k_p))
        top_k_list = new_top

    return top_label, top_conf, top_k_list

def check_image_quality(img_pil: Image.Image) -> bool:
    """
    فحص جودة الصورة قبل المعالجة
    """
    w, h = img_pil.size
    
    # 1. حجم أدنى
    if w < 100 or h < 100:
        logger.warning(f"⚠️ الصورة صغيرة جداً: {w}x{h}")
        return False
    
    # 2. فحص الضبابية (Laplacian variance)
    gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < 50:  # عتبة تجريبية
        logger.warning(f"⚠️ الصورة ضبابية: variance={laplacian_var:.2f}")
        return False
    
    return True

def predict_word(image: Union[str, Path, Image.Image], top_k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    تنبؤ مع فحص الجودة
    """
    if isinstance(image, (str, Path)):
        img_pil = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img_pil = image
    else:
        raise TypeError("image يجب أن يكون مساراً أو PIL.Image")
    
    # ✅ فحص الجودة
    if not check_image_quality(img_pil):
        logger.warning("⚠️ جودة الصورة منخفضة - النتيجة قد لا تكون دقيقة")
    
    return predict_word_from_pil(img_pil, top_k=top_k)

def predict_word_with_tta(
    img_pil: Image.Image, 
    top_k: int = 5,
    use_tta: bool = False
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    تنبؤ مع TTA (اختياري)
    """
    if not use_tta:
        return predict_word_from_pil(img_pil, top_k)
    
    # ✅ Test Time Augmentation
    predictions = []
    
    # 1. الأصلية
    predictions.append(predict_word_from_pil(img_pil, top_k))
    
    # 2. انعكاس أفقي
    flipped = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    predictions.append(predict_word_from_pil(flipped, top_k))
    
    # 3. دوران طفيف
    for angle in [-5, 5]:
        rotated = img_pil.rotate(angle, fillcolor='white')
        predictions.append(predict_word_from_pil(rotated, top_k))
    
    # ✅ دمج النتائج (voting)
    label_votes = {}
    for label, conf, _ in predictions:
        label_votes[label] = label_votes.get(label, 0) + conf
    
    # أعلى تصويت
    best_label = max(label_votes, key=label_votes.get)
    avg_conf = label_votes[best_label] / len(predictions)
    
    # إعادة الـ top_k
    sorted_labels = sorted(label_votes.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_k_list = [(lbl, conf / len(predictions)) for lbl, conf in sorted_labels]
    
    return best_label, avg_conf, top_k_list

# اختياري: دالة بسيطة تُرجع نصاً فقط (للتوافق مع بعض الواجهات)
def dummy_extract_text(image: Union[str, Path, Image.Image]) -> str:
    try:
        label, conf, _ = predict_word(image, top_k=5)
        return f"{label} ({conf:.2f})"
    except Exception as e:
        return f"prediction_failed: {e}"
