"""
نموذج التنبؤ بلغة الإشارة العربية باستخدام ONNX Runtime و MediaPipe

✅ تم تحديثه لاستخدام Lazy Loading لتسريع startup السيرفر
   - cv2, mediapipe, onnxruntime يتم تحميلها عند الحاجة فقط
"""
import json
from pathlib import Path
from typing import Tuple, Optional, List
import logging

import numpy as np
from PIL import Image
# ✅ تأخير imports الثقيلة - سيتم تحميلها داخل SignLanguagePredictor

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignLanguagePredictor:
    """
    كلاس للتنبؤ بلغة الإشارة العربية من الصور
    """
    
    def __init__(
        self,
        model_path: str = "Mubser_model.onnx",
        metadata_path: str = "Mubser_model.meta.json",
        providers: List[str] = None
    ):
        """
        تهيئة المتنبئ
        """
        # ✅ تحميل المكتبات الثقيلة عند الحاجة فقط
        import onnxruntime as ort
        import mediapipe as mp
        self._cv2 = None  # سيتم تحميله عند الحاجة
        self._mp = mp
        
        # تحميل الميتاداتا + المابنق
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.classes = metadata["classes"]
                self.img_size = metadata["img_size"]
                self.mapping = metadata.get("mapping", {})  # ✅ هنا التحميل فقط
            logger.info(f"✅ تم تحميل {len(self.classes)} صنف من الميتاداتا")
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل الميتاداتا: {e}")
            raise
        
        # تحميل نموذج ONNX
        try:
            if providers is None:
                providers = ["CPUExecutionProvider"]
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            logger.info(f"✅ تم تحميل النموذج باستخدام: {providers}")
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل النموذج: {e}")
            raise
        
        # تهيئة MediaPipe (Fail-safe)
        self.mp_hands = None
        try:
            if hasattr(mp, 'solutions'):
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("✅ تم تهيئة MediaPipe")
            else:
                logger.warning("⚠️ MediaPipe solutions غير موجود (تأكد من توافق إصدار Python)")
        except Exception as e:
            logger.warning(f"⚠️ فشل تهيئة MediaPipe (سيتم استخدام القص المركزي): {e}")
    
    def _get_cv2(self):
        """تحميل cv2 عند الحاجة"""
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2
    
    def detect_hand_box(self, bgr: np.ndarray, pad: int = 20) -> Optional[Tuple[int, int, int, int]]:
        cv2 = self._get_cv2()
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        try:
            if self.mp_hands is None:
                return None
                
            results = self.mp_hands.process(rgb)
            if not results.multi_hand_landmarks:
                return None
            
            landmarks = results.multi_hand_landmarks[0].landmark
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]
            
            x1 = max(0, min(xs) - pad)
            y1 = max(0, min(ys) - pad)
            x2 = min(w, max(xs) + pad)
            y2 = min(h, max(ys) + pad)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            return (x1, y1, x2, y2)
        except:
            return None
    
    def crop_hand(self, img_pil: Image.Image, pad: int = 20) -> Image.Image:
        cv2 = self._get_cv2()
        try:
            bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            box = self.detect_hand_box(bgr, pad=pad)
            if box is None:
                return img_pil
            
            x1, y1, x2, y2 = box
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return img_pil
            
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        except:
            return img_pil
    
    def preprocess(self, img_pil: Image.Image) -> np.ndarray:
        img_pil = self.crop_hand(img_pil, pad=20)
        img_pil = img_pil.convert("L").resize((self.img_size, self.img_size))
        x = np.array(img_pil, dtype=np.float32) / 255.0
        x = x[None, None, :, :]
        return x
    
    def predict(self, image_path: str, top_k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
        img = Image.open(image_path).convert("RGB")
        x = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_name: x})
        logits = outputs[0][0]
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        top_idx = int(np.argmax(probs))
        top_label = self.classes[top_idx]
        top_confidence = float(probs[top_idx])
        
        top_k_indices = np.argsort(probs)[-top_k:][::-1]
        top_k_predictions = [(self.classes[idx], float(probs[idx])) for idx in top_k_indices]
        
        return top_label, top_confidence, top_k_predictions
    
    def __del__(self):
        if hasattr(self, 'mp_hands') and self.mp_hands:
            self.mp_hands.close()


# ====== Lazy Loading Pattern ======
_predictor = None
_predictor_loading = False


def get_predictor():
    """تحميل المتنبئ عند الحاجة فقط (Lazy Loading)"""
    global _predictor, _predictor_loading
    
    if _predictor is not None:
        return _predictor
    
    if _predictor_loading:
        # منع التحميل المتزامن
        import time
        while _predictor_loading and _predictor is None:
            time.sleep(0.1)
        return _predictor
    
    try:
        _predictor_loading = True
        logger.info("🔄 جاري تحميل نموذج الحروف (Lazy Loading)...")
        _predictor = SignLanguagePredictor()
        logger.info("✅ تم تحميل نموذج الحروف بنجاح")
    except Exception as e:
        logger.error(f"❌ فشل تهيئة المتنبئ: {e}")
        _predictor = None
    finally:
        _predictor_loading = False
    
    return _predictor


# ====== دوال توافق مع الكود القديم ======

def predict(image_path: str) -> Tuple[str, float]:
    predictor = get_predictor()
    if predictor is None:
        raise RuntimeError("المتنبئ غير متاح")
    label, confidence, _ = predictor.predict(image_path)

    # ✅ تحويل اللابل إلى عربي حسب ملف الميتاداتا
    if hasattr(predictor, "mapping") and predictor.mapping:
        label = predictor.mapping.get(label, label)

    return label, confidence


def dummy_extract_text(image_path: str) -> str:
    try:
        label, confidence = predict(image_path)
        return f"{label} (ثقة: {confidence:.2%})"
    except Exception as e:
        return f"فشل استخراج النص: {str(e)}"


def get_top_predictions(image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
    predictor = get_predictor()
    if predictor is None:
        raise RuntimeError("المتنبئ غير متاح")
    _, _, top_k_preds = predictor.predict(image_path, top_k=top_k)
    return top_k_preds

