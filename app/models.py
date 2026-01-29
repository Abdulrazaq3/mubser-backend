from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from .database import Base

class ImageItem(Base):
    __tablename__ = "image_items"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    saved_path = Column(String(500), nullable=False)  # مسار الملف على القرص
    notes = Column(Text, nullable=True)               # نص يجي من الواجهة (اختياري)
    extracted_text = Column(Text, nullable=True)      # لاحقاً: نص ناتج نموذج/تعرف OCR
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)