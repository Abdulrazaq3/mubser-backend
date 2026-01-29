from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ImageBase(BaseModel):
    notes: Optional[str] = Field(None, description="نص مرتبط بالصورة (اختياري)")

class ImageOut(BaseModel):
    id: int
    filename: str
    content_type: str
    size_bytes: int
    saved_path: str
    notes: Optional[str] = None
    extracted_text: Optional[str] = None
    created_at: datetime
    updated_at: datetime    

    class Config:
        from_attributes = True

class AnalyzeResponse(BaseModel):
    label: str
    confidence: float
    text: str



class AnalyzeBase64Request(BaseModel):
    image_b64: str  # data URL أو Base64 خام
    filename: str | None = None