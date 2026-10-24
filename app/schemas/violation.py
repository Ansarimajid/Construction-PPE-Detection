from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ViolationResponse(BaseModel):
    id: int
    camera_id: int
    violation_type: str
    confidence: float
    timestamp: datetime
    frame_path: Optional[str]
    track_id: Optional[int] = None
    frame_url: Optional[str] = None

    model_config = {"from_attributes": True}


class ViolationListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[ViolationResponse]
