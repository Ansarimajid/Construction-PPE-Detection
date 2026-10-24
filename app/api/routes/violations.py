from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Violation
from app.db.session import get_db
from app.schemas.violation import ViolationListResponse, ViolationResponse

router = APIRouter(prefix="/violations", tags=["violations"])


def _to_response(v: Violation) -> ViolationResponse:
    frame_url = f"/frames/{v.frame_path}" if v.frame_path else None
    return ViolationResponse(
        id=v.id,
        camera_id=v.camera_id,
        violation_type=v.violation_type,
        confidence=v.confidence,
        timestamp=v.timestamp,
        frame_path=v.frame_path,
        track_id=v.track_id,
        frame_url=frame_url,
    )


@router.get("", response_model=ViolationListResponse)
async def list_violations(
    camera_id: Optional[int] = Query(None),
    violation_type: Optional[str] = Query(None),
    from_dt: Optional[datetime] = Query(None, alias="from"),
    to_dt: Optional[datetime] = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    q = select(Violation).order_by(Violation.timestamp.desc())
    if camera_id is not None:
        q = q.where(Violation.camera_id == camera_id)
    if violation_type is not None:
        q = q.where(Violation.violation_type == violation_type)
    if from_dt is not None:
        q = q.where(Violation.timestamp >= from_dt)
    if to_dt is not None:
        q = q.where(Violation.timestamp <= to_dt)

    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar_one()

    q = q.offset((page - 1) * page_size).limit(page_size)
    items = (await db.execute(q)).scalars().all()

    return ViolationListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[_to_response(v) for v in items],
    )


@router.get("/export")
async def export_violations(
    camera_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    q = select(Violation).order_by(Violation.timestamp.desc())
    if camera_id is not None:
        q = q.where(Violation.camera_id == camera_id)
    violations = (await db.execute(q)).scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        ["id", "camera_id", "track_id", "violation_type", "confidence", "timestamp", "frame_path"]
    )
    for v in violations:
        writer.writerow(
            [v.id, v.camera_id, v.track_id, v.violation_type, v.confidence, v.timestamp, v.frame_path]
        )

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=violations.csv"},
    )


@router.get("/{violation_id}", response_model=ViolationResponse)
async def get_violation(
    violation_id: int,
    db: AsyncSession = Depends(get_db),
):
    v = await db.get(Violation, violation_id)
    if v is None:
        raise HTTPException(status_code=404, detail="Violation not found")
    return _to_response(v)
