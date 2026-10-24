from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_ENV == "dev",
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def _add_missing_columns(sync_conn) -> None:
    # create_all doesn't alter existing tables; add columns introduced after the first release
    columns = {c["name"] for c in inspect(sync_conn).get_columns("violations")}
    if "track_id" not in columns:
        sync_conn.execute(text("ALTER TABLE violations ADD COLUMN track_id INTEGER"))


async def init_db() -> None:
    from app.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)
