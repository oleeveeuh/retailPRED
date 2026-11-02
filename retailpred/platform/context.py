"""Shared provenance context for immutable source extracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ExtractContext:
    extract_id: str
    source_url: str
    vintage_time: datetime
    ingested_at: datetime
    release_time: datetime | None = None

    def __post_init__(self) -> None:
        for field in ("vintage_time", "ingested_at", "release_time"):
            value = getattr(self, field)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{field} must be timezone-aware")
        if self.ingested_at < self.vintage_time:
            raise ValueError("ingested_at cannot precede vintage_time")
        if self.release_time is not None and self.ingested_at < self.release_time:
            raise ValueError("ingested_at cannot precede release_time")
