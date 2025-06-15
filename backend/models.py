from pydantic import BaseModel
from enum import Enum
from typing import Optional


class ProcessingStatus(str, Enum):
    """Status of a processing step."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SummaryResult(BaseModel):
    """Result of a summarization operation."""

    original_text: str
    summary: str
    method: str
    language: str
    original_length: int
    summary_length: int
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error: Optional[str] = None
