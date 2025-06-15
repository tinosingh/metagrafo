"""Text summarization functionality."""

import logging
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum

from backend.models import SummaryResult, ProcessingStatus
from backend.config import SummarizationMode

logger = logging.getLogger(__name__)


class SummaryLength(str, Enum):
    """Available summary length presets."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class SummarizerConfig(BaseModel):
    """Configuration for the summarizer."""

    mode: SummarizationMode = Field(
        default=SummarizationMode.ABSTRACTIVE, description="Summarization strategy"
    )
    length: SummaryLength = Field(
        default=SummaryLength.MEDIUM, description="Summary length preset"
    )
    language: Optional[str] = Field(
        default=None, description="Language code for summarization"
    )
    abstractive_model: str = Field(
        default="facebook/bart-large-cnn",
        description="Hugging Face model for abstractive summarization",
    )
    min_length: int = Field(default=30)
    max_length: int = Field(default=150)

    class Config:
        use_enum_values = True


class Summarizer:
    """
    Text summarizer supporting abstractive summarization.
    """

    def __init__(self, **kwargs):
        self.config = SummarizerConfig(**kwargs)
        self._pipeline = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of models."""
        if self._initialized:
            return
        try:
            from transformers import pipeline

            logger.info(f"Loading summarization model: {self.config.abstractive_model}")
            self._pipeline = pipeline(
                "summarization",
                model=self.config.abstractive_model,
            )
            logger.info("Summarization model loaded successfully")
            self._initialized = True
        except ImportError:
            logger.warning(
                "Transformers library not available. Summarization will be skipped."
            )
            self._pipeline = None
            self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing summarizer: {e}")
            self._pipeline = None
            self._initialized = True

    def summarize(self, text: str, **kwargs) -> SummaryResult:
        if not text or not text.strip():
            return SummaryResult(
                original_text=text,
                summary="",
                method="none",
                language=self.config.language or "",
                original_length=0,
                summary_length=0,
                status=ProcessingStatus.SKIPPED,
                error="Empty input text",
            )

        config = self.config.model_copy(update=kwargs)

        # Skip summarization if transformers is not available
        try:
            self._initialize()
            if self._pipeline is None:
                return SummaryResult(
                    original_text=text,
                    summary=text,  # Return original text as summary
                    method=config.mode.value,
                    language=config.language or "unknown",
                    original_length=len(text),
                    summary_length=len(text),
                    status=ProcessingStatus.SKIPPED,
                    error="Summarization requires the transformers library which is not available.",
                )

            summary_list = self._pipeline(
                text,
                min_length=config.min_length,
                max_length=config.max_length,
                do_sample=False,
            )
            summary_text = summary_list[0]["summary_text"]

            return SummaryResult(
                original_text=text,
                summary=summary_text,
                method=config.mode.value,
                language=config.language or "unknown",
                original_length=len(text),
                summary_length=len(summary_text),
                status=ProcessingStatus.COMPLETED,
            )
        except Exception as e:
            logger.exception("Error during summarization")
            return SummaryResult(
                original_text=text,
                summary=text,  # Return original text as fallback
                method=config.mode.value,
                language=config.language or "unknown",
                original_length=len(text),
                summary_length=len(text),
                status=ProcessingStatus.FAILED,
                error=str(e),
            )
