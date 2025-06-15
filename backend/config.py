from enum import Enum


class SummarizationMode(str, Enum):
    """Summarization strategies."""

    NONE = "none"
    EXTRACTIVE = "extractive"  # Extractive summarization
    ABSTRACTIVE = "abstractive"  # Abstractive summarization (using models)
