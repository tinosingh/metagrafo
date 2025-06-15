"""Spell checking functionality."""

import re


def clean_text(text: str) -> str:
    """Removes extra whitespace and ensures consistent punctuation spacing."""
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space.
    text = re.sub(
        r"([.,!?])(?=\S)", r"\1 ", text
    )  # Add space after punctuation if followed by non-space.
    return text.strip()  # Remove leading/trailing whitespace.
