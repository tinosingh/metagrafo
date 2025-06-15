"""Export functionality for transcription results."""

from docx import Document
import io


def export_to_docx(text: str) -> bytes:
    """Export text to DOCX format."""
    doc = Document()
    doc.add_paragraph(text)

    stream = io.BytesIO()
    doc.save(stream)
    stream.seek(0)
    return stream.getvalue()


def export_to_txt(text: str) -> bytes:
    """Export text to plain text format."""
    return text.encode("utf-8")
