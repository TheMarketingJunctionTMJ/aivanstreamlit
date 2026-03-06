from __future__ import annotations

from io import BytesIO
from docx import Document


def export_blog_docx(title: str, sections: list[dict[str, str]]) -> bytes:
    doc = Document()
    doc.add_heading(title, level=0)

    for section in sections:
        heading = section.get("heading", "")
        content = section.get("content", "")
        if heading:
            doc.add_heading(heading, level=1)
        for paragraph in content.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                doc.add_paragraph(paragraph)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
