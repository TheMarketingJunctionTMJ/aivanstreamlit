from __future__ import annotations

from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def export_blog_pdf(title: str, sections: list[dict]) -> bytes:
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=11,
        leading=16,
        spaceAfter=10,
    )

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.25 * inch))

    for section in sections:
        heading = section.get("heading", "").strip()
        content = section.get("content", "").strip()

        if not heading and not content:
            continue

        if heading:
            story.append(Paragraph(heading, heading_style))
            story.append(Spacer(1, 0.08 * inch))

        if content:
            paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
            for para in paragraphs:
                safe_para = (
                    para.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                story.append(Paragraph(safe_para, body_style))
                story.append(Spacer(1, 0.06 * inch))

        story.append(Spacer(1, 0.18 * inch))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
