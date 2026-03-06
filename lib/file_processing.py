from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path

import mammoth
import pandas as pd
from pypdf import PdfReader


def _read_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def _read_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts).strip()


def _read_docx(data: bytes) -> str:
    result = mammoth.extract_raw_text(BytesIO(data))
    return result.value.strip()


def _read_csv(data: bytes) -> str:
    df = pd.read_csv(BytesIO(data))
    return _summarize_dataframe(df)


def _read_xlsx(data: bytes) -> str:
    xls = pd.ExcelFile(BytesIO(data))
    summaries = []
    for sheet_name in xls.sheet_names[:3]:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        summaries.append(f"Sheet: {sheet_name}\n{_summarize_dataframe(df)}")
    return "\n\n".join(summaries)


def _summarize_dataframe(df: pd.DataFrame) -> str:
    head = df.head(10).fillna("")
    return (
        f"Rows: {len(df)}\n"
        f"Columns: {', '.join(map(str, df.columns.tolist()))}\n"
        f"Sample rows:\n{head.to_csv(index=False)}"
    )


def extract_text_from_upload(file_name: str, data: bytes) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".txt":
        return _read_txt(data)
    if suffix == ".pdf":
        return _read_pdf(data)
    if suffix == ".docx":
        return _read_docx(data)
    if suffix == ".csv":
        return _read_csv(data)
    if suffix in {".xlsx", ".xls"}:
        return _read_xlsx(data)
    raise ValueError(f"Unsupported file type: {suffix}")
