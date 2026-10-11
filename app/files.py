"""Turn an uploaded .txt or .pdf into plain text."""
from __future__ import annotations

import io
from pathlib import PurePath


class UnsupportedFile(ValueError):
    pass


def extract_text(filename: str, data: bytes, max_pdf_pages: int) -> str:
    suffix = PurePath(filename).suffix.lower()
    if suffix == ".txt":
        return data.decode("utf-8", errors="replace").strip()
    if suffix == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(io.BytesIO(data))
            pages = reader.pages[:max_pdf_pages]
            return "\n".join((p.extract_text() or "") for p in pages).strip()
        except (PdfReadError, ValueError, KeyError) as exc:
            raise UnsupportedFile(f"{filename}: could not read PDF ({exc})") from exc
    raise UnsupportedFile(f"{filename}: only .txt and .pdf files are supported")
