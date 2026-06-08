"""PDF loader for the RAG agent.

Public API:
    load_pages(path, range_=[(s, e), ...], fmt="text"|"Document") -> list
    load_pdf_file_tool(...)                                        # LangChain @tool

Page numbers in the public API are 1-indexed.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

import pypdf
from langchain_core.documents import Document
from langchain_core.tools import tool


class PageFormat(str, Enum):
    """Output shape for loaded pages."""

    TEXT = "text"
    DOCUMENT = "Document"


def _resolve_targets(
    total: int,
    range_: Optional[Sequence[Tuple[int, int]]],
) -> List[int]:
    """Convert inclusive (start, end) tuples into a sorted, de-duplicated
    list of valid 1-indexed page numbers."""
    if not range_:
        return list(range(1, total + 1))

    seen: set[int] = set()
    out: List[int] = []
    for start, end in range_:
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError(f"range bounds must be int, got ({start!r}, {end!r})")
        s = max(1, min(start, total))
        e = max(1, min(end, total))
        for p in range(s, e + 1):
            if p not in seen:
                seen.add(p)
                out.append(p)
    out.sort()
    return out


def load_pages(
    pdf_path: str,
    range_: Optional[Sequence[Tuple[int, int]]] = None,
    fmt: Union[PageFormat, str] = PageFormat.TEXT,
) -> List[Union[str, Document]]:
    """Load pages from a PDF.

    Args:
        pdf_path: Path to the PDF file.
        range_: Optional list of inclusive (start, end) page-number tuples,
            1-indexed. Examples:
                [(1, 5)]         -> pages 1..5
                [(1, 3), (7, 9)] -> pages 1,2,3,7,8,9
            If `None`, all pages are returned.
        fmt: `PageFormat.TEXT` returns a list of `str` (one per page).
             `PageFormat.DOCUMENT` returns a list of `langchain_core.documents.Document`.

    Returns:
        List of pages in the requested format, in ascending page order.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    if isinstance(fmt, str):
        try:
            fmt = PageFormat(fmt)
        except ValueError as e:
            raise ValueError(
                f"Invalid fmt={fmt!r}. Use 'text' or 'Document'."
            ) from e

    reader = pypdf.PdfReader(pdf_path)
    total = len(reader.pages)
    targets = _resolve_targets(total, range_)

    filename = os.path.basename(pdf_path)
    if fmt is PageFormat.DOCUMENT:
        return [
            Document(
                page_content=(reader.pages[p - 1].extract_text() or "").strip(),
                metadata={"source": filename, "page": p, "total_pages": total},
            )
            for p in targets
        ]
    return [(reader.pages[p - 1].extract_text() or "").strip() for p in targets]


@tool
def load_pdf_file_tool(
    pdf_path: str,
    range_: Optional[List[Tuple[int, int]]] = None,
    fmt: str = "text",
) -> List[Union[str, Document]]:
    """Load specific page ranges from a PDF.

    Args:
        pdf_path: Local path to the PDF file.
        range_: List of inclusive (start_page, end_page) tuples, 1-indexed.
            Example: `[[1, 5]]` for pages 1-5, `[[1, 3], [7, 9]]` for
            pages 1-3 and 7-9. If `None`, all pages.
        fmt: `"text"` (default) returns a list of strings.
             `"Document"` returns a list of LangChain Document objects.

    Returns:
        List of pages in the requested format, sorted by page number.
    """
    try:
        return load_pages(pdf_path, range_=range_, fmt=fmt)
    except Exception as e:
        return [f"Error loading PDF file: {e}"]
