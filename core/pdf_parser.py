"""
PDF parsing for the Fact-Check Agent.

Extracts page-wise text (and flattened table content) from an uploaded PDF
using pdfplumber. Enforces a page cap to bound cost/runtime on pathological
uploads, and detects scanned/image-only PDFs (where text extraction yields
almost nothing) so the UI can surface an explicit warning instead of
silently reporting zero claims later in the pipeline.
"""

import logging
from dataclasses import dataclass
from typing import BinaryIO, List

import pdfplumber

logger = logging.getLogger(__name__)

MAX_PAGES = 30
# Below this many extracted characters on average per page, treat the
# document as likely scanned/image-only rather than just claim-free.
MIN_CHARS_PER_PAGE_THRESHOLD = 20


class PdfParsingError(RuntimeError):
    """Raised when a PDF cannot be opened or read at all (corrupt file,
    wrong format, or encrypted without a password)."""


@dataclass(frozen=True)
class ParsedPage:
    page_number: int  # 1-indexed, matches what a human would see in a viewer
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    pages: List[ParsedPage]
    truncated: bool  # True if the source file had more than MAX_PAGES
    total_pages_in_file: int  # actual page count before any truncation
    likely_scanned: bool  # True if text yield is suspiciously low

    @property
    def full_text(self) -> str:
        """Page-tagged full text, ready to hand to the claim extractor
        (Phase 2). Page tags let extracted claims carry a page reference."""
        return "\n\n".join(
            f"[Page {p.page_number}]\n{p.text}" for p in self.pages if p.text.strip()
        )

    @property
    def has_extractable_text(self) -> bool:
        return bool(self.full_text.strip())


def parse_pdf(file: BinaryIO, max_pages: int = MAX_PAGES) -> ParsedDocument:
    """Parse a PDF file-like object into page-wise text.

    Args:
        file: a file-like object opened in binary mode (e.g. Streamlit's
            UploadedFile, or a regular `open(path, "rb")` handle).
        max_pages: hard cap on pages processed. Protects API budget
            downstream against pathological uploads (default 30).

    Raises:
        PdfParsingError: if the file cannot be opened as a PDF at all.
    """
    try:
        with pdfplumber.open(file) as pdf:
            total_pages = len(pdf.pages)
            truncated = total_pages > max_pages
            pages_to_read = pdf.pages[:max_pages]

            parsed_pages: List[ParsedPage] = []
            for i, page in enumerate(pages_to_read, start=1):
                text = (page.extract_text() or "").strip()
                table_text = _extract_tables_as_text(page)
                combined = f"{text}\n{table_text}".strip() if table_text else text
                parsed_pages.append(ParsedPage(page_number=i, text=combined))
    except Exception as exc:
        raise PdfParsingError(f"Could not read PDF: {exc}") from exc

    likely_scanned = _is_likely_scanned(parsed_pages)

    if truncated:
        logger.warning(
            "PDF has %d pages; only processing the first %d (MAX_PAGES cap).",
            total_pages,
            max_pages,
        )
    if likely_scanned:
        logger.warning(
            "PDF appears to be scanned/image-only -- little to no extractable text."
        )

    return ParsedDocument(
        pages=parsed_pages,
        truncated=truncated,
        total_pages_in_file=total_pages,
        likely_scanned=likely_scanned,
    )


def _extract_tables_as_text(page) -> str:
    """Flatten any tables on a page into pipe-separated text rows so table
    data (often where stats live in marketing PDFs) reaches the extractor."""
    try:
        tables = page.extract_tables()
    except Exception:
        return ""

    if not tables:
        return ""

    blocks = []
    for table in tables:
        rows = [" | ".join(cell or "" for cell in row) for row in table if row]
        blocks.append("\n".join(rows))
    return "\n".join(blocks)


def _is_likely_scanned(pages: List[ParsedPage]) -> bool:
    """Heuristic: if average extracted text per page is below threshold,
    the PDF is probably scanned/image-only rather than just empty of
    claims. An empty file (zero pages) is treated as scanned too, since
    downstream code should warn rather than silently proceed either way."""
    if not pages:
        return True
    total_chars = sum(len(p.text) for p in pages)
    avg_chars_per_page = total_chars / len(pages)
    return avg_chars_per_page < MIN_CHARS_PER_PAGE_THRESHOLD
