"""
Phase 1 PDF loader.

Fetches a PDF from a local path or URL, extracts raw text, and produces a
cleaned text bundle. This module isolates PDF I/O and text extraction concerns
from the orchestrator.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from paper_novelty_pipeline.config import PDF_DOWNLOAD_DIR, MAX_RETRIES, RETRY_DELAY
from paper_novelty_pipeline.models import PaperInput
from paper_novelty_pipeline.services import pdf_processor


@dataclass(frozen=True)
class PdfTextBundle:
    pdf_path: Path
    raw_text: str
    cleaned_text: str


class PdfTextLoader:
    """Load PDF and extract raw/cleaned text for Phase1."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def load(self, paper: PaperInput) -> Optional[PdfTextBundle]:
        source = self._resolve_pdf_source(paper)
        if not source:
            self.logger.error("Phase1: no PDF source available for %s", paper.paper_id)
            return None

        result = pdf_processor.fetch_pdf(
            source,
            dest_dir=PDF_DOWNLOAD_DIR,
            retry_total=MAX_RETRIES,
            backoff_factor=RETRY_DELAY,
            verify_pdf=False,
            logger=self.logger,
        )
        if not result.ok or not result.pdf_path:
            return None

        raw_text = pdf_processor.extract_raw_text(result.pdf_path, logger=self.logger)
        if not raw_text:
            return None

        cleaned_text = pdf_processor.process_extracted_text(
            raw_text,
            clean_text=True,
            truncate_refs=False,
            max_chars=None,
            logger=self.logger,
        )
        if cleaned_text is None:
            cleaned_text = ""

        return PdfTextBundle(result.pdf_path, raw_text, cleaned_text)

    def _resolve_pdf_source(self, paper: PaperInput) -> Optional[str]:
        candidates = [
            getattr(paper, "original_pdf_url", None),
            getattr(paper, "openreview_pdf_path", None),
            getattr(paper, "paper_id", None),
        ]
        for source in candidates:
            if not source:
                continue
            source_str = str(source).strip()
            if not source_str:
                continue
            if os.path.exists(source_str):
                return source_str
            if source_str.startswith("/pdf"):
                source_str = "https://openreview.net" + source_str
            try:
                from paper_novelty_pipeline.phases.phase1 import url_parsers

                forum_id = url_parsers.extract_openreview_forum_id(source_str)
                if forum_id:
                    return pdf_processor.build_openreview_pdf_url(forum_id)

                arxiv_id = url_parsers.extract_arxiv_id(source_str)
                if arxiv_id:
                    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            except Exception:
                pass

            return source_str
        return None
