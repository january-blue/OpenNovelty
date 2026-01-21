"""PDF download and text extraction helpers used across phases."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import certifi
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib.parse import urlparse
from urllib3.util.retry import Retry

from paper_novelty_pipeline.config import (
    PDF_DOWNLOAD_DIR as DEFAULT_PDF_DOWNLOAD_DIR,
    PDF_DOWNLOAD_MAX_RETRIES,
    PDF_DOWNLOAD_BACKOFF,
    OPENREVIEW_PDF_URL,
)
from paper_novelty_pipeline.utils.text_cleaning import (
    clean_extracted_text,
    sanitize_unicode,
    truncate_at_references,
)


PDF_DOWNLOAD_DIR = DEFAULT_PDF_DOWNLOAD_DIR

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; PaperNoveltyBot/1.0)"


@dataclass(frozen=True)
class PdfDownloadResult:
    ok: bool
    pdf_path: Optional[Path]
    pdf_url: Optional[str]
    is_local: bool
    reason: Optional[str] = None


def build_openreview_pdf_url(paper_id: str, base_url: str = OPENREVIEW_PDF_URL) -> str:
    return f"{base_url}?id={paper_id}"


def infer_filename_from_source(source: str, default_stem: str = "paper") -> str:
    if not source:
        return default_stem
    if source.startswith("http"):
        if "id=" in source:
            return source.split("id=")[-1].split("&")[0] or default_stem
        return default_stem
    return source


def create_session(
    *,
    retry_total: int = PDF_DOWNLOAD_MAX_RETRIES,
    backoff_factor: float = PDF_DOWNLOAD_BACKOFF,
    user_agent: str = DEFAULT_USER_AGENT,
) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=retry_total,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504, 520, 524),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


def _classify_pdf_url(url: str) -> str:
    if not url:
        return "unknown"
    url_lower = url.lower()
    if url_lower.endswith(".pdf") or "/pdf/" in url_lower:
        return "direct_pdf"
    if "/download/" in url_lower:
        return "download_endpoint"
    for pattern in ("/article/download/", "/file/download/", "/getpdf/", "/downloadfile/"):
        if pattern in url_lower:
            return "download_endpoint"
    return "unknown"


def _verify_pdf_content(content: bytes) -> bool:
    if not content or len(content) < 4:
        return False
    return content[:4] == b"%PDF"


def _download_pdf_direct(
    url: str,
    dest: Path,
    session: requests.Session,
    *,
    logger: logging.Logger,
    verify_pdf: bool,
) -> bool:
    try:
        resp = session.get(url, timeout=30, allow_redirects=True, verify=certifi.where())
        resp.raise_for_status()
        if verify_pdf and not _verify_pdf_content(resp.content):
            logger.warning("Direct PDF URL returned non-PDF content: %s", url)
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        logger.info("Direct PDF downloaded: %s", dest)
        return True
    except RequestException as e:
        logger.error("Direct PDF download failed: %s", e)
        return False


def _download_pdf_from_endpoint(
    url: str,
    dest: Path,
    session: requests.Session,
    *,
    logger: logging.Logger,
    verify_pdf: bool,
) -> bool:
    try:
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        enhanced_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,application/octet-stream,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": base_url,
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        original_headers = session.headers.copy()
        session.headers.update(enhanced_headers)

        try:
            resp = session.get(url, timeout=30, allow_redirects=True, verify=certifi.where())
            resp.raise_for_status()
            if verify_pdf and not _verify_pdf_content(resp.content):
                try:
                    content_preview = resp.content[:500].decode("utf-8", errors="ignore")
                    if "<html" in content_preview.lower() or "<!doctype" in content_preview.lower():
                        logger.error("Download endpoint returned HTML instead of PDF: %s", url)
                        logger.debug("HTML preview: %s", content_preview[:200])
                    else:
                        logger.warning("Download endpoint returned non-PDF content: %s", url)
                except Exception:
                    logger.warning("Download endpoint returned non-PDF content (unable to preview): %s", url)
                return False

            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            logger.info("PDF downloaded from endpoint: %s", dest)
            return True
        finally:
            session.headers.clear()
            session.headers.update(original_headers)
    except RequestException as e:
        logger.error("Endpoint PDF download failed: %s", e)
        return False


def download_pdf(
    url: str,
    dest_path: Path,
    *,
    logger: Optional[logging.Logger] = None,
    session: Optional[requests.Session] = None,
    strategy: str = "auto",
    verify_pdf: bool = True,
    retry_total: int = PDF_DOWNLOAD_MAX_RETRIES,
    backoff_factor: float = PDF_DOWNLOAD_BACKOFF,
) -> bool:
    log = logger or logging.getLogger(__name__)
    sess = session or create_session(retry_total=retry_total, backoff_factor=backoff_factor)

    if strategy == "direct":
        return _download_pdf_direct(url, dest_path, sess, logger=log, verify_pdf=verify_pdf)
    if strategy == "endpoint":
        return _download_pdf_from_endpoint(url, dest_path, sess, logger=log, verify_pdf=verify_pdf)

    url_type = _classify_pdf_url(url)
    if url_type == "direct_pdf":
        return _download_pdf_direct(url, dest_path, sess, logger=log, verify_pdf=verify_pdf)
    if url_type == "download_endpoint":
        return _download_pdf_from_endpoint(url, dest_path, sess, logger=log, verify_pdf=verify_pdf)

    log.warning("Unknown URL type, trying direct download: %s", url)
    if _download_pdf_direct(url, dest_path, sess, logger=log, verify_pdf=verify_pdf):
        return True
    log.info("Direct download failed, trying endpoint strategy: %s", url)
    return _download_pdf_from_endpoint(url, dest_path, sess, logger=log, verify_pdf=verify_pdf)


def fetch_pdf(
    source: str,
    *,
    dest_dir: Optional[Path] = None,
    filename_hint: Optional[str] = None,
    openreview_base_url: str = OPENREVIEW_PDF_URL,
    retry_total: int = PDF_DOWNLOAD_MAX_RETRIES,
    backoff_factor: float = PDF_DOWNLOAD_BACKOFF,
    strategy: str = "auto",
    verify_pdf: bool = True,
    logger: Optional[logging.Logger] = None,
) -> PdfDownloadResult:
    log = logger or logging.getLogger(__name__)
    if not source:
        return PdfDownloadResult(False, None, None, False, "empty_source")

    if os.path.exists(source):
        return PdfDownloadResult(True, Path(source).resolve(), None, True, None)

    pdf_url = source if source.startswith("http") else build_openreview_pdf_url(source, openreview_base_url)
    target_dir = Path(dest_dir or PDF_DOWNLOAD_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = filename_hint or infer_filename_from_source(source)
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    dest_path = target_dir / filename

    ok = download_pdf(
        pdf_url,
        dest_path,
        logger=log,
        strategy=strategy,
        verify_pdf=verify_pdf,
        retry_total=retry_total,
        backoff_factor=backoff_factor,
    )
    if not ok:
        return PdfDownloadResult(False, dest_path, pdf_url, False, "download_failed")
    return PdfDownloadResult(True, dest_path, pdf_url, False, None)


def extract_raw_text(pdf_path: Path, *, logger: Optional[logging.Logger] = None) -> Optional[str]:
    log = logger or logging.getLogger(__name__)
    path = Path(pdf_path)
    if not path.exists():
        log.error("PDF not found: %s", path)
        return None
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        log.error("pypdf import failed: %s", e)
        return None

    try:
        reader = PdfReader(str(path))
        chunks = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception as pe:
                log.warning("PyPDF failed on page %s: %s", i, pe)
                txt = ""
            chunks.append(txt)
        full_text = "\n\n".join(chunks).strip()
        full_text = sanitize_unicode(full_text)
        log.info("PyPDF extracted %s characters", len(full_text))
        return full_text if full_text else None
    except Exception as e:
        log.error("PyPDF extraction error: %s", e)
        return None


def process_extracted_text(
    raw_text: str,
    *,
    clean_text: bool = False,
    truncate_refs: bool = False,
    max_chars: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    log = logger or logging.getLogger(__name__)
    if not raw_text:
        return None

    text = raw_text
    if truncate_refs:
        trimmed = truncate_at_references(text)
        if trimmed is not None and len(trimmed) < len(text):
            log.info(
                "Truncated full text at references: reduced from %s to %s chars",
                len(text),
                len(trimmed),
            )
            text = trimmed
    if max_chars is not None and len(text) > max_chars:
        log.info("Truncating full text from %s to %s chars", len(text), max_chars)
        text = text[:max_chars]
    if clean_text:
        text = clean_extracted_text(text)
    return text if text else None


def extract_text(
    pdf_path: Path,
    *,
    clean_text: bool = False,
    truncate_refs: bool = False,
    max_chars: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    raw = extract_raw_text(pdf_path, logger=logger)
    if raw is None:
        return None
    return process_extracted_text(
        raw,
        clean_text=clean_text,
        truncate_refs=truncate_refs,
        max_chars=max_chars,
        logger=logger,
    )
