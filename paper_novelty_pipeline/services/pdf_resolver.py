"""Best-effort PDF URL resolution for candidate papers (arXiv/DOI/S2)."""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Literal

import requests
from requests.exceptions import RequestException

from paper_novelty_pipeline.models import RetrievedPaper


PdfStatus = Literal["ok", "no_pdf", "error"]
PdfSourceType = Literal[
    "explicit_pdf",
    "arxiv",
    "arxiv_doi",
    "doi_pdf",
    "direct_pdf",
    "semanticscholar_oa",
    "semanticscholar",
    "unknown",
]


@dataclass
class PdfResolution:
    """Result of attempting to resolve a candidate's PDF URL."""

    status: PdfStatus
    pdf_url: Optional[str] = None
    source_type: Optional[PdfSourceType] = None
    note: Optional[str] = None


def _normalize_candidate_pdf_url(url: str, rp: RetrievedPaper) -> Optional[str]:
    """Normalize explicit pdf_url values (especially arXiv links missing .pdf).

    This is extracted from Phase3 so Phase2/Phase3 can share the same behavior.
    """
    if not url:
        return None
    trimmed = url.strip()
    if not trimmed:
        return None

    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(trimmed)
    netloc = (parts.netloc or "").lower()
    path = parts.path or ""
    query = parts.query or ""
    fragment = parts.fragment or ""

    def _rebuild(new_path: str) -> str:
        return urlunsplit((parts.scheme or "https", parts.netloc, new_path, query, fragment))

    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        return trimmed

    is_arxiv = "arxiv.org" in netloc
    if is_arxiv:
        if path_lower.startswith("/pdf/"):
            normalized_path = path.rstrip("/") + ".pdf"
            return _rebuild(normalized_path)
        if path_lower.startswith("/abs/"):
            token = path.split("/abs/", 1)[-1].strip("/")
            token = token.split("?")[0]
            if not token and rp.arxiv_id:
                token = rp.arxiv_id
            if token:
                return f"https://arxiv.org/pdf/{token}.pdf"
        if rp.arxiv_id:
            return f"https://arxiv.org/pdf/{rp.arxiv_id}.pdf"

    # For non-arXiv URLs, return as-is (caller may still succeed even without .pdf suffix)
    return trimmed


def resolve_pdf_url_for_candidate(
    rp: RetrievedPaper,
    logger: Optional[logging.Logger] = None,
) -> PdfResolution:
    """Best-effort inference of a candidate's PDF URL (arXiv preferred, then DOI).

    This function is a refactoring of the previous Phase3 `_resolve_candidate_pdf_url`
    logic so that Phase2 and Phase3 can share the same behavior.
    """
    log = logger or logging.getLogger(__name__)

    # Resolution order:
    # 1) explicit pdf_url from backend (normalized for arXiv)
    # 2) source_url if it already points to a PDF
    # 3) Semantic Scholar Graph API openAccessPdf (and collect arXiv/DOI hints)
    # 4) arXiv id (from metadata or heuristics)
    # 5) DOI -> follow redirects and detect PDF content
    # otherwise: no_pdf

    # 0) Prefer an explicit pdf_url provided by backend, but fix common arXiv variants
    explicit_pdf = getattr(rp, "pdf_url", None)
    if explicit_pdf:
        normalized_pdf = _normalize_candidate_pdf_url(explicit_pdf, rp)
        if normalized_pdf:
            return PdfResolution(
                status="ok",
                pdf_url=normalized_pdf,
                source_type="explicit_pdf",
                note="from_backend_or_phase2",
            )

    # 0.5) If the backend provided a source_url that already points to a PDF, prefer that
    src = getattr(rp, "source_url", None)
    if src and (src.lower().endswith(".pdf") or "/pdf/" in src.lower()):
        return PdfResolution(
            status="ok",
            pdf_url=src,
            source_type="direct_pdf",
            note="from_source_url_pdf",
        )

    # 0.6) Semantic Scholar URL: try Graph API for open-access PDF
    if src:
        try:
            from urllib.parse import urlsplit

            parts = urlsplit(src)
            host = (parts.netloc or "").lower()
            if "semanticscholar.org" in host and "/paper/" in parts.path:
                # Extract paperId from /paper/<paperId>(/...)?
                paper_id = parts.path.split("/paper/", 1)[-1].strip("/").split("/")[0]
                if paper_id:
                    api_url = (
                        "https://api.semanticscholar.org/graph/v1/"
                        f"paper/{paper_id}?fields=openAccessPdf,externalIds"
                    )
                    resp = requests.get(api_url, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        oa = data.get("openAccessPdf") or {}
                        oa_url = oa.get("url")
                        if oa_url:
                            return PdfResolution(
                                status="ok",
                                pdf_url=oa_url,
                                source_type="semanticscholar_oa",
                                note="from_semanticscholar_openAccessPdf",
                            )
                        # Fallback: use externalIds to discover arXiv / DOI
                        ext = data.get("externalIds") or {}
                        arxiv_from_s2 = ext.get("ArXiv") or ext.get("arXiv")
                        doi_from_s2 = ext.get("DOI") or ext.get("doi")
                        if arxiv_from_s2 and not rp.arxiv_id:
                            rp.arxiv_id = str(arxiv_from_s2)
                        if doi_from_s2 and not rp.doi:
                            rp.doi = str(doi_from_s2)
                        # Continue to arXiv/DOI resolution below
                    else:
                        log.warning(
                            "[PdfResolver] Semantic Scholar API returned %s for %s",
                            resp.status_code,
                            api_url,
                        )
        except RequestException as e:
            log.warning("[PdfResolver] Semantic Scholar API error for %s: %s", src, e)
        except Exception as e:
            log.debug("[PdfResolver] Semantic Scholar handling failed: %s", e)

    # 1) arXiv id - try multiple fallbacks and normalize to canonical numeric id
    arxiv_id = (rp.arxiv_id or "").strip()
    if not arxiv_id:
        pid = (rp.paper_id or "").strip()
        # pid may contain forms like 'arxiv:arXiv:2503.01328' or an arxiv.org url
        m = re.search(r"arxiv\.org/(abs|pdf)/([\w\.-:]+)", pid, re.IGNORECASE)
        if m:
            arxiv_id = m.group(2)
        else:
            # also check for bare numeric id in paper_id
            m2 = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", pid)
            if m2:
                arxiv_id = m2.group(1)

    # If we have an arXiv-like token, prefer direct PDF using the provided token (keep version if present)
    if arxiv_id:
        mnum = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", arxiv_id)
        if mnum:
            pdf = f"https://arxiv.org/pdf/{mnum.group(1)}.pdf"
        else:
            pdf = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return PdfResolution(
            status="ok",
            pdf_url=pdf,
            source_type="arxiv",
            note="constructed_pdf_from_arxiv",
        )

    # 2) DOI: best-effort via doi.org (current behavior; can later be replaced by OA-only logic)
    doi = (rp.doi or "").strip()
    # Normalize DOI: strip leading https?://doi.org/
    if doi.lower().startswith("http://doi.org/") or doi.lower().startswith("https://doi.org/"):
        doi = re.sub(r"^https?://doi\.org/", "", doi, flags=re.IGNORECASE)

    # Special case: arXiv DOI via 10.48550/arxiv.*
    if doi.lower().startswith("10.48550/arxiv."):
        arxiv_from_doi = doi.split("arxiv.", 1)[-1]
        if arxiv_from_doi:
            mnum = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", arxiv_from_doi)
            token = mnum.group(1) if mnum else arxiv_from_doi
            pdf = f"https://arxiv.org/pdf/{token}.pdf"
            return PdfResolution(
                status="ok",
                pdf_url=pdf,
                source_type="arxiv_doi",
                note="constructed_pdf_from_arxiv_doi",
            )

    if doi:
        # Guard against clearly invalid DOIs (e.g., just "10.1002" with no suffix)
        if "/" not in doi:
            log.debug("[PdfResolver] Skipping malformed DOI without slash: %s", doi)
        else:
            test_url = f"https://doi.org/{doi}"
            try:
                head = requests.head(test_url, timeout=15, allow_redirects=True)
                final_url = head.url
                ctype = head.headers.get("Content-Type", "")
                if "pdf" in ctype.lower() or (final_url and final_url.lower().endswith(".pdf")):
                    return PdfResolution(
                        status="ok",
                        pdf_url=final_url,
                        source_type="doi_pdf",
                        note="resolved_via_doi_head",
                    )
                get = requests.get(test_url, timeout=20, allow_redirects=True)
                final_url = get.url
                ctype = get.headers.get("Content-Type", "")
                if "pdf" in ctype.lower() or (final_url and final_url.lower().endswith(".pdf")):
                    return PdfResolution(
                        status="ok",
                        pdf_url=final_url,
                        source_type="doi_pdf",
                        note="resolved_via_doi_get",
                    )
            except Exception as e:
                log.warning("[PdfResolver] DOI→PDF resolution failed for %s: %s", doi, e)
                return PdfResolution(
                    status="error",
                    pdf_url=None,
                    source_type="doi_pdf",
                    note=f"doi_resolution_error: {e}",
                )

    # 3) No generic resolver for other sources at the moment
    return PdfResolution(
        status="no_pdf",
        pdf_url=None,
        source_type="unknown",
        note=f"no_pdf_url; source_url={src}",
    )


