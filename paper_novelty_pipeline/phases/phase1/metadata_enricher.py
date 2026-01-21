"""
Phase 1 metadata enrichment.

Enriches a PaperInput with metadata from APIs, URL hints, and PDF text
heuristics. This module centralizes enrichment logic so the orchestrator only
coordinates the flow.

TODO: Extend with arXiv/ACL/etc. Scholar API enrichers.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from paper_novelty_pipeline.models import PaperInput


class MetadataEnricher:
    """Enrich paper metadata from multiple sources."""

    def __init__(self, logger: Optional[logging.Logger] = None, llm_client=None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.llm_client = llm_client

    def enrich(
        self,
        paper: PaperInput,
        *,
        full_text: str,
        source_url: Optional[str],
        original_pdf_url: Optional[str] = None,
        call_llm_json_fn=None,
        call_llm_text_fn=None,
        extract_title: bool = True,
    ) -> Dict[str, Any]:
        """
        Enrich metadata from APIs, URLs, and full text.

        Returns publication info for persistence.
        """
        # API/URL-based metadata (OpenReview first)
        self.enrich_openreview(paper, source_url=source_url, original_pdf_url=original_pdf_url)
        self._apply_url_identifiers(paper, source_url or paper.paper_id)

        # Text-based enrichment
        pub_info = self.extract_publication_date(
            full_text=full_text,
            source_url=source_url,
            call_llm_json_fn=call_llm_json_fn,
        )
        self._apply_publication_date(paper, pub_info)

        if extract_title and not self.is_trustworthy_title(paper):
            extracted_title = self.extract_title_from_text(
                full_text, call_llm_fn=call_llm_text_fn
            )
            if extracted_title:
                paper.title = extracted_title
                self.logger.info(
                    "Phase1: filled title from PDF text: %s...", extracted_title[:80]
                )

        return pub_info

    def enrich_openreview(self, paper: PaperInput, source_url: Optional[str], original_pdf_url: Optional[str]) -> None:
        forum_id = self._extract_openreview_forum_id(source_url) or self._extract_openreview_forum_id(
            original_pdf_url
        )
        if not forum_id:
            return

        try:
            inferred_year = self.infer_year(paper, source_url=source_url)
            from paper_novelty_pipeline.services.openreview_client import fetch_meta_for_forum

            meta = fetch_meta_for_forum(forum_id, year=inferred_year)
            self.logger.info(
                "Phase1: fetched OpenReview metadata for forum_id=%s (year=%s, ratings=%s, mean=%s)",
                forum_id,
                inferred_year,
                len(meta.ratings),
                meta.rating_mean,
            )

            paper.openreview_forum_id = meta.forum_id

            if meta.title and not self.is_trustworthy_title(paper):
                paper.title = meta.title
            if meta.abstract and not paper.abstract:
                paper.abstract = meta.abstract
            if meta.authors:
                paper.authors = meta.authors
            if meta.raw_submission_content and "venue" in meta.raw_submission_content:
                venue_val = meta.raw_submission_content["venue"]
                if venue_val and not paper.venue:
                    paper.venue = venue_val

            if meta.keywords:
                paper.keywords = meta.keywords
            if meta.primary_area:
                paper.primary_area = meta.primary_area
            if meta.submission_number:
                paper.submission_number = meta.submission_number
            if meta.raw_submission_content and "venueid" in meta.raw_submission_content:
                venueid_val = meta.raw_submission_content["venueid"]
                if venueid_val:
                    paper.openreview_venueid = venueid_val
            if meta.raw_submission_content and "pdf" in meta.raw_submission_content:
                pdf_val = meta.raw_submission_content["pdf"]
                if pdf_val:
                    paper.openreview_pdf_path = pdf_val
            if meta.raw_submission_content and "_bibtex" in meta.raw_submission_content:
                bibtex_val = meta.raw_submission_content["_bibtex"]
                if bibtex_val:
                    paper.bibtex = bibtex_val

            if meta.rating_mean is not None:
                paper.openreview_rating_mean = meta.rating_mean
            if meta.ratings:
                paper.openreview_ratings = meta.ratings
        except Exception as e:
            self.logger.warning("Phase1: OpenReview enrichment failed for %s: %s", paper.paper_id, e)

    def _extract_openreview_forum_id(self, url: Optional[str]) -> Optional[str]:
        """Delegate to shared URL parser."""
        from paper_novelty_pipeline.phases.phase1.url_parsers import extract_openreview_forum_id
        return extract_openreview_forum_id(url)

    def _apply_url_identifiers(self, paper: PaperInput, source_url: Optional[str]) -> None:
        if not source_url:
            return
        try:
            from paper_novelty_pipeline.phases.phase1.url_parsers import extract_arxiv_id, extract_openreview_forum_id

            if not paper.arxiv_id:
                arxiv_id = extract_arxiv_id(source_url)
                if arxiv_id:
                    paper.arxiv_id = arxiv_id

            if not paper.openreview_forum_id:
                forum_id = extract_openreview_forum_id(source_url)
                if forum_id:
                    paper.openreview_forum_id = forum_id
        except Exception as e:
            self.logger.debug("URL metadata extraction failed: %s", e)

    def infer_year(
        self,
        paper: PaperInput,
        pub_info: Optional[Dict[str, Any]] = None,
        source_url: Optional[str] = None,
    ) -> int:
        if paper.year is not None:
            return paper.year
        if pub_info and pub_info.get("year"):
            return pub_info["year"]
        url_year = self._infer_year_from_url(source_url or paper.paper_id)
        if url_year is not None:
            return url_year
        from datetime import datetime

        return datetime.now().year

    def _infer_year_from_url(self, source_url: Optional[str]) -> Optional[int]:
        if not source_url:
            return None
        year, _ = self._infer_from_url(source_url)
        return year

    def _infer_from_url(self, url: Optional[str]) -> tuple[Optional[int], Optional[int]]:
        """Delegate to shared URL parser."""
        from paper_novelty_pipeline.phases.phase1.url_parsers import infer_date_from_url
        return infer_date_from_url(url)

    def is_trustworthy_title(self, paper: PaperInput) -> bool:
        title = (paper.title or "").strip()
        if not title:
            return False
        paper_id = (paper.paper_id or "").strip()
        if title == paper_id:
            return False
        lower = title.lower()
        if lower.startswith("http://") or lower.startswith("https://"):
            return False
        return True

    def extract_publication_date(self, full_text: str, source_url: Optional[str], call_llm_json_fn=None) -> Dict[str, Any]:
        """
        Best-effort publication date extraction.

        Tries multiple strategies:
        1. URL-based inference (arXiv)
        2. Regex on front matter (4 strategies)
        3. LLM fallback (if llm_client available)

        Args:
            full_text: Full PDF text
            source_url: Paper source URL (for URL-based inference)
            call_llm_json_fn: Optional LLM JSON caller (for LLM fallback)

        Returns:
            Dict with keys: year, month, day, granularity, source, confidence
        """
        # 1) URL-based inference
        from paper_novelty_pipeline.phases.phase1.url_parsers import infer_date_from_url
        year, month = infer_date_from_url(source_url)
        if year:
            return {
                "year": year,
                "month": month,
                "day": None,
                "granularity": "year-month" if month else "year",
                "source": "url",
                "confidence": 0.95,
            }

        # 2) Regex on front matter
        front = (full_text or "").strip()[:4000]
        info = self._regex_pubdate(front)
        if info and info.get("year"):
            return info

        # 3) LLM fallback
        if self.llm_client and call_llm_json_fn:
            try:
                return self._llm_extract_date(front, call_llm_json_fn)
            except Exception as e:
                self.logger.debug(f"LLM date extraction failed: {e}")

        return {
            "year": None,
            "month": None,
            "day": None,
            "granularity": "none",
            "source": "none",
            "confidence": 0.0,
        }

    def _apply_publication_date(self, paper: PaperInput, pub_info: Dict[str, Any]) -> None:
        if pub_info and pub_info.get("year"):
            year = pub_info.get("year")
            paper.year = year
            month = pub_info.get("month")
            day = pub_info.get("day")
            granularity = pub_info.get("granularity") or "none"
            source = pub_info.get("source")
            confidence = pub_info.get("confidence")
            if month and day:
                self.logger.info(
                    "Phase1: publication date inferred: %04d-%02d-%02d (granularity=%s, source=%s, confidence=%s)",
                    year,
                    month,
                    day,
                    granularity,
                    source,
                    confidence,
                )
            elif month:
                self.logger.info(
                    "Phase1: publication date inferred: %04d-%02d (granularity=%s, source=%s, confidence=%s)",
                    year,
                    month,
                    granularity,
                    source,
                    confidence,
                )
            else:
                self.logger.info(
                    "Phase1: publication date inferred: %04d (granularity=%s, source=%s, confidence=%s)",
                    year,
                    granularity,
                    source,
                    confidence,
                )
        else:
            self.logger.warning(
                "Phase1: no publication date inferred (granularity=none); "
                "Phase2 cutoff will be disabled unless provided via --pub-date"
            )

    def _regex_pubdate(self, text: str) -> Optional[Dict[str, Any]]:
        """Regex-based date extraction (4 strategies)."""
        # 1) Full date YYYY-MM-DD or YYYY/MM/DD
        match = re.search(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", text)
        if match:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return {
                    "year": year,
                    "month": month,
                    "day": day,
                    "granularity": "year-month-day",
                    "source": "regex_full",
                    "confidence": 0.85,
                }

        # 2) Month name YYYY (English)
        months = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
        match = re.search(rf"\b({months})\.?\s+(20\d{{2}})\b", text, re.IGNORECASE)
        if match:
            month_map = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
            }
            month_str = match.group(1).lower()
            month_str = month_str[:4] if month_str.startswith("sept") else month_str[:3]
            month = month_map.get(month_str)
            year = int(match.group(2))
            if month:
                return {
                    "year": year,
                    "month": month,
                    "day": None,
                    "granularity": "year-month",
                    "source": "regex_month_year",
                    "confidence": 0.8,
                }

        # 3) Venue + Year
        venue_keywords = (
            r"conference|journal|proceedings|workshop|symposium|transactions|"
            r"会议|期刊|研讨会|大会"
        )
        for line in text.splitlines():
            match = re.search(r"\b(20\d{2})\b", line)
            if match and re.search(venue_keywords, line, re.IGNORECASE):
                year = int(match.group(1))
                return {
                    "year": year,
                    "month": None,
                    "day": None,
                    "granularity": "year",
                    "source": "regex_venueyear",
                    "confidence": 0.7,
                }

        # 4) Plain year
        match = re.search(r"\b(20\d{2})\b", text)
        if match:
            year = int(match.group(1))
            return {
                "year": year,
                "month": None,
                "day": None,
                "granularity": "year",
                "source": "regex_year",
                "confidence": 0.6,
            }

        return None

    def _llm_extract_date(self, front_text: str, call_llm_json_fn) -> Dict[str, Any]:
        """LLM-based date extraction."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract a publication date from the first page of a PDF text. "
                    "Return JSON with fields: year (int or null), month (1-12 or null), day (1-31 or null), "
                    "granularity ('year'|'year-month'|'year-month-day'|'none'), source ('frontmatter_llm'), confidence (0-1). "
                    "If unsure, prefer year; if month exists but no day, set granularity='year-month'. If nothing, set granularity='none'. "
                    "Return raw JSON only (no ```json``` fences)."
                ),
            },
            {
                "role": "user",
                "content": f"First pages text (truncated):\n\n{front_text}\n\nExtract earliest publication date.",
            },
        ]
        data = call_llm_json_fn(messages, max_tokens=600, temperature=0.0)
        year = data.get("year") if isinstance(data, dict) else None
        month = data.get("month") if isinstance(data, dict) else None
        day = data.get("day") if isinstance(data, dict) else None
        granularity = data.get("granularity") if isinstance(data, dict) else None

        if year:
            # Clamp month/day
            try:
                month = int(month) if month else None
                if month and not (1 <= month <= 12):
                    month = None
            except Exception:
                month = None
            try:
                day = int(day) if day else None
                if day and not (1 <= day <= 31):
                    day = None
            except Exception:
                day = None

            if month and day:
                granularity = "year-month-day"
            elif month:
                granularity = "year-month"
            else:
                granularity = "year"

            confidence = data.get("confidence", 0.6) if isinstance(data, dict) else 0.6
            return {
                "year": int(year),
                "month": month,
                "day": day,
                "granularity": granularity,
                "source": "frontmatter_llm",
                "confidence": confidence,
            }

        return {
            "year": None,
            "month": None,
            "day": None,
            "granularity": "none",
            "source": "none",
            "confidence": 0.0,
        }

    def extract_title_from_text(self, full_text: str, call_llm_fn=None) -> Optional[str]:
        """
        Extract paper title from PDF text.

        Tries multiple strategies:
        1. LLM extraction (if llm_client available)
        2. Heuristic - longest line in first few lines

        Args:
            full_text: Full PDF text

        Returns:
            Extracted title, or None if failed
        """
        if not full_text:
            return None

        # Method 1: LLM extraction
        if self.llm_client or call_llm_fn:
            try:
                title = self._llm_extract_title(full_text[:3000], call_llm_fn=call_llm_fn)
                if title and 10 < len(title) < 300:
                    return title
            except Exception as e:
                self.logger.debug(f"LLM title extraction failed: {e}")

        # Method 2: Heuristic
        return self._heuristic_extract_title(full_text)

    def _llm_extract_title(self, first_page: str, call_llm_fn=None) -> Optional[str]:
        """LLM-based title extraction."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract the paper title from the first page text. "
                    "Return only the title as a plain string, no quotes, no JSON. "
                    "If you cannot find a clear title, return an empty string."
                ),
            },
            {"role": "user", "content": f"First page text:\n\n{first_page}\n\nExtract the paper title:"},
        ]
        if call_llm_fn:
            response = call_llm_fn(messages, max_tokens=200, temperature=0.0)
        else:
            response = self.llm_client.generate(messages, max_tokens=200, temperature=0.0)
        title = response.strip().strip('"').strip("'")
        return title if title else None

    def _heuristic_extract_title(self, full_text: str) -> Optional[str]:
        """Heuristic title extraction (longest line in first few lines)."""
        lines = full_text.split("\n")[:20]
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        if not non_empty_lines:
            return None

        # Filter out common non-title patterns
        title_candidates = [
            line
            for line in non_empty_lines[:10]
            if 15 < len(line) < 250
            and not line.lower().startswith(("abstract", "introduction", "keywords", "author", "affiliation"))
            and not any(word in line.lower() for word in ["@", "http", "doi:", "arxiv:", "email"])
        ]

        if title_candidates:
            # Return the longest candidate (usually the title)
            title = max(title_candidates, key=len)
            return title if len(title) > 20 else None

        return None
