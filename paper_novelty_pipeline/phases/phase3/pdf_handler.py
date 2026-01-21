"""
PDF Handler Module for Phase3

Handles all PDF download, verification, and text extraction operations.
Separated from phase3_comparison.py to improve modularity and maintainability.
"""

import os
import logging
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from paper_novelty_pipeline.models import PaperInput, RetrievedPaper
from paper_novelty_pipeline.services.pdf_resolver import resolve_pdf_url_for_candidate
from paper_novelty_pipeline.services import pdf_processor
from paper_novelty_pipeline.utils.text_cache import TextCache
from paper_novelty_pipeline.config import (
    PDF_DOWNLOAD_DIR,
    OPENREVIEW_PDF_URL,
    PHASE3_KEEP_DOWNLOADED_PDFS,
    MAX_CONTEXT_CHARS,
)


class PDFHandler:
    """
    Handles PDF download and text extraction for Phase3 comparison.
    
    Responsibilities:
    - Download PDFs from various sources (OpenReview, arXiv, direct URLs, endpoints)
    - Verify PDF content validity
    - Extract text using PyPDF
    - Manage PDF and text file storage
    - Integrate with TextCache to avoid redundant downloads/extractions
    """
    
    def __init__(
        self, 
        output_base_dir: Optional[str] = None, 
        logger: Optional[logging.Logger] = None,
        text_cache: Optional[TextCache] = None
    ):
        """
        Initialize PDFHandler.
        
        Args:
            output_base_dir: Base directory for output files
            logger: Logger instance (if None, creates new logger)
            text_cache: TextCache instance for caching extracted texts (optional)
        """
        self.output_base_dir = output_base_dir
        self.logger = logger or logging.getLogger(__name__)
        self.text_cache = text_cache
        
        # Ensure PDF download directory exists
        os.makedirs(PDF_DOWNLOAD_DIR, exist_ok=True)
    
    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================
    
    def get_original_full_text(self, paper: PaperInput, canonical_id: Optional[str] = None) -> Optional[str]:
        """
        Get the original paper's full text from PDF.
        
        Supports:
        - OpenReview ID (constructs URL and downloads)
        - Direct HTTP/HTTPS URL (downloads)
        - Local file path (uses directly)
        - TextCache integration: avoid redundant downloads/extractions
        
        Args:
            paper: PaperInput object with paper metadata
            canonical_id: Optional canonical_id for caching (recommended to pass from Phase2)
        
        Note: Phase 1 now preserves the downloaded PDF, so we first check if it exists
        before re-downloading to avoid duplicate downloads.
        If anything fails, return None and continue.
        """
        # Try TextCache first if available
        if self.text_cache and canonical_id:
            cached_text = self.text_cache.get_cached_text(canonical_id)
            if cached_text:
                self.logger.info(f"[Original] ✓ Cache hit for {canonical_id}")
                return cached_text
            else:
                self.logger.info(f"[Original] ✗ Cache miss for {canonical_id}, extracting...")
        
        pdf_path = None
        is_local_file = False
        is_phase1_pdf = False
        try:
            paper_id = paper.paper_id
            
            # Case 1: Local file path (check if it exists)
            if os.path.exists(paper_id):
                pdf_path = os.path.abspath(paper_id)
                is_local_file = True
                self.logger.info(f"[Original] Using local PDF file: {pdf_path}")
                text = pdf_processor.extract_text(
                    Path(pdf_path),
                    truncate_refs=True,
                    max_chars=MAX_CONTEXT_CHARS,
                    logger=self.logger,
                )
                return text
            
            # Check if Phase1 already downloaded this PDF
            phase1_pdf_path = None
            if isinstance(paper_id, str) and paper_id.startswith("http"):
                safe = paper_id.split("id=")[-1].split("&")[0] if "id=" in paper_id else "paper"
                phase1_pdf_path = os.path.join(PDF_DOWNLOAD_DIR, f"{safe}.pdf")
            else:
                # OpenReview ID case
                phase1_pdf_path = os.path.join(PDF_DOWNLOAD_DIR, f"{paper_id}.pdf")
            
            # If Phase1 PDF exists, use it directly
            if phase1_pdf_path and os.path.exists(phase1_pdf_path):
                pdf_path = phase1_pdf_path
                is_phase1_pdf = True
                self.logger.info(f"[Original] Using Phase1 downloaded PDF: {pdf_path}")
                text = pdf_processor.extract_text(
                    Path(pdf_path),
                    truncate_refs=True,
                    max_chars=MAX_CONTEXT_CHARS,
                    logger=self.logger,
                )
                return text
            
            # Phase1 PDF not found, need to download
            # Case 2: HTTP/HTTPS URL
            if isinstance(paper_id, str) and paper_id.startswith("http"):
                pdf_url = paper_id
                safe = paper_id.split("id=")[-1].split("&")[0] if "id=" in paper_id else "paper"
                pdf_path = os.path.join(PDF_DOWNLOAD_DIR, f"{safe}_orig.pdf")
            else:
                # Case 3: OpenReview ID (construct URL)
                pdf_url = f"{OPENREVIEW_PDF_URL}?id={paper_id}"
                pdf_path = os.path.join(PDF_DOWNLOAD_DIR, f"{paper_id}_orig.pdf")

            self.logger.info(f"[Original] Phase1 PDF not found, downloading PDF: {pdf_url}")
            download_ok = pdf_processor.download_pdf(
                pdf_url,
                Path(pdf_path),
                logger=self.logger,
                verify_pdf=False,
            )
            if not download_ok:
                self.logger.error(f"[Original] giving up fetching PDF: download failed")
                return None

            text = pdf_processor.extract_text(
                Path(pdf_path),
                truncate_refs=True,
                max_chars=MAX_CONTEXT_CHARS,
                logger=self.logger,
            )
            
            # Cache the extracted text if TextCache is available
            if text and self.text_cache and canonical_id:
                self.text_cache.cache_text(
                    canonical_id=canonical_id,
                    full_text=text,
                    metadata={
                        "title": paper.title,
                        "paper_id": paper.paper_id,
                        "source": {
                            "type": "pdf_full",
                            "url": pdf_path if not is_local_file else None,
                            "extraction_method": "pypdf",
                            "fallback_reason": None
                        },
                        "paper": {
                            "title": paper.title,
                            "paper_id": paper.paper_id,
                            "authors": getattr(paper, "authors", []),
                            "year": getattr(paper, "year", None),
                            "venue": getattr(paper, "venue", None)
                        }
                    }
                )
            
            return text
        except Exception as e:
            self.logger.error(f"[Original] fulltext extraction failed: {e}")
            return None
        finally:
            # Clean up the downloaded PDF to avoid piling up temp files
            # But don't delete local files that were passed in, and don't delete Phase1 PDFs
            if (
                pdf_path
                and not is_local_file
                and not is_phase1_pdf
                and pdf_path.startswith(str(PDF_DOWNLOAD_DIR))
            ):
                try:
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                except Exception:
                    pass
    
    def get_candidate_full_text(
        self, rp: RetrievedPaper, safe_name_func=None
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Resolve a candidate PDF URL and extract full text via PyPDF.
        Now supports different download strategies based on URL type.
        Integrated with TextCache to avoid redundant downloads/extractions.

        Args:
            rp: Retrieved paper object
            safe_name_func: Function to generate safe filenames (optional)

        Returns:
            Tuple of (text_or_abstract, source_type, skip_reason)
            - source_type: 'fulltext' if extracted from a downloaded PDF,
                           'constructed_pdf' if we used an arXiv-constructed PDF URL,
                           'abstract' if we fall back to using the provided abstract,
                           'cached' if loaded from TextCache.
            - skip_reason: non-None only when both PDF and abstract are unavailable.
        """
        # Try TextCache first if available
        if self.text_cache and rp.canonical_id:
            cached_text = self.text_cache.get_cached_text(rp.canonical_id)
            if cached_text:
                self.logger.info(f"[Candidate] ✓ Cache hit for {rp.canonical_id}")
                return cached_text, "cached", None
            else:
                self.logger.info(f"[Candidate] ✗ Cache miss for {rp.canonical_id}, extracting...")
        
        url = self.resolve_candidate_pdf_url(rp)
        if not url:
            # Fallback to abstract if no URL
            if getattr(rp, "abstract", None):
                self.logger.info(f"[Candidate] falling back to abstract for paper_id={rp.paper_id}")
                
                # Cache abstract to avoid repeated attempts
                if self.text_cache and rp.canonical_id:
                    self.text_cache.cache_text(
                        canonical_id=rp.canonical_id,
                        full_text=rp.abstract,
                        metadata={
                            "title": rp.title,
                            "paper_id": rp.paper_id,
                            "source": {
                                "type": "abstract_fallback",
                                "url": None,
                                "extraction_method": "abstract",
                                "fallback_reason": "no_pdf_url"
                            },
                            "paper": {
                                "title": rp.title,
                                "paper_id": rp.paper_id,
                                "authors": getattr(rp, "authors", []),
                                "year": getattr(rp, "year", None),
                                "venue": getattr(rp, "venue", None)
                            }
                        }
                    )
                
                return rp.abstract, "abstract", None
            return None, None, "no_pdf_no_abstract"
        
        self.logger.info(f"[Candidate] resolved PDF URL: {url}")
        
        # Generate safe filename
        if safe_name_func:
            safe_id = safe_name_func(rp.paper_id)
        else:
            # Create filesystem-safe name inline
            import re
            safe_id = re.sub(r"[^\w\-.]+", "_", rp.paper_id or "paper")[:80]
        
        dest = os.path.join(PDF_DOWNLOAD_DIR, f"candidate_{safe_id}.pdf")
        
        session = pdf_processor.create_session()
        download_success = pdf_processor.download_pdf(
            url,
            Path(dest),
            session=session,
            logger=self.logger,
            strategy="auto",
        )
        
        if not download_success:
            # Fallback to abstract
            if getattr(rp, "abstract", None):
                self.logger.info(f"[Candidate] PDF download failed, falling back to abstract for paper_id={rp.paper_id}")
                
                # Cache abstract to avoid repeated download attempts
                if self.text_cache and rp.canonical_id:
                    self.text_cache.cache_text(
                        canonical_id=rp.canonical_id,
                        full_text=rp.abstract,
                        metadata={
                            "title": rp.title,
                            "paper_id": rp.paper_id,
                            "source": {
                                "type": "abstract_fallback",
                                "url": url,
                                "extraction_method": "abstract",
                                "fallback_reason": "pdf_download_failed"
                            },
                            "paper": {
                                "title": rp.title,
                                "paper_id": rp.paper_id,
                                "authors": getattr(rp, "authors", []),
                                "year": getattr(rp, "year", None),
                                "venue": getattr(rp, "venue", None)
                            }
                        }
                    )
                
                return rp.abstract, "abstract", "download_failed"
            return None, None, "download_failed_no_abstract"
        
        # Extract text from downloaded PDF
        try:
            text = pdf_processor.extract_text(
                Path(dest),
                truncate_refs=True,
                max_chars=MAX_CONTEXT_CHARS,
                logger=self.logger,
            )
            
            # Cache the extracted text if TextCache is available
            if text and self.text_cache and rp.canonical_id:
                self.text_cache.cache_text(
                    canonical_id=rp.canonical_id,
                    full_text=text,
                    metadata={
                        "title": rp.title,
                        "paper_id": rp.paper_id,
                        "source": {
                            "type": "pdf_full",
                            "url": url,
                            "extraction_method": "pypdf",
                            "fallback_reason": None
                        },
                        "paper": {
                            "title": rp.title,
                            "paper_id": rp.paper_id,
                            "authors": getattr(rp, "authors", []),
                            "year": getattr(rp, "year", None),
                            "venue": getattr(rp, "venue", None)
                        }
                    }
                )
            
            # Clean up downloaded PDF (if not keeping it)
            if not PHASE3_KEEP_DOWNLOADED_PDFS:
                try:
                    if dest and os.path.exists(dest):
                        os.remove(dest)
                except Exception:
                    pass
            
            if text:
                src_type = (
                    "constructed_pdf"
                    if getattr(rp, "note", "") == "constructed_pdf_from_arxiv"
                    else "fulltext"
                )
                return text, src_type, None
            else:
                # If extraction produced no text, fall back to abstract when available
                self.logger.warning(
                    "[Candidate] PDF extraction yielded no text; falling back to abstract if available"
                )
                if getattr(rp, "abstract", None):
                    # Cache abstract to avoid repeated extraction attempts
                    if self.text_cache and rp.canonical_id:
                        self.text_cache.cache_text(
                            canonical_id=rp.canonical_id,
                            full_text=rp.abstract,
                            metadata={
                                "title": rp.title,
                                "paper_id": rp.paper_id,
                                "source": {
                                    "type": "abstract_fallback",
                                    "url": url,
                                    "extraction_method": "abstract",
                                    "fallback_reason": "pdf_extraction_empty"
                                },
                                "paper": {
                                    "title": rp.title,
                                    "paper_id": rp.paper_id,
                                    "authors": getattr(rp, "authors", []),
                                    "year": getattr(rp, "year", None),
                                    "venue": getattr(rp, "venue", None)
                                }
                            }
                        )
                    
                    return rp.abstract, "abstract", "extraction_failed"
                return None, None, "extraction_failed"
        except Exception as e:
            self.logger.error(f"[Candidate] PDF extraction error: {e}")
            if getattr(rp, "abstract", None):
                return rp.abstract, "abstract", f"extraction_error: {e}"
            return None, None, f"extraction_error: {e}"
    
    def resolve_candidate_pdf_url(self, rp: RetrievedPaper) -> Optional[str]:
        """Best-effort inference of a candidate's PDF URL using shared resolver."""
        # Delegate heavy lifting to the shared resolver in services/pdf_resolver.py
        res = resolve_pdf_url_for_candidate(rp, logger=self.logger)
        if res.status == "ok" and res.pdf_url:
            # Propagate note back to the candidate object for downstream inspection
            if res.note and not getattr(rp, "note", None):
                rp.note = res.note
            return res.pdf_url

        # Fallback: if resolver failed but an explicit pdf_url is present, use it as-is
        explicit_pdf = getattr(rp, "pdf_url", None)
        if explicit_pdf:
            return explicit_pdf

        # Otherwise, give up – Phase3 will fall back to abstract
        return None
    
    def download_batch(
        self,
        papers: List[RetrievedPaper],
        max_workers: int = 3
    ) -> Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """
        Download PDFs and extract texts for multiple papers in parallel.
        
        Args:
            papers: List of RetrievedPaper objects
            max_workers: Maximum number of concurrent downloads (default: 3)
            
        Returns:
            Dict mapping canonical_id to (full_text, source_type, skip_reason)
            - full_text: Extracted text or abstract fallback
            - source_type: 'pdf_full', 'cached', 'abstract_fallback', 'error'
            - skip_reason: Error message if failed, None otherwise
        """
        results: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {}
        
        if not papers:
            return results
        
        self.logger.info(f"Starting parallel download of {len(papers)} papers (max_workers={max_workers})")
        
        def download_one(paper: RetrievedPaper) -> Tuple[str, str, str, str]:
            """Download a single paper, returns (canonical_id, text, source, reason)."""
            cid = paper.canonical_id or paper.paper_id
            try:
                text, source, reason = self.get_candidate_full_text(paper)
                if text:
                    return (cid, text, source or "pdf_full", reason)
                else:
                    # Fallback to abstract
                    abstract = paper.abstract or ""
                    return (cid, abstract, "abstract_fallback", reason or "no_pdf_text")
            except Exception as e:
                # Error fallback to abstract
                abstract = paper.abstract or ""
                return (cid, abstract, "error", str(e))
        
        # Execute in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one, paper): paper for paper in papers}
            
            for future in as_completed(futures):
                paper = futures[future]
                cid = paper.canonical_id or paper.paper_id
                completed += 1
                
                try:
                    cid, text, source, reason = future.result()
                    results[cid] = (text, source, reason)
                    
                    # Log progress
                    text_len = len(text) if text else 0
                    if source == "abstract_fallback" or source == "error":
                        self.logger.warning(
                            f"[{completed}/{len(papers)}] {cid[:20]}... -> {source} ({reason})"
                        )
                    else:
                        self.logger.info(
                            f"[{completed}/{len(papers)}] {cid[:20]}... -> {source} ({text_len} chars)"
                        )
                except Exception as e:
                    # Unexpected error - use abstract fallback
                    abstract = paper.abstract or ""
                    results[cid] = (abstract, "error", str(e))
                    self.logger.error(f"[{completed}/{len(papers)}] {cid[:20]}... -> error: {e}")
        
        # Summary
        success = sum(1 for _, src, _ in results.values() if src in ("pdf_full", "cached"))
        fallback = sum(1 for _, src, _ in results.values() if src == "abstract_fallback")
        errors = sum(1 for _, src, _ in results.values() if src == "error")
        
        self.logger.info(
            f"Parallel download complete: {success} success, {fallback} fallback, {errors} errors"
        )
        
        return results
    
    # ========================================================================
    # PRIVATE METHODS - File Management
    # ========================================================================

