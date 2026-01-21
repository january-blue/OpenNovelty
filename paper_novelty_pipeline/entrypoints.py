"""
Library entrypoints for the Paper Novelty Pipeline.

Goal:
- Provide a single "true" in-process entry for running the pipeline (CLI calls here).
- Keep existing scripts as thin CLI wrappers for backward compatibility.

Important compatibility notes:
- Some modules read `PDF_DOWNLOAD_DIR` from env at import time (via config constants).
  To preserve per-run temp PDF isolation (previously achieved via subprocess env),
  this module provides `set_pdf_download_dir()` which updates env AND patches
  already-imported module-level constants when possible.
- Many scripts rely on relative paths (e.g. default output dir "output").
  This module provides `_temp_cwd()` to emulate prior subprocess `cwd=...`.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------- small compat helpers -----------------------------

# Import path constants from centralized utils (single source of truth)
from paper_novelty_pipeline.utils.paths import (  # noqa: E402
    safe_dir_name,
    PAPER_JSON,
    PHASE1_EXTRACTED_JSON,
    PUB_DATE_JSON,
    PHASE2_SEARCH_RESULTS_JSON,
)
from paper_novelty_pipeline.config import PHASE2_QUERY_CONCURRENCY  # noqa: E402


@contextlib.contextmanager
def _temp_cwd(path: Optional[Path]):
    """Temporarily chdir to `path` (restores original cwd)."""
    if not path:
        yield
        return
    prev = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))


def set_pdf_download_dir(dir_path: Optional[Path]) -> None:
    """
    Best-effort override of PDF download directory for the current process.

    This preserves the semantics of prior subprocess-based isolation by:
    - setting env var `PDF_DOWNLOAD_DIR`
    - updating `paper_novelty_pipeline.config.PDF_DOWNLOAD_DIR` if loaded
    - updating module-level imports of `PDF_DOWNLOAD_DIR` in known modules if loaded
    """
    if not dir_path:
        return
    dp = Path(dir_path).resolve()
    os.environ["PDF_DOWNLOAD_DIR"] = str(dp)

    # Patch already-imported modules (best effort; safe if missing).
    try:
        import paper_novelty_pipeline.config as _cfg

        _cfg.PDF_DOWNLOAD_DIR = dp  # type: ignore[attr-defined]
    except Exception:
        pass

    for mod_name in (
        "paper_novelty_pipeline.phases.phase1.orchestrator",
        "paper_novelty_pipeline.phases.phase1.pdf_text_loader",
        "paper_novelty_pipeline.services.pdf_processor",
        "paper_novelty_pipeline.utils.pdf_handler",
    ):
        try:
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, "PDF_DOWNLOAD_DIR"):
                setattr(mod, "PDF_DOWNLOAD_DIR", dp)
        except Exception:
            pass


def _parse_pub_date(s: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    try:
        if not s:
            return None, None
        parts = s.strip().split("-")
        y = int(parts[0]) if parts and parts[0] else None
        m = int(parts[1]) if len(parts) > 1 else None
        if y and (m is None or (1 <= m <= 12)):
            return y, m
    except Exception:
        return None, None
    return None, None


def _infer_pub_date_from_url(u: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        if "arxiv.org" in (u or ""):
            m = re.search(r"/(abs|pdf)/(\d{2})(\d{2})\.", u)
            if m:
                yy = int(m.group(2))
                mm = int(m.group(3))
                return 2000 + yy, (mm if 1 <= mm <= 12 else None)
    except Exception:
        pass
    return None, None


# ----------------------------- common helpers (DRY) -----------------------------

def _log_contributions_summary(extracted: Any, log: logging.Logger) -> None:
    """Log a summary of extracted contributions (shared by Phase1 entrypoints)."""
    try:
        contributions = getattr(extracted, "contributions", []) or []
        if contributions:
            names = [
                getattr(c, "name", "").strip()
                for c in contributions
                if getattr(c, "name", "").strip()
            ]
            preview = ", ".join(names[:3]) if names else ""
            summary = f"contributions={len(contributions)}"
            if preview:
                summary += f" ({preview})"
            log.info("Extracted contributions: " + summary)
        else:
            log.info("Extracted contributions: none found")
    except Exception as e:
        log.debug(f"Failed to summarize extracted contributions: {e}")


def _save_phase1_outputs(paper: Any, extracted: Any, phase1_dir: Path) -> None:
    """Save paper.json and phase1_extracted.json (shared by Phase1 entrypoints)."""
    with open(phase1_dir / PAPER_JSON, "w", encoding="utf-8") as f:
        json.dump(asdict(paper), f, ensure_ascii=False, indent=2)
    extracted_dict = asdict(extracted)
    extracted_dict.pop("core_task_survey", None)
    with open(phase1_dir / PHASE1_EXTRACTED_JSON, "w", encoding="utf-8") as f:
        json.dump(extracted_dict, f, ensure_ascii=False, indent=2)


# ----------------------------- entrypoint: Phase1 only -----------------------------

def run_phase1_only(
    *,
    paper_url: str,
    out_dir: str = "output",
    log_level: str = "INFO",
    paper_title: Optional[str] = None,
    cwd: Optional[Path] = None,
    pdf_download_dir: Optional[Path] = None,
) -> int:
    """In-process equivalent of `scripts/run_phase1_only.py`."""
    set_pdf_download_dir(pdf_download_dir)
    with _temp_cwd(cwd):
        from paper_novelty_pipeline.models import PaperInput
        from paper_novelty_pipeline.phases.phase1.orchestrator import ContentExtractionPhase
        from paper_novelty_pipeline.utils.logger import setup_logger

        setup_logger(level=log_level)
        log = logging.getLogger("run_phase1_only")

        log.info(f"Starting Phase 1 for paper: {paper_url}")
        paper = PaperInput(paper_id=paper_url, title=(paper_title or ""))

        phase1 = ContentExtractionPhase()
        log.info("Running Phase 1 (extraction)...")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        phase1_dir = out_path
        extracted = phase1.extract_content(paper, phase1_dir=phase1_dir)
        if not extracted:
            log.error("Phase 1 failed. Exiting.")
            return 1

        _log_contributions_summary(extracted, log)
        _save_phase1_outputs(paper, extracted, phase1_dir)

        # Copy local PDF into out dir (same behavior)
        try:
            if isinstance(paper.paper_id, str) and os.path.exists(paper.paper_id):
                pdf_src = Path(paper.paper_id)
                pdf_dst = phase1_dir / pdf_src.name
                if not pdf_dst.exists():
                    import shutil

                    shutil.copy(pdf_src, pdf_dst)
                    log.info(f"Copied original PDF to {pdf_dst}")
        except Exception as e:
            log.debug(f"Failed to copy original PDF into out_dir: {e}")

        print("\n== Phase 1 done ==")
        print(f"Saved: {out_path / PAPER_JSON}, {out_path / PHASE1_EXTRACTED_JSON}")
        return 0


# ----------------------------- entrypoint: Phase2 (full) -----------------------------

def run_phase2_full(
    *,
    phase1_dir: Optional[Path] = None,
    extracted_json: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    paper_title: Optional[str] = None,
    pub_date: Optional[str] = None,
    max_per_query: int = 75,
    max_per_query_core_task: int = 30,
    max_per_query_contrib: int = 15,
    concurrency: int = int(PHASE2_QUERY_CONCURRENCY),
    filter_self: bool = True,
    core_topk: int = 50,
    contrib_topk: int = 10,
    final_subdir: str = "final",
    log_level: str = "INFO",
    cwd: Optional[Path] = None,
    pdf_download_dir: Optional[Path] = None,
) -> int:
    """
    In-process equivalent of `scripts/run_phase2_only.py` (search + postprocess).

    Note: It still invokes `postprocess_phase2_topk.py` via subprocess, to minimize
    behavioral risk. This is internal to Phase2 and does not affect the "single true
    entry" requirement for the top-level CLI.
    """
    set_pdf_download_dir(pdf_download_dir)
    with _temp_cwd(cwd):
        # Local imports to keep env-based config overrides viable
        from paper_novelty_pipeline.models import ExtractedContent, ContributionClaim, CoreTask
        from paper_novelty_pipeline.phases.phase2.searching import PaperSearcher
        from paper_novelty_pipeline.utils.logger import setup_logger

        def _parse_extracted_data(data: dict) -> ExtractedContent:
            """Parse extracted JSON data into ExtractedContent object."""
            ct = data.get("core_task") or {}
            core_task = CoreTask(text=ct.get("text", ""), query_variants=ct.get("query_variants", []) or [])
            contribs: List[ContributionClaim] = []
            for i, c in enumerate(data.get("contributions", []) or [], 1):
                contribs.append(
                    ContributionClaim(
                        id=c.get("id") or f"c{i}",
                        name=c.get("name", ""),
                        author_claim_text=c.get("author_claim_text", ""),
                        description=c.get("description", ""),
                        prior_work_query=c.get("prior_work_query", ""),
                        query_variants=c.get("query_variants", []) or [],
                        source_hint=c.get("source_hint", "unknown"),
                    )
                )
            return ExtractedContent(core_task=core_task, core_task_survey=None, contributions=contribs)

        def _load_extracted_via_phase1_dir(p1: Path) -> Tuple[ExtractedContent, dict]:
            data = json.load(open(p1 / PHASE1_EXTRACTED_JSON, "r", encoding="utf-8"))
            extracted = _parse_extracted_data(data)
            meta = {"paper": {}, "pub_date": {}}
            try:
                meta["paper"] = json.load(open(p1 / PAPER_JSON, "r", encoding="utf-8"))
            except Exception:
                pass
            try:
                meta["pub_date"] = json.load(open(p1 / PUB_DATE_JSON, "r", encoding="utf-8"))
            except Exception:
                pass
            return extracted, meta

        def _load_extracted_via_file(p: Path) -> ExtractedContent:
            data = json.load(open(p, "r", encoding="utf-8"))
            return _parse_extracted_data(data)

        setup_logger(level=log_level)
        log = logging.getLogger("run_phase2_full")

        if not phase1_dir and not extracted_json:
            raise ValueError("Provide phase1_dir or extracted_json")

        # Determine base_out
        if out_dir:
            base_out = Path(out_dir)
        elif phase1_dir:
            base_out = Path(phase1_dir).parent
        else:
            base_out = Path("output")

        # Load Phase1 extraction
        if phase1_dir:
            extracted, meta = _load_extracted_via_phase1_dir(Path(phase1_dir))
        else:
            extracted = _load_extracted_via_file(Path(extracted_json))  # type: ignore[arg-type]
            meta = {"paper": {}, "pub_date": {}}
            # Try read pub_date.json next to extracted-json when under phase1/
            try:
                pj = Path(extracted_json).resolve()  # type: ignore[arg-type]
                if pj.parent.name == "phase1":
                    pd = pj.parent / "pub_date.json"
                    if pd.exists():
                        meta["pub_date"] = json.load(open(pd, "r", encoding="utf-8"))
            except Exception:
                pass

        # Initialize searcher
        searcher = PaperSearcher(concurrency=max(1, int(concurrency)) if concurrency else None)
        
        # Self-filter info for postprocess
        self_title = (paper_title or meta.get("paper", {}).get("title") or "").strip() or None

        # Publication cutoff: CLI > Phase1 meta
        y, m = _parse_pub_date(pub_date)
        if not y:
            pd = meta.get("pub_date", {})
            try:
                y = int(pd.get("year")) if pd.get("year") else None
                mm = pd.get("month")
                m = int(mm) if mm and 1 <= int(mm) <= 12 else None
            except Exception:
                y, m = None, None
        if y:
            log.info(f"Year filter enabled: cutoff={y}-{m or 0}")
        else:
            log.warning("Year filter disabled: no publication date available")

        phase2_dir = Path(base_out) / "phase2"
        phase2_dir.mkdir(parents=True, exist_ok=True)

        # Run candidate search
        log.info("Phase2: searching candidates...")
        search_stats = searcher.search_all(extracted, phase2_dir)
        
        if search_stats['failed'] > 0:
            log.warning(f"Phase2 search had {search_stats['failed']} failures out of {search_stats['total_queries']} queries")

        # Postprocess in-process (no subprocess)
        log.info(
            f"Running Phase2 postprocess in-process: phase2_dir={phase2_dir} "
            f"core_topk={core_topk} contrib_topk={contrib_topk} final_subdir={final_subdir}"
        )
        try:
            from paper_novelty_pipeline.phases.phase2.postprocess import postprocess_phase2_outputs

            postprocess_phase2_outputs(
                phase2_dir,
                core_topk=int(core_topk),
                contrib_topk=int(contrib_topk),
                final_subdir=str(final_subdir),
            )
            log.info("Phase2 postprocess completed successfully.")
        except Exception as e:
            log.error(f"Phase2 postprocess failed: {e}")

        print("\n== Phase 2 (full: search + postprocess) done ==")
        print(f"Phase2 directory: {phase2_dir}")
        print(f"Final outputs (TopK + citation_index) under: {phase2_dir / final_subdir}")
        return 0


# ----------------------------- entrypoint: Phase3 only -----------------------------

def run_phase3_only(
    *,
    paper_json: Path,
    extracted_json: Path,
    search_results_json: Path,
    run_dir: Path,
    phase3_top_k: Optional[int] = None,
    candidate_filter: Optional[str] = None,
    log_level: str = "INFO",
    cwd: Optional[Path] = None,
    pdf_download_dir: Optional[Path] = None,
) -> int:
    """In-process equivalent of `scripts/run_phase3_only.py`."""
    set_pdf_download_dir(pdf_download_dir)
    with _temp_cwd(cwd):
        from paper_novelty_pipeline.models import (
            PaperInput,
            ExtractedContent,
            SearchQuery,
            RetrievedPaper,
            SearchResult,
            ComparisonResult,
            Slot,
            Subslot,
        )
        from paper_novelty_pipeline.phases import NoveltyComparisonPhase
        from paper_novelty_pipeline.utils.logger import setup_logger

        def _load_json(path: Path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        def _dict_to_paper_input(d: dict) -> PaperInput:
            return PaperInput(
                paper_id=d.get("paper_id"),
                title=d.get("title", ""),
                abstract=d.get("abstract"),
                authors=d.get("authors"),
                venue=d.get("venue"),
                year=d.get("year"),
                doi=d.get("doi"),
                arxiv_id=d.get("arxiv_id"),
                keywords=d.get("keywords"),
                primary_area=d.get("primary_area"),
                openreview_rating_mean=d.get("openreview_rating_mean"),
            )

        def _dict_to_extracted_content(d: dict) -> ExtractedContent:
            slots: List[Slot] = []
            for slot_d in d.get("slots", []) or []:
                aspect = slot_d.get("aspect")
                subslots: List[Subslot] = []
                for sub_d in slot_d.get("subslots", []) or []:
                    subslots.append(
                        Subslot(
                            aspect=sub_d.get("aspect") or aspect,
                            name=sub_d.get("name", "") or "",
                            description=sub_d.get("description", "") or "",
                            prior_work_query=sub_d.get("prior_work_query", "") or "",
                            source_hint=sub_d.get("source_hint"),
                        )
                    )
                if aspect and subslots:
                    slots.append(Slot(aspect=aspect, subslots=subslots))
            return ExtractedContent(slots=slots)

        def _dict_to_search_results(data: List[dict]) -> List[SearchResult]:
            results: List[SearchResult] = []
            for item in data:
                q = item.get("query", {})
                query = SearchQuery(
                    query=q.get("query", ""),
                    aspect=q.get("aspect", "task"),
                    subslot_name=q.get("subslot_name", ""),
                    source_description=q.get("source_description", ""),
                )
                rps = []
                for rp in item.get("retrieved_papers", []):
                    source_url = rp.get("source_url") or rp.get("url") or rp.get("link")
                    rps.append(
                        RetrievedPaper(
                            paper_id=rp.get("paper_id", ""),
                            title=rp.get("title", ""),
                            abstract=rp.get("abstract", ""),
                            authors=rp.get("authors", []) or [],
                            venue=rp.get("venue", ""),
                            year=rp.get("year", 0),
                            doi=rp.get("doi"),
                            arxiv_id=rp.get("arxiv_id"),
                            relevance_score=float(rp.get("relevance_score", 0.0)),
                            pdf_url=rp.get("pdf_url"),
                            source_url=source_url,
                            raw_metadata=rp,
                        )
                    )
                results.append(
                    SearchResult(
                        query=query,
                        retrieved_papers=rps,
                        total_results=int(item.get("total_results", len(rps))),
                    )
                )
            return results

        setup_logger(level=log_level)
        log = logging.getLogger("run_phase3_only")

        paper = _dict_to_paper_input(_load_json(Path(paper_json)))
        extracted = _dict_to_extracted_content(_load_json(Path(extracted_json)))
        search_results = _dict_to_search_results(_load_json(Path(search_results_json)))

        if candidate_filter:
            log.info(f"Filtering candidates to only: {candidate_filter}")
            filtered_results = []
            for sr in search_results:
                filtered_papers = [
                    rp
                    for rp in sr.retrieved_papers
                    if candidate_filter in rp.paper_id or candidate_filter in (rp.source_url or "")
                ]
                if filtered_papers:
                    filtered_results.append(
                        SearchResult(query=sr.query, retrieved_papers=filtered_papers, total_results=len(filtered_papers))
                    )
            search_results = filtered_results
            log.info(
                f"After filtering: {len(search_results)} search results with "
                f"{sum(len(sr.retrieved_papers) for sr in search_results)} total candidates"
            )

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        phase3 = NoveltyComparisonPhase(output_base_dir=str(run_dir))
        if phase3_top_k is not None:
            phase3.top_k = int(phase3_top_k)
            log.info(f"Phase3 top_k overridden to {phase3.top_k}")

        log.info("Running Phase 3 comparison ...")
        comp_results: List[ComparisonResult] = phase3.compare_papers(paper, extracted, search_results)

        print("\n== Phase3 results ==")
        print(f"comparisons: {len(comp_results)}")
        for r in comp_results:
            print(f"\n- candidate: {r.candidate_paper_title} (ID: {r.retrieved_paper_id})")
            if r.comparison_note:
                print(f"  note: {r.comparison_note}")
            if r.textual_similarity_segments:
                print(f"  WARNING: Found {len(r.textual_similarity_segments)} high-similarity segments!")
                for seg in r.textual_similarity_segments:
                    print(f"    - {seg.segment_type}: score={seg.similarity_score:.2f}, words={seg.word_count}")
                    print(f"      Location: {seg.original_location} <-> {seg.candidate_location}")
            for slot in r.analyzed_slots:
                print(f"  [Subslot: {slot.subslot_name} (Aspect: {slot.aspect})]")
                # Display new fields
                print(f"    Refutation Status: {slot.refutation_status}")
                if slot.refutation_status == "can_refute" and slot.refutation_evidence:
                    ref_summary = slot.refutation_evidence.get("summary", "")
                    print(
                        f"    Refutation Summary: {ref_summary[:200]}..."
                        if len(ref_summary) > 200
                        else f"    Refutation Summary: {ref_summary}"
                    )
                    ref_pairs = slot.refutation_evidence.get("evidence_pairs", [])
                    print(f"    Refutation Evidence: {len(ref_pairs)} pairs")
                elif slot.brief_note:
                    print(f"    Brief Note: {slot.brief_note}")

        return 0


# ----------------------------- entrypoint: full pipeline (Phase1+2(+3)) -----------------------------

def cleanup_repo(*, move_runs: bool = False) -> int:
    """
    Minimal housekeeping utility for the CLI `cleanup` command.
    (A previous subprocess target `scripts.cleanup_repo` did not exist.)
    """
    try:
        repo_root = Path(__file__).resolve().parents[1]
        archive_dir = repo_root / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        if move_runs:
            # Move a top-level `runs/` directory if present (legacy).
            src = repo_root / "runs"
            if src.exists() and src.is_dir():
                ts = time.strftime("%Y%m%d_%H%M%S")
                dst = archive_dir / f"runs_{ts}"
                src.rename(dst)
                print(f"Moved {src} -> {dst}")
            else:
                print("No top-level runs/ directory found; nothing to move.")
        return 0
    except Exception as e:
        print(f"cleanup failed: {e}")
        return 1

# ----------------------------- entrypoint: batch runner -----------------------------

