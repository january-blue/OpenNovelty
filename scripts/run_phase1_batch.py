#!/usr/bin/env python3
"""
Interactive / CLI batch runner for Phase1 (fulltext + LLM structuring).

Usage examples:

1) Fully interactive:
    python scripts/run_phase1_batch.py

2) Pass papers via CLI:
    python scripts/run_phase1_batch.py --papers \
        https://openreview.net/pdf?id=tP01GCr0nS \
        https://openreview.net/pdf?id=meRHki4HwQ \
        ...

3) Read from a file (one id or URL per line):
    python scripts/run_phase1_batch.py --paper-file papers.txt

Each paper will be saved under:
    <out-root>/<run-prefix>/openreview_<id>_<YYYYMMDD>/phase1/
with files:
    - paper.json
    - phase1_extracted.json
    - original_fulltext.txt
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from pathlib import Path
from dataclasses import asdict
import json
import re
import shutil
from datetime import datetime

from paper_novelty_pipeline.models import PaperInput
from paper_novelty_pipeline.phases.phase1.orchestrator import ContentExtractionPhase
from paper_novelty_pipeline.utils.paths import safe_dir_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Phase1 runner (interactive or from CLI).")
    parser.add_argument(
        "--papers",
        nargs="+",
        help="Paper ids or URLs, e.g. https://openreview.net/pdf?id=XXX",
    )
    parser.add_argument(
        "--paper-file",
        type=str,
        help="Path to a text file; each line is one paper id or URL.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Max concurrent Phase1 workers.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="output",
        help="Root directory for batch outputs, e.g. 'output/fdu_iclr_2026'.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="",
        help="Optional extra prefix under out-root, e.g. 'iclr_2026'. "
             "If set, final root is '<out-root>/<run-prefix>'.",
    )
    parser.add_argument(
        "--force-year",
        type=int,
        help="Force the publication year for all papers (e.g. 2026). "
             "This overrides automatic detection and affects canonical_id generation.",
    )
    return parser.parse_args()


def collect_papers_interactively() -> List[PaperInput]:
    print("Enter papers to run Phase1 on (OpenReview id or PDF URL).")
    print("One per line. Empty line to finish. Example:")
    print("  https://openreview.net/pdf?id=tP01GCr0nS")
    print("  https://openreview.net/pdf?id=meRHki4HwQ")
    print("Press Enter on an empty line to stop.\n")

    papers: List[PaperInput] = []
    while True:
        try:
            line = input("paper id/url (empty line to finish): ").strip()
        except EOFError:
            # Ctrl+D to finish
            break
        if not line:
            break
        # 仅把输入当作 paper_id（通常是 PDF URL），标题留空，后续由 Phase1 填写真实标题
        papers.append(PaperInput(paper_id=line, title=""))

    return papers


def collect_papers_from_file(path: str) -> List[PaperInput]:
    papers: List[PaperInput] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # 文件中每一行视为 paper_id（PDF URL 或本地路径），标题留空，后续由 Phase1 填充
            papers.append(PaperInput(paper_id=line, title=""))
    return papers


def _make_out_dir_for_paper(base_root: Path, date_str: str, paper_id: str) -> Path:
    """
    Create an output directory for a single paper, using centralized safe_dir_name logic.
    """
    base_name = safe_dir_name(paper_id)
    dir_name = f"{base_name}_{date_str}"
    out_dir = base_root / dir_name / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _get_processed_ids(base_root: Path, log: logging.Logger) -> set:
    """
    Scan base_root for existing output directories and extract paper IDs.
    Supports any prefix (openreview, local, arxiv, acl, ijcai, etc.)
    """
    processed_ids = set()
    if not base_root.exists():
        return processed_ids
    
    # Generic pattern: {prefix}_{id}_{YYYYMMDD}
    # We look for folders ending with 8 digits, preceded by an underscore.
    # The part before that is the base_name (prefix_id).
    # Regex: ^(.*)_(\d{8})$
    pattern = re.compile(r"^(.*)_(\d{8})$")
    
    for item in base_root.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                base_name = match.group(1)
                # Note: safe_dir_name handles prefix+id, so we store and compare base_name
                processed_ids.add(base_name)
                log.debug(f"Found processed paper: {base_name} (from {item.name})")
    
    return processed_ids


def _persist_phase1_outputs(paper: PaperInput, extracted, out_dir: Path, log: logging.Logger) -> None:
    """
    Save paper.json and phase1_extracted.json under out_dir.
    """
    # Save paper + extraction
    with open(out_dir / "paper.json", "w", encoding="utf-8") as f:
        json.dump(asdict(paper), f, ensure_ascii=False, indent=2)
    data = asdict(extracted)
    data.pop("core_task_survey", None)
    with open(out_dir / "phase1_extracted.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Note: fulltext debug files and pub_date.json are now written directly
    # by Phase1 into out_dir (phase1_dir) and no longer go through global locations.


def run_one(phase1: ContentExtractionPhase, paper: PaperInput, base_root: Path, date_str: str, force_year: int = None):
    log = logging.getLogger("run_phase1_batch")
    t0 = time.time()
    try:
        log.info(f"[Phase1] start: {paper.paper_id}")
        out_dir = _make_out_dir_for_paper(base_root, date_str, paper.paper_id)
        
        # If force_year is set, we set it on paper input first
        if force_year:
            paper.year = force_year
            
        extracted = phase1.extract_content(paper, phase1_dir=out_dir)
        
        # If extracted successfully and force_year is set, we ensure it's overridden
        # because extract_content might have overwritten paper.year with detected values.
        if extracted and force_year:
            paper.year = force_year
            
            # Re-calculate canonical_id since it depends on year
            from paper_novelty_pipeline.utils.paper_id import make_canonical_id
            new_id = make_canonical_id(
                paper_id=paper.paper_id,
                doi=getattr(paper, "doi", None),
                arxiv_id=getattr(paper, "arxiv_id", None),
                url=getattr(paper, "original_pdf_url", None) or paper.paper_id,
                title=paper.title,
                year=paper.year
            )
            paper.canonical_id = new_id
            extracted.canonical_id = new_id
            
            # Overwrite pub_date.json
            pub_info = {
                "year": force_year,
                "month": None,
                "day": None,
                "granularity": "year",
                "source": "force_year_cli",
                "confidence": 1.0
            }
            with open(out_dir / "pub_date.json", "w", encoding="utf-8") as f:
                json.dump(pub_info, f, ensure_ascii=False, indent=2)
            log.info(f"[Phase1] Forced year to {force_year} and updated canonical_id to {new_id}")

        dt = time.time() - t0
        if extracted is None:
            log.error(f"[Phase1] FAILED: {paper.paper_id} (duration={dt:.1f}s)")
            return paper.paper_id, False, dt
        else:
            _persist_phase1_outputs(paper, extracted, out_dir, log)

            log.info(f"[Phase1] done: {paper.paper_id} (duration={dt:.1f}s, out_dir={out_dir})")
            return paper.paper_id, True, dt
    except Exception as e:
        dt = time.time() - t0
        log.exception(f"[Phase1] EXCEPTION for {paper.paper_id} (duration={dt:.1f}s): {e}")
        return paper.paper_id, False, dt


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("run_phase1_batch")

    # Compute base_root: <out-root>[/<run-prefix>]
    out_root = Path(args.out_root)
    if args.run_prefix:
        base_root = out_root / args.run_prefix
    else:
        base_root = out_root
    base_root.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    log.info(f"Base output root for Phase1 batch: {base_root} (date={date_str})")

    papers: List[PaperInput] = []

    # 1) Use CLI --papers if provided
    if args.papers:
        # 仅把传入的字符串视为 paper_id（PDF URL），标题留空
        papers = [PaperInput(paper_id=p, title="") for p in args.papers]

    # 2) Append from --paper-file if provided
    if args.paper_file:
        from_file = collect_papers_from_file(args.paper_file)
        papers.extend(from_file)

    # 3) If still empty, fall back to interactive mode
    if not papers:
        papers = collect_papers_interactively()

    if not papers:
        log.error("No paper ids/urls provided. Exiting.")
        sys.exit(1)

    # Deduplicate by paper_id
    seen = set()
    unique_papers: List[PaperInput] = []
    for p in papers:
        if p.paper_id not in seen:
            seen.add(p.paper_id)
            unique_papers.append(p)

    # Get already processed paper IDs (resume from checkpoint)
    processed_ids = _get_processed_ids(base_root, log)
    if processed_ids:
        log.info(f"Found {len(processed_ids)} already processed papers in {base_root}")
        # Filter out already processed papers
        papers_to_run = [
            p for p in unique_papers 
            if safe_dir_name(p.paper_id) not in processed_ids
        ]
        skipped = len(unique_papers) - len(papers_to_run)
        if skipped > 0:
            log.info(f"Skipping {skipped} already processed papers")
        unique_papers = papers_to_run

    if not unique_papers:
        log.info("All papers have already been processed. Nothing to do.")
        sys.exit(0)

    log.info(f"Total papers to run Phase1: {len(unique_papers)}, max_workers={args.max_workers}")
    for p in unique_papers:
        log.info(f"  - {p.paper_id}")

    phase1 = ContentExtractionPhase()

    t_global0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        future_map = {
            ex.submit(run_one, phase1, p, base_root, date_str, force_year=args.force_year): p
            for p in unique_papers
        }
        for fut in as_completed(future_map):
            results.append(fut.result())

    dt_global = time.time() - t_global0
    success = sum(1 for _, ok, _ in results if ok)
    fail = len(results) - success
    log.info(f"Phase1 batch done. success={success}, fail={fail}, total_time={dt_global/60:.1f} min")

    for pid, ok, dt in results:
        log.info(f"  - {pid}: {'OK' if ok else 'FAIL'} ({dt:.1f}s)")


if __name__ == "__main__":
    main()
