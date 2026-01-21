#!/usr/bin/env python3
"""
Run Phase 2 (search + postprocess) for a single paper.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from paper_novelty_pipeline.entrypoints import run_phase2_full
from paper_novelty_pipeline.config import PHASE2_QUERY_CONCURRENCY


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 (search + postprocess) using Phase1 outputs",
    )
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        required=False,
        help="Path to Phase1 directory containing phase1_extracted.json",
    )
    parser.add_argument(
        "--extracted-json",
        type=Path,
        required=False,
        help="Path to phase1_extracted.json (if not using --phase1-dir)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=False,
        help="Base out directory; results go to <out-dir>/phase2",
    )
    parser.add_argument(
        "--paper-title",
        type=str,
        default=None,
        help="Override paper title for self-filter (else read from Phase1 paper.json)",
    )
    parser.add_argument(
        "--pub-date",
        type=str,
        default=None,
        help="YYYY or YYYY-MM; candidates strictly after this are dropped",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(PHASE2_QUERY_CONCURRENCY),
        help="Phase2 concurrent queries (threads)",
    )
    parser.add_argument(
        "--no-filter-self",
        dest="filter_self",
        action="store_false",
        help="Disable self-filter (default: enabled)",
    )
    parser.add_argument(
        "--core-topk",
        type=int,
        default=50,
        help="TopK to keep for core task finals (default: 50)",
    )
    parser.add_argument(
        "--contrib-topk",
        type=int,
        default=10,
        help="TopK to keep for each contribution finals (default: 10)",
    )
    parser.add_argument(
        "--final-subdir",
        type=str,
        default="final",
        help="Subdirectory name under phase2_dir to write final files (default: final)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    args = parser.parse_args()
    if not hasattr(args, "filter_self") or args.filter_self is None:
        args.filter_self = True

    if not args.phase1_dir and not args.extracted_json:
        parser.error("Provide --phase1-dir or --extracted-json")

    rc = run_phase2_full(
        phase1_dir=args.phase1_dir,
        extracted_json=args.extracted_json,
        out_dir=args.out_dir,
        paper_title=args.paper_title,
        pub_date=args.pub_date,
        concurrency=args.concurrency,
        filter_self=bool(args.filter_self),
        core_topk=args.core_topk,
        contrib_topk=args.contrib_topk,
        final_subdir=args.final_subdir,
        log_level=args.log_level,
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()


