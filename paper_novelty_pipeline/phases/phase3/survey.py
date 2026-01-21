from __future__ import annotations

"""
Phase 3 (Short Survey).

This module builds the short core-task survey for Phase 3:

Step 1  (build_initial_report):
  - Prepare phase3/core_task_survey/survey_report.json with placeholders.
  - Dump basic TopK (Top50) appendix.

Step 2  (build_taxonomy_via_llm):
  - Call LLM once to build a hierarchical taxonomy over the TopK papers.
  - Validate coverage + MECE-ish constraints.
  - Store taxonomy + per-paper taxonomy_path + display_index.

Step 3  (build_narrative_via_llm):
  - Call LLM once to generate a concise 2-paragraph narrative + original-paper
    position commentary.
  - Render a markdown report containing:
      - Core task (one line)
      - Text-only taxonomy tree
      - Narrative (2 paragraphs, no hard numbers)
      - Original paper position (path + neighbors + commentary)

Mermaid figures are no longer generated here; front-end is expected to render
visuals from taxonomy JSON + mapping.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from paper_novelty_pipeline.utils.paths import PAPER_JSON, PHASE1_EXTRACTED_JSON

from paper_novelty_pipeline.utils.phase3_io import (
    read_final_index,
    load_topk_core,
    load_contrib_file,
    write_json,
    read_paper_info,
    build_paper_label_en,
)
from paper_novelty_pipeline.services.llm_client import create_llm_client
from paper_novelty_pipeline.config import (
    TAXONOMY_LLM_API_KEY,
    TAXONOMY_LLM_MODEL_NAME,
    TAXONOMY_LLM_API_ENDPOINT,
    PHASE3_MAX_LEAF_CAPACITY,
    PHASE3_MAX_TOKENS_TAXONOMY,
    PHASE3_MAX_TOKENS_NARRATIVE,
    PHASE3_MAX_TOKENS_APPENDIX,
    PHASE3_ONE_LINER_MIN_WORDS,
    PHASE3_ONE_LINER_MAX_WORDS,
    PHASE3_MAX_ABSTRACT_LENGTH,
)


# ============================================================================
# Constants
# ============================================================================

# Taxonomy configuration (configurable)
MAX_LEAF_CAPACITY = PHASE3_MAX_LEAF_CAPACITY
# Allow a larger completion budget for taxonomy to reduce truncation risk.
# EFFECTIVE_LLM_MAX_TOKENS (from config) currently defaults to 8000, so 5000 is safe.
DEFAULT_MAX_TOKENS_TAXONOMY = PHASE3_MAX_TOKENS_TAXONOMY
DEFAULT_MAX_TOKENS_NARRATIVE = PHASE3_MAX_TOKENS_NARRATIVE
DEFAULT_MAX_TOKENS_APPENDIX = PHASE3_MAX_TOKENS_APPENDIX  # Increased from 800 to accommodate 50 papers × ~30 words each

# UI hints defaults
DEFAULT_NODE_SPACING = 12
DEFAULT_RANK_SPACING = 28

# One-liner configuration
ONE_LINER_MIN_WORDS = PHASE3_ONE_LINER_MIN_WORDS
ONE_LINER_MAX_WORDS = PHASE3_ONE_LINER_MAX_WORDS

# Abstract truncation (shortened to keep prompts compact and leave room for JSON)
MAX_ABSTRACT_LENGTH = PHASE3_MAX_ABSTRACT_LENGTH


# ============================================================================
# Phase3Survey Class
# ============================================================================


class Phase3Survey:
    """
    Phase 3 Survey: Core task taxonomy and narrative generation.

    This class handles the survey workflow:
    1. Build initial report with placeholders
    2. Generate taxonomy via LLM
    3. Generate 2-paragraph narrative + original-paper position
    """

    def __init__(self) -> None:
        """Initialize Phase3Survey with logger and lazy-loaded LLM client."""
        self.logger = logging.getLogger(__name__)
        self._llm_client: Optional[Any] = None
        self._taxonomy_llm_client: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # LLM client
    # ------------------------------------------------------------------ #
    @property
    def llm_client(self) -> Optional[Any]:
        """Get LLM client with lazy initialization."""
        if self._llm_client is None:
            self._llm_client = create_llm_client()
        return self._llm_client

    @property
    def taxonomy_llm_client(self) -> Optional[Any]:
        """
        LLM client dedicated to taxonomy generation.

        If TAXONOMY_LLM_* overrides are provided in the environment/config,
        this will use a separate API key / model / endpoint just for the
        taxonomy call; otherwise it falls back to the main llm_client.
        """
        if self._taxonomy_llm_client is None:
            if any([TAXONOMY_LLM_API_KEY, TAXONOMY_LLM_MODEL_NAME, TAXONOMY_LLM_API_ENDPOINT]):
                self._taxonomy_llm_client = create_llm_client(
                    model_name=TAXONOMY_LLM_MODEL_NAME,
                    api_endpoint=TAXONOMY_LLM_API_ENDPOINT,
                    api_key=TAXONOMY_LLM_API_KEY,
                )
            else:
                self._taxonomy_llm_client = self.llm_client
        return self._taxonomy_llm_client

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _save_llm_response(
        self, raw_response: Any, file_prefix: str, out_dir: Path
    ) -> Optional[str]:
        """
        Save raw LLM response to phase3/raw_llm_responses/survey/.

        Args:
            raw_response: The raw response from the LLM (any JSON-serializable object)
            file_prefix: Prefix for the filename (e.g., "taxonomy", "narrative")
            out_dir: Base output directory

        Returns:
            Path to the saved file, or None if save failed
        """
        try:
            import hashlib
            import time
            
            # Create output directory
            raw_dir = out_dir / "phase3" / "raw_llm_responses" / "survey"
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp and hash
            timestamp = int(time.time())
            content_str = json.dumps(raw_response, ensure_ascii=False)
            content_hash = abs(hash(content_str)) % (10 ** 10)
            
            filename = f"{file_prefix}_{timestamp}_{content_hash}.json"
            filepath = raw_dir / filename
            
            # Save the raw response
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(raw_response, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved raw LLM response to {filepath}")
            return str(filepath)
        except Exception as e:
            self.logger.warning(f"Failed to save raw LLM response: {e}")
            return None
    
    def _infer_phase1_dir(self, out_dir: Path) -> Optional[Path]:
        """
        Infer Phase1 directory path from output directory.

        Args:
            out_dir: Base output directory

        Returns:
            Path to Phase1 directory if it exists, None otherwise
        """
        cand = Path(out_dir) / "phase1"
        return cand if cand.exists() else None

    def _select_top_contributions(
        self, contrib_files: List[str], max_contrib: int = 3
    ) -> List[str]:
        """
        Select top N contribution files.

        Current policy: keep first N; future: sort by avg relevance_score or length.

        Args:
            contrib_files: List of contribution file paths
            max_contrib: Maximum number of contributions to select

        Returns:
            List of selected contribution file paths
        """
        return list(contrib_files)[: max(0, int(max_contrib))]

    # ------------------------------------------------------------------ #
    # Step 1: Initial report
    # ------------------------------------------------------------------ #
    def build_initial_report(
        self,
        phase2_dir: Path,
        out_dir: Path,
        *,
        language: str = "en",
        max_contrib: int = 3,
    ) -> Dict[str, Any]:
        """
        Prepare phase3/core_task_survey/survey_report.json with placeholders.

        This is intentionally LLM-free and should be fast / robust.
        """
        phase2_dir = Path(phase2_dir)
        out_dir = Path(out_dir).resolve()  # Use absolute path to avoid creating dirs in wrong location
        phase3_dir = out_dir / "phase3"
        survey_dir = phase3_dir / "core_task_survey"
        survey_dir.mkdir(parents=True, exist_ok=True)
        (survey_dir / "visual").mkdir(parents=True, exist_ok=True)
        (survey_dir / "reports").mkdir(parents=True, exist_ok=True)
        (survey_dir / "appendix").mkdir(parents=True, exist_ok=True)

        idx = read_final_index(phase2_dir)
        core_path = idx.get("core_task_file")
        contrib_files: List[str] = idx.get("contribution_files", [])
        if not core_path:
            raise FileNotFoundError(
                "Phase2 final core_task file not found. Expected under phase2/final/."
            )

        core_items = load_topk_core(Path(core_path))

        # prepare appendix basic dump (no one-liner yet)
        basic_appendix = [
            {
                "canonical_id": it.get("canonical_id"),
                "rank": it.get("rank"),
                "title": it.get("title"),
                "venue": it.get("venue"),
                "year": it.get("year"),
                "doi": it.get("doi"),
                "url": it.get("url_pref"),
            }
            for it in core_items
        ]
        write_json(survey_dir / "appendix" / "top50_basic.json", basic_appendix)

        # Phase1 paper info (best-effort)
        phase1_dir = self._infer_phase1_dir(out_dir)
        paper_info = read_paper_info(phase1_dir) if phase1_dir else {}
        
        # Try to read Phase1 core_task description (for topic)
        core_task_text = None
        if phase1_dir:
            try:
                phase1_extracted = phase1_dir / PHASE1_EXTRACTED_JSON
                if phase1_extracted.exists():
                    extracted_data = json.loads(
                        phase1_extracted.read_text(encoding="utf-8")
                    )
                    core_task_text = (
                        extracted_data.get("core_task", {}).get("text", "").strip()
                    )
            except (KeyError, TypeError, AttributeError, json.JSONDecodeError):
                self.logger.debug("Failed to read core_task from Phase1.", exc_info=True)

        # choose up to N contributions
        selected_contrib_paths = self._select_top_contributions(
            contrib_files, max_contrib=max_contrib
        )
        contributions_novelty = []
        for p in selected_contrib_paths:
            contributions_novelty.append(
                {
                    "contribution_id": (
                        Path(p).stem.split("_")[1]
                        if "_" in Path(p).stem
                        else Path(p).stem
                    ),
                    "name": "",
                    "top10_file": str(p),
                    "narrative": {
                        "summary": "",
                        "relations_to_core_task": "",
                    },
                    "selected": True,
                }
            )

        # duplicates/unique stats (soft report, no mutation)
        key_counts: Dict[str, int] = {}
        for it in core_items:
            k = it.get("canonical_id")
            if not k:
                continue
            key_counts[k] = key_counts.get(k, 0) + 1
        duplicate_groups = [
            {
                "canonical_id": k,
                "count": c,
                "ranks": [
                    it["rank"] for it in core_items if it.get("canonical_id") == k
                ],
            }
            for k, c in key_counts.items()
            if c > 1
        ]

        topic = core_task_text or paper_info.get("title") or "Core Task"
        taxo_name = f"{topic} Survey Taxonomy"

        report: Dict[str, Any] = {
            "paper_info": paper_info,
            "short_survey": {
                "topic": topic,
                "language": language,
                "taxonomy": {"name": taxo_name, "subtopics": []},
                "mapping": [],
                "highlight": {
                    "original_paper_id": None,
                    "style": {"color": "#2b6cb0", "bold": True},
                },
                # figure_spec kept as a hint for front-end; no Mermaid content here.
                "figure_spec": {
                    "type": "taxonomy_json",
                    "theme": "default",
                    "content": "",
                    "legend": {
                        "original_label": "Original paper",
                        "color": "#2b6cb0",
                    },
                    "label_format": "title_short_author_year",
                    "label_max_len": 60,
                    "wrap_width": 26,
                    "list_layout": "comma_newline",
                    "numbering": True,
                },
                "narrative": {
                    "narrative": "",
                    "original_position": {
                        "taxonomy_path": [],
                        "neighbors": [],
                        "commentary": "",
                    },
                },
            },
            "contributions_novelty": contributions_novelty,
            "appendix": {
                "top50_briefs": [
                    {
                        "canonical_id": it.get("canonical_id"),
                        "rank": it.get("rank"),
                        "title": it.get("title"),
                        "venue": it.get("venue"),
                        "year": it.get("year"),
                        "brief_one_liner": "",
                        "leaf_path": [],
                    }
                    for it in core_items
                ]
            },
            "meta": {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "source_files": {
                    "core_task_file": str(core_path),
                    "contribution_files": selected_contrib_paths,
                },
                "version": "phase3-v1",
                "stats": {
                    "top50_count": len(core_items),
                    "unique_count": len(key_counts),
                    "duplicate_groups": duplicate_groups,
                    # tree_depth / leaf_count will be populated after taxonomy is built
                },
            },
        }

        # Fill original_paper_id if possible
        try:
            from paper_novelty_pipeline.utils.phase3_io import canonical_id_from_paper_info

            if paper_info:
                report["short_survey"]["highlight"][
                    "original_paper_id"
                ] = canonical_id_from_paper_info(paper_info)
        except (ImportError, AttributeError, TypeError) as e:
            self.logger.debug(f"Failed to set original_paper_id: {e}")

        # Save to new location: core_task_survey/survey_report.json
        write_json(survey_dir / "survey_report.json", report)
        self.logger.info(
            "Phase3: wrote initial survey_report.json with placeholders to %s",
            survey_dir,
        )
        return report

    # ------------------------------------------------------------------ #
    # Step 2: Taxonomy via LLM
    # ------------------------------------------------------------------ #
    def _expected_id_set(self, core_items: List[Dict[str, Any]]) -> List[str]:
        """Return stable list of unique paper_ids (preserving first occurrence)."""
        seen: List[str] = []
        s: set[str] = set()
        for it in core_items:
            pid = it.get("canonical_id")
            if pid and pid not in s:
                s.add(pid)
                seen.append(pid)
        return seen

    def _collect_leaf_assignments(
        self,
        node: Dict[str, Any],
        path: List[str],
        out: Dict[str, List[str]],
    ) -> None:
        """
        Recursively collect mapping from paper_id to taxonomy_path.

        This version assumes `path` does NOT yet include the current node's name.
        """
        name = node.get("name") or ""
        subs = node.get("subtopics")
        papers = node.get("papers")

        if isinstance(subs, list) and subs:
            new_path = path + [name] if name else list(path)
            for ch in subs:
                if isinstance(ch, dict):
                    self._collect_leaf_assignments(ch, new_path, out)
        elif isinstance(papers, list):
            leaf_path = path + [name] if name else list(path)
            for pid in papers:
                # last assignment wins, but we try to avoid duplicates earlier
                out[pid] = leaf_path

    def _collect_leaf_assignments_multi(
        self,
        node: Dict[str, Any],
        path: List[str],
        out: Dict[str, List[List[str]]],
    ) -> None:
        """
        Like _collect_leaf_assignments, but store ALL paths per paper_id to detect duplicates.
        """
        name = node.get("name") or ""
        subs = node.get("subtopics")
        papers = node.get("papers")

        if isinstance(subs, list) and subs:
            new_path = path + [name] if name else list(path)
            for ch in subs:
                if isinstance(ch, dict):
                    self._collect_leaf_assignments_multi(ch, new_path, out)
        elif isinstance(papers, list):
            leaf_path = path + [name] if name else list(path)
            for pid in papers:
                out.setdefault(pid, []).append(leaf_path)

    def _validate_taxonomy(
        self, taxo: Dict[str, Any], expected_ids: List[str]
    ) -> Tuple[bool, str]:
        """
        Validate structural constraints + coverage + uniqueness.

        Hard requirements:
          - Only leaves may have "papers".
          - Internal nodes must have "subtopics".
          - Every expected id appears in the tree。
          - NO id appears in more than one leaf (禁止重复 paper_id)。
        """
        try:
            # --- structural check: leaves vs internal nodes ---
            def check(node: Dict[str, Any]) -> Tuple[bool, str]:
                has_sub = isinstance(node.get("subtopics"), list) and node.get(
                    "subtopics"
                )
                has_papers = isinstance(node.get("papers"), list) and node.get("papers")
                if has_sub and has_papers:
                    return (
                        False,
                        f"A node ('{node.get('name', '')}') has both subtopics and papers",
                    )
                if has_sub:
                    for ch in node.get("subtopics"):
                        if isinstance(ch, dict):
                            ok, msg = check(ch)
                            if not ok:
                                return ok, msg
                return True, "ok"

            ok, msg = check(taxo)
            if not ok:
                return False, msg

            # --- coverage + uniqueness check ---
            assigned_multi: Dict[str, List[List[str]]] = {}
            self._collect_leaf_assignments_multi(taxo, [], assigned_multi)

            exp = set(expected_ids)
            got = set(assigned_multi.keys())

            missing = list(exp - got)
            extra = list(got - exp)
            if missing or extra:
                return (
                    False,
                    f"coverage_error missing={len(missing)} extra={len(extra)}",
                )

            # duplicates: any id with more than one path
            dup_ids = [pid for pid, paths in assigned_multi.items() if len(paths) > 1]
            if dup_ids:
                return False, f"duplicate_ids count={len(dup_ids)}"

            # everything looks fine
            return True, "ok"
        except (KeyError, TypeError, AttributeError) as e:
            return False, f"exception:{e}"

    def _list_assigned_ids(self, taxo: Dict[str, Any]) -> List[str]:
        """
        List all paper IDs assigned in the taxonomy (ignores duplicates).

        Args:
            taxo: Taxonomy dictionary

        Returns:
            List of assigned paper IDs
        """
        assigned: Dict[str, List[str]] = {}
        self._collect_leaf_assignments(taxo, [], assigned)
        return list(assigned.keys())

    def _remove_extra_ids(self, node: Dict[str, Any], allowed: set[str]) -> None:
        """
        Remove papers not in `allowed` from the tree; prune now-empty leaves.
        """
        subs = node.get("subtopics")
        if isinstance(subs, list) and subs:
            for ch in subs:
                if isinstance(ch, dict):
                    self._remove_extra_ids(ch, allowed)
            # Prune empty children (no subtopics and no papers)
            node["subtopics"] = [
                ch
                for ch in subs
                if isinstance(ch, dict)
                and (
                    (ch.get("subtopics") and len(ch.get("subtopics")) > 0)
                    or (ch.get("papers") and len(ch.get("papers")) > 0)
                )
            ]
            return

        papers = node.get("papers")
        if isinstance(papers, list):
            node["papers"] = [pid for pid in papers if pid in allowed]

    def _deduplicate_ids(self, node: Dict[str, Any], seen: set[str]) -> None:
        """
        Ensure each paper_id appears at most once in the tree (programmatic fallback).

        We traverse depth-first; the first leaf that contains a given id keeps it,
        later leaves drop that id. Empty leaves are pruned by callers if desired.
        """
        subs = node.get("subtopics")
        papers = node.get("papers")

        if isinstance(subs, list) and subs:
            for ch in subs:
                if isinstance(ch, dict):
                    self._deduplicate_ids(ch, seen)
            # Optional pruning of empty children (no subtopics and no papers)
            node["subtopics"] = [
                ch
                for ch in subs
                if isinstance(ch, dict)
                and (
                    (ch.get("subtopics") and len(ch.get("subtopics")) > 0)
                    or (ch.get("papers") and len(ch.get("papers")) > 0)
                )
            ]
            return

        if isinstance(papers, list):
            new_papers: List[str] = []
            for pid in papers:
                if pid in seen:
                    continue
                seen.add(pid)
                new_papers.append(pid)
            node["papers"] = new_papers

    def _ensure_leaf_capacity(
        self, node: Dict[str, Any], max_leaf: int = MAX_LEAF_CAPACITY
    ) -> None:
        """
        Split oversized leaf nodes into chunks to maintain capacity limit.

        Args:
            node: Taxonomy node to process
            max_leaf: Maximum number of papers per leaf (default: MAX_LEAF_CAPACITY)
        """
        papers = node.get("papers")
        if isinstance(papers, list) and len(papers) > max_leaf:
            name = node.get("name") or "Cluster"
            chunks = [papers[i : i + max_leaf] for i in range(0, len(papers), max_leaf)]
            node.pop("papers", None)
            node["subtopics"] = [
                {"name": f"{name}-{i + 1}", "papers": chunk}
                for i, chunk in enumerate(chunks)
            ]

    def _append_missing_bucket(self, taxo: Dict[str, Any], missing_ids: List[str]) -> None:
        """
        Append missing paper IDs to an "Unassigned" bucket in the taxonomy.

        Args:
            taxo: Taxonomy dictionary to modify
            missing_ids: List of paper IDs that were not assigned to any leaf
        """
        if not missing_ids:
            return
        bucket = {"name": "Unassigned", "papers": list(missing_ids)}
        self._ensure_leaf_capacity(bucket, max_leaf=MAX_LEAF_CAPACITY)
        subs = taxo.get("subtopics")
        if not isinstance(subs, list):
            subs = []
        subs.append(bucket)
        taxo["subtopics"] = subs

    # -------------------- Helpers: render text-only taxonomy --------------------
    def _render_text_tree(
        self,
        taxonomy: Dict[str, Any],
        *,
        original_id: Optional[str] = None,
        display_index: Optional[Dict[str, int]] = None,
        id_meta: Optional[Dict[str, Any]] = None,
        max_reprs: Optional[int] = 0,
    ) -> str:
        """Return a Markdown text-only tree (bulleted list) for the taxonomy.

        - Shows Parent → Child → Leaf as nested bullets.
        - For each leaf: appends star if it contains original_id and shows paper count.
        - Optionally lists up to `max_reprs` representative papers, ordered by display_index.
        """
        lines: List[str] = []

        def _surname_from(author: Any) -> str:
            try:
                s = str(author or "").strip()
                if not s:
                    return "Anon"
                # 'Surname, Given' first
                if "," in s:
                    return s.split(",", 1)[0].strip()
                parts = [p for p in s.split() if p]
                return parts[-1] if parts else "Anon"
            except (AttributeError, TypeError, IndexError):
                return "Anon"

        def label_for(pid: str) -> str:
            if not id_meta:
                return pid
            pm = id_meta.get(pid) or {}
            title = (pm.get("title") or "").strip()
            authors = pm.get("authors") or []
            year = pm.get("year") or ""
            doi = pm.get("doi") or ""
            url = pm.get("url") or pm.get("url_pref") or ""
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = "n.d."
            # author string
            if isinstance(authors, list) and authors:
                surname = _surname_from(authors[0])
                auth_str = f"{surname} et al."
                if len(authors) == 1:
                    auth_str = surname
            elif isinstance(authors, str) and authors.strip():
                surname = _surname_from(authors)
                auth_str = surname
            else:
                auth_str = "Anon et al."
            base = f"{title}" if title else pid
            tail = f"({auth_str}, {year})" if year else f"({auth_str})"
            link = ""
            if isinstance(url, str) and url.strip():
                link = f" [{url}]"
            elif isinstance(doi, str) and doi.strip():
                link = f" [https://doi.org/{doi}]"
            return f"{base} {tail}{link}".strip()

        def walk(node: Dict[str, Any], depth: int) -> None:
            name = node.get("name") or "Unnamed"
            subs = node.get("subtopics")
            papers = node.get("papers")
            indent = "  " * depth
            if isinstance(subs, list) and subs:
                lines.append(f"{indent}- {name}")
                for ch in subs:
                    if isinstance(ch, dict):
                        walk(ch, depth + 1)
                return
            if isinstance(papers, list) and papers:
                star = " ★" if (original_id and original_id in papers) else ""
                lines.append(f"{indent}- {name}{star} ({len(papers)} papers)")
                if max_reprs is None or (
                    isinstance(max_reprs, int) and max_reprs != 0
                ):
                    order = list(papers)
                    if display_index:
                        order = sorted(order, key=lambda x: display_index.get(x, 10**9))

                    # local dedup: arXiv DOI -> arxiv id; then doi; then normalized title+year
                    def alias_key(pid: str) -> str:
                        pm = (id_meta or {}).get(pid, {}) if id_meta else {}
                        url = str(pm.get("url") or "")
                        doi = str(pm.get("doi") or "")
                        title = str(pm.get("title") or "")
                        year = str(pm.get("year") or "")
                        import re

                        m = re.search(
                            r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})",
                            url,
                            flags=re.IGNORECASE,
                        )
                        if m:
                            return f"arxiv:{m.group(1).lower()}"
                        m2 = re.match(
                            r"^10\.48550/arxiv\.(\d{4}\.\d{4,5})$",
                            doi.strip().lower(),
                        )
                        if m2:
                            return f"arxiv:{m2.group(1).lower()}"
                        if doi.strip():
                            return f"doi:{doi.strip().lower()}"
                        # title+year fallback
                        import re as _re

                        t = _re.sub(
                            r"[^a-z0-9 ]+",
                            " ",
                            (title or "").lower(),
                        )
                        t = _re.sub(r"\s+", " ", t).strip()
                        y = year if str(year).strip() else "n.d."
                        return f"title:{t}|{y}"

                    seen_keys = set()
                    deduped: List[str] = []
                    for pid in order:
                        k = alias_key(pid)
                        if k in seen_keys:
                            continue
                        seen_keys.add(k)
                        deduped.append(pid)
                    if isinstance(max_reprs, int) and max_reprs > 0:
                        deduped = deduped[: max_reprs]
                    for pid in deduped:
                        idx = (
                            display_index.get(pid)
                            if isinstance(display_index, dict)
                            else None
                        )
                        if idx is None and original_id and pid == original_id:
                            prefix = "[0] "
                        else:
                            prefix = f"[{idx}] " if isinstance(idx, int) else "- "
                        lines.append(f"{indent}  - {prefix}{label_for(pid)}")
        
        walk(taxonomy, 0)
        return "\n".join(lines)

    def build_taxonomy_via_llm(
        self,
        phase2_dir: Path,
        out_dir: Path,
        *,
        language: str = "en",
        max_tokens: int = DEFAULT_MAX_TOKENS_TAXONOMY,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate taxonomy JSON via LLM, validate, and update survey_report.json.

        No Mermaid text is generated here; only taxonomy + mapping + display_index.
        """
        phase2_dir = Path(phase2_dir)
        out_dir = Path(out_dir).resolve()  # Use absolute path to avoid creating dirs in wrong location
        phase3_dir = out_dir / "phase3"
        survey_dir = phase3_dir / "core_task_survey"

        report_path = survey_dir / "survey_report.json"
        if not report_path.exists():
            # backward-compat: old path
            report_path = phase3_dir / "json_report.json"
        if not report_path.exists():
            self.logger.error(
                "survey_report.json or json_report.json not found; run Step 1 first."
            )
            return None

        # Load TopK items
        idx = read_final_index(phase2_dir)
        core_path = idx.get("core_task_file")
        if not core_path:
            self.logger.error("Phase2 final core_task file missing.")
            return None
        core_items = load_topk_core(Path(core_path))
        expected_ids = self._expected_id_set(core_items)

        # Prepare prompt inputs (title + short abstract)
        papers_payload = [
            {
                "id": it.get("canonical_id"),
                "title": it.get("title"),
                "abstract": it.get("abstract_short")
                or (it.get("abstract") or "")[:MAX_ABSTRACT_LENGTH],
                "rank": it.get("rank"),
            }
            for it in core_items
            if it.get("canonical_id")
        ]

        # Read report to obtain topic and original paper id for ordering
        try:
            report_obj = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            report_obj = {}
        
        # Try again reading core_task from Phase1 (for safety)
        core_task_text = None
        try:
            phase1_dir = out_dir / "phase1"
            phase1_extracted = phase1_dir / PHASE1_EXTRACTED_JSON
            if phase1_extracted.exists():
                extracted_data = json.loads(
                    phase1_extracted.read_text(encoding="utf-8")
                )
                core_task_text = (
                    extracted_data.get("core_task", {}).get("text", "").strip()
                )
        except (json.JSONDecodeError, FileNotFoundError, IOError, AttributeError):
            core_task_text = None
        
        topic_name = (
            core_task_text
            or report_obj.get("short_survey", {}).get("topic")
            or report_obj.get("paper_info", {}).get("title")
            or "Core Task"
        )
        original_paper_id = (
            report_obj.get("short_survey", {})
            .get("highlight", {})
            .get("original_paper_id")
        )

        # Ensure original paper participates in taxonomy construction
        if original_paper_id and original_paper_id not in expected_ids:
            expected_ids.append(original_paper_id)
            pinfo = report_obj.get("paper_info", {}) or {}
            # try to read abstract from Phase1 paper.json
            abs_text = ""
            try:
                p1 = out_dir / "phase1" / PAPER_JSON
                if p1.exists():
                    _pd = json.loads(p1.read_text(encoding="utf-8"))
                    abs_text = (_pd.get("abstract") or "").strip()
            except (json.JSONDecodeError, FileNotFoundError, IOError):
                abs_text = ""
            papers_payload.append(
                {
                "id": original_paper_id,
                "title": pinfo.get("title") or "",
                "abstract": abs_text,
                "rank": 0,
                }
            )

        # MECE + ordering-aware prompt (STRICT JSON)
        sys_prompt = (
            "You are a senior survey researcher specializing in building rigorous academic taxonomies.\n"
            "Return EXACTLY ONE JSON object and NOTHING ELSE (no markdown, no code fences, no explanations).\n\n"
            "INPUT (user JSON)\n"
            "- topic: a short core-task phrase (typically one line). Use it as the base topic label.\n"
            "- original_paper_id: optional (may be null).\n"
            "- papers: list of {id, title, abstract, rank}. IMPORTANT: ids are canonical identifiers; copy them verbatim.\n\n"
            "HARD CONSTRAINTS (must satisfy all)\n"
            "1) Output must be valid JSON parseable by json.loads: use DOUBLE QUOTES for all keys and all string values.\n"
            "2) Output must follow the schema exactly. Do not add any extra keys.\n"
            "3) Use ONLY the provided paper ids. Do NOT invent ids. Do NOT drop ids. Do NOT duplicate ids.\n"
            "   Every unique input id must appear EXACTLY ONCE across ALL leaves.\n"
            "4) Tree structure:\n"
            "   - Root MUST have: name, subtopics.\n"
            "   - Root MUST NOT contain: scope_note, exclude_note, papers.\n"
            "   - Non-leaf nodes MUST have: name, scope_note, exclude_note, subtopics (non-empty).\n"
            "   - Leaf nodes MUST have: name, scope_note, exclude_note, papers (non-empty array of ids).\n"
            "   - Non-leaf nodes MUST NOT contain 'papers'. Leaf nodes MUST NOT contain 'subtopics'.\n"
            "5) Root naming:\n"
            "   - Set TOPIC_LABEL by taking user.topic and, only if needed, compressing it to <= 8 words WITHOUT changing meaning.\n"
            "   - Allowed compression operations: delete redundant modifiers/articles/auxiliary prepositional phrases.\n"
            "     Do NOT replace core technical nouns/verbs with synonyms, and do NOT introduce new terms.\n"
            "   - Root name MUST be exactly: \"TOPIC_LABEL Survey Taxonomy\".\n"
            "   - Do NOT output the literal string \"<topic>\" and do NOT include angle brackets.\n"
            "6) If original_paper_id is provided, it must appear in exactly one leaf.\n\n"
            "ACADEMIC BOTTOM-UP PROCEDURE (survey style)\n"
            "A) Read titles+abstracts and extract key attributes per paper (use domain-appropriate equivalents):\n"
            "   core approach / intervention type / theoretical framework; research question / objective / outcome or endpoint;\n"
            "   evidence basis & study design (experimental/empirical/theoretical; validation protocol/setting);\n"
            "   data/materials/subjects and context (modality if applicable; application domain only if it separates papers meaningfully).\n"
            "B) Create micro-clusters of highly similar papers.\n"
            "C) Name each micro-cluster with precise technical terminology.\n"
            "D) Iteratively abstract upward into parents, maintaining clear boundaries.\n\n"
            "DEFAULT ORGANIZATION PRINCIPLE (domain-agnostic; override if corpus suggests otherwise)\n"
            "Choose the top-level split by the axis that maximizes discriminability and interpretability for this corpus:\n"
            "clear boundaries, reasonably balanced coverage, and minimal cross-membership.\n"
            "Consider axes in this order of preference ONLY IF they yield strong boundaries:\n"
            "1) Core approach / intervention type / theoretical framework (what is being proposed, changed, or assumed)\n"
            "2) Research question / objective / outcome (what is being solved, measured, or optimized)\n"
            "3) Study context and evidence basis (data/materials/subjects; experimental/empirical design; evaluation/validation protocol; setting)\n"
            "If multiple axes tie, prefer the one that yields more stable, reusable categories across papers.\n\n"
            "MECE REQUIREMENT AT EVERY SIBLING GROUP\n"
            "- Mutually exclusive scopes; collectively exhaustive coverage under the parent.\n"
            "- Each node must include:\n"
            "  - scope_note: exactly ONE sentence (<= 25 words) stating a clear inclusion rule.\n"
            "  - exclude_note: exactly ONE sentence (<= 25 words) stating a clear exclusion rule AND where excluded items belong.\n\n"
            "NAMING RULES (important)\n"
            "- Use concrete technical terms. Avoid vague buckets.\n"
            "- Forbidden words in category names: other, others, misc, miscellaneous, general, uncategorized, unclear.\n"
            "- Keep names concise (<= 5–7 words typically) but prioritize clarity.\n\n"
            "STRUCTURE BALANCE (soft constraints; never override HARD constraints)\n"
            "- Prefer depth 3–5.\n"
            "- Typical leaf size 2–7. If a leaf would exceed ~7 papers, split it using the next most informative axis.\n"
            "- Leaf size 1 is allowed ONLY if the paper is semantically distinct and merging would blur boundaries; otherwise merge into the closest sibling.\n\n"
            "NEAR-DUPLICATES / VERSIONS\n"
            "- Different ids may be versions of the same work. KEEP all ids separate (no merging/deletion).\n"
            "- You MAY place suspected versions under the same leaf if it is the best-fit category.\n\n"
            "ORDERING WITHIN EACH LEAF'S 'papers'\n"
            "- If original_paper_id is in that leaf, put it first.\n"
            "- Then order remaining ids by ascending input rank (ties: preserve the original input order).\n\n"
            "FINAL SELF-CHECK (mandatory before returning)\n"
            "- The union of all leaf 'papers' arrays equals the set of unique input ids (exact set equality).\n"
            "- No id appears twice; no unknown id appears; no empty subtopics; no empty papers.\n"
            "- Root name matches exactly \"TOPIC_LABEL Survey Taxonomy\".\n\n"
            "OUTPUT SCHEMA (STRICT; valid JSON; no extra keys)\n"
            "{\n"
            "  \"name\": \"TOPIC_LABEL Survey Taxonomy\",\n"
            "  \"subtopics\": [\n"
            "    {\n"
            "      \"name\": \"Parent\",\n"
            "      \"scope_note\": \"...\",\n"
            "      \"exclude_note\": \"...\",\n"
            "      \"subtopics\": [\n"
            "        {\n"
            "          \"name\": \"Child\",\n"
            "          \"scope_note\": \"...\",\n"
            "          \"exclude_note\": \"...\",\n"
            "          \"subtopics\": [\n"
            "            {\n"
            "              \"name\": \"Leaf\",\n"
            "              \"scope_note\": \"...\",\n"
            "              \"exclude_note\": \"...\",\n"
            "              \"papers\": [\"<paper_id>\", \"...\"]\n"
            "            }\n"
            "          ]\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

        user_prompt = {
            "topic": topic_name,
            "original_paper_id": original_paper_id,
            "papers": papers_payload,
        }

        client = self.taxonomy_llm_client
        if client is None:
            self.logger.error("LLM client is not configured.")
            return None
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ]

        # Primary attempt: provider-side JSON mode.
        # Enable prompt caching for the long system prompt (saves ~90% on repeated calls)
        taxo = client.generate_json(messages, max_tokens=max_tokens, use_cache=True)
        
        # Save raw taxonomy LLM response
        self._save_llm_response(taxo, "taxonomy", out_dir)

        # Fallback: if JSON mode failed (e.g., code fences, unterminated strings),
        # retry in plain-text mode and apply our own coercion logic.
        if not isinstance(taxo, dict):
            self.logger.warning(
                "LLM generate_json for taxonomy failed; retrying via text mode + manual JSON coercion."
            )
            try:
                raw = client.generate(
                    messages,
                    max_tokens=max_tokens,  # type: ignore[arg-type]
                    use_cache=True  # Enable prompt caching
                )
            except TypeError:
                # Some clients may not accept temperature as kwarg; fall back.
                raw = client.generate(messages, max_tokens=max_tokens, use_cache=True)  # type: ignore[call-arg]

            taxo_coerced = None
            if isinstance(raw, str) and raw.strip():
                parse_fn = getattr(client, "_parse_json_content", None)
                if callable(parse_fn):
                    try:
                        taxo_coerced = parse_fn(raw)
                    except Exception as e:
                        self.logger.error(f"Manual JSON coercion for taxonomy failed: {e}")
                else:
                    try:
                        taxo_coerced = json.loads(raw)
                    except Exception as e:
                        self.logger.error(f"json.loads on taxonomy text failed: {e}")
            taxo = taxo_coerced

        if not isinstance(taxo, dict):
            self.logger.error("LLM did not return taxonomy JSON")
            return None

        # Track repair actions for diagnostics (always written to survey_report.json)
        repair_steps: List[str] = []
        repair_attempts: int = 0
        repair_success: bool = False

        initial_ok, initial_msg = self._validate_taxonomy(taxo, expected_ids)
        ok, msg = initial_ok, initial_msg
        if not ok:
            # Repair strategy:
            # 1) Deterministic pre-fix (remove extras + deduplicate) because these do not require semantics.
            # 2) Re-validate; if still invalid, do ONE LLM repair round focused on placing missing papers
            #    with semantic context (title/abstract/rank), while keeping structure stable.
            # 3) STRICT: If still invalid, do NOT apply any mechanical "missing bucket" fallback.
            #    We keep the best-effort taxonomy, mark it as needs_review, and continue the pipeline.
            allowed = set(expected_ids)
            # Pre-fix: remove extras + deduplicate (best-effort)
            try:
                repair_steps.extend(["prefix_remove_extra", "prefix_dedup"])
                self._remove_extra_ids(taxo, allowed)
                self._deduplicate_ids(taxo, seen=set())
            except Exception as e:
                self.logger.debug(f"Pre-fix (remove extras/dedup) failed: {e}")

            ok_pre, msg_pre = self._validate_taxonomy(taxo, expected_ids)
            if ok_pre:
                # Pre-fix was sufficient; skip LLM repair.
                repair_success = True
                self.logger.info(
                    "Taxonomy validation failed initially (%s) but was fixed by deterministic pre-fix.",
                    msg,
                )
            else:
                assigned_ids = set(self._list_assigned_ids(taxo))
                missing = list(allowed - assigned_ids)
                extra = list(assigned_ids - allowed)
                
                # Log detailed pre-repair state
                self.logger.warning(
                    "Taxonomy validation failed after pre-fix: %s; missing=%d extra=%d – trying semantic repair",
                    msg_pre,
                    len(missing),
                    len(extra),
                )
                if missing:
                    self.logger.info(f"  Missing IDs (not assigned): {missing[:5]}{'...' if len(missing) > 5 else ''}")
                if extra:
                    self.logger.info(f"  Extra IDs (not allowed): {extra[:5]}{'...' if len(extra) > 5 else ''}")
                
                try:
                    # Provide semantic context for missing papers to place them correctly.
                    missing_set = set(missing)
                    missing_papers = [p for p in papers_payload if p.get("id") in missing_set]
                    root_name = f"{topic_name} Survey Taxonomy"

                    repair_steps.append("llm_semantic_repair")
                    repair_attempts += 1
                    
                    self.logger.info("🔧 Starting semantic repair (attempt %d)...", repair_attempts)
                    
                    repair_sys = (
                        "You will receive a taxonomy JSON and constraints for fixing it.\n"
                        "Return EXACTLY ONE valid JSON object parseable by json.loads (DOUBLE QUOTES; no code fences; no extra text).\n"
                        "Hard constraints:\n"
                        "- Single root. Root name MUST exactly equal root_name.\n"
                        "- Only leaf nodes may have 'papers' (non-empty array of ids). Non-leaf nodes must NOT have 'papers'.\n"
                        "- Internal nodes must have non-empty 'subtopics'. Leaf nodes must NOT have 'subtopics'.\n"
                        "- Use ONLY allowed_ids. Remove any extra_ids. Every allowed id must appear EXACTLY ONCE across all leaves.\n"
                        "- If original_paper_id is present in the taxonomy input, keep it assigned to exactly one leaf.\n\n"
                        "Repair style (important):\n"
                        "- MINIMAL-CHANGE: keep the existing node names and hierarchy whenever possible.\n"
                        "- First ensure extra_ids are removed and duplicates are eliminated.\n"
                        "- Then place missing_papers into the best-fit existing leaves using their titles/abstracts.\n"
                        "- Only if no existing leaf fits, create a small new leaf or a minimal new branch.\n"
                    )
                    repair_user = {
                        "root_name": root_name,
                        "allowed_ids": expected_ids,
                        "missing_ids": missing,
                        "extra_ids": extra,
                        "missing_papers": missing_papers,
                        "taxonomy": taxo,
                    }
                    messages2 = [
                        {"role": "system", "content": repair_sys},
                        {"role": "user", "content": json.dumps(repair_user, ensure_ascii=False)},
                    ]
                    repaired = client.generate_json(
                        messages2, max_tokens=max_tokens
                    )
                    
                    # Save raw repair LLM response
                    self._save_llm_response(repaired, "taxonomy_repair", out_dir)
                    
                    if isinstance(repaired, dict):
                        taxo = repaired
                        ok2, msg2 = self._validate_taxonomy(taxo, expected_ids)
                        
                        # Calculate post-repair state for detailed logging
                        assigned_ids_after = set(self._list_assigned_ids(taxo))
                        missing_after = list(allowed - assigned_ids_after)
                        extra_after = list(assigned_ids_after - allowed)
                        
                        if not ok2:
                            self.logger.warning(
                                "❌ Semantic repair validation failed: %s",
                                msg2,
                            )
                            self.logger.warning(
                                "   Post-repair state: missing=%d extra=%d (STRICT mode, no missing-bucket fallback)",
                                len(missing_after),
                                len(extra_after),
                            )
                            if missing_after:
                                self.logger.debug(f"   Still missing: {missing_after[:5]}{'...' if len(missing_after) > 5 else ''}")
                            if extra_after:
                                self.logger.debug(f"   Still extra: {extra_after[:5]}{'...' if len(extra_after) > 5 else ''}")
                            
                            # Best-effort deterministic cleanup (still strict)
                            try:
                                self._remove_extra_ids(taxo, allowed)
                                self._deduplicate_ids(taxo, seen=set())
                            except Exception:
                                pass
                        else:
                            # Success - log detailed results
                            fixed_missing = len(missing) - len(missing_after)
                            fixed_extra = len(extra) - len(extra_after)
                            
                            self.logger.info(
                                "✅ Semantic repair succeeded: %s",
                                msg2,
                            )
                            self.logger.info(
                                "   Fixed: %d missing, %d extra → Final state: missing=%d extra=%d",
                                fixed_missing,
                                fixed_extra,
                                len(missing_after),
                                len(extra_after),
                            )
                            
                            ok, msg = True, msg2
                            repair_success = True
                    else:
                        self.logger.warning(
                            "❌ Repair LLM did not return JSON (STRICT mode, no missing-bucket fallback)"
                        )
                except (TypeError, ValueError, KeyError) as e:
                    self.logger.warning(
                        "❌ Repair round exception (STRICT mode, no missing-bucket fallback): %s", e
                    )

        # Final validation for diagnostics (does NOT block pipeline)
        final_ok, final_msg = self._validate_taxonomy(taxo, expected_ids)

        # Build labels per paper id for text tree / front-end helpers
        id_to_label: Dict[str, str] = {}
        for it in core_items:
            pid = it.get("canonical_id")
            if not pid:
                continue
            id_to_label[pid] = build_paper_label_en(it, numbering=False)
        # Include original paper label if missing
        if original_paper_id and original_paper_id not in id_to_label:
            pinfo = report_obj.get("paper_info", {}) or {}
            pseudo = {
                "title": pinfo.get("title"),
                "authors": pinfo.get("authors") or [],
                "year": pinfo.get("year"),
                "venue": pinfo.get("venue"),
            }
            id_to_label[original_paper_id] = build_paper_label_en(pseudo, numbering=False)

        # Load report to access highlight settings
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            report = {"short_survey": {}}

        # Update report with taxonomy
        report.setdefault("short_survey", {})
        report["short_survey"]["taxonomy"] = taxo

        # -------------------- Taxonomy diagnostics (always written) --------------------
        try:
            allowed_set = set(expected_ids)
            assigned_list = self._list_assigned_ids(taxo)
            assigned_set = set(assigned_list)
            missing_ids = sorted(list(allowed_set - assigned_set))
            extra_ids = sorted(list(assigned_set - allowed_set))

            dup_count = 0
            dup_examples: List[Dict[str, Any]] = []
            try:
                multi: Dict[str, List[List[str]]] = {}
                self._collect_leaf_assignments_multi(taxo, [], multi)
                dup_ids = [pid for pid, paths in (multi or {}).items() if isinstance(paths, list) and len(paths) > 1]
                dup_count = len(dup_ids)
                for pid in dup_ids[:5]:
                    dup_examples.append(
                        {"id": pid, "paths": (multi.get(pid) or [])[:3], "reason": "duplicate_across_leaves"}
                    )
            except Exception:
                dup_count = 0

            issues: List[Dict[str, Any]] = []
            if missing_ids:
                issues.append(
                    {
                        "type": "missing",
                        "count": len(missing_ids),
                        "examples": [{"id": x, "paths": [], "reason": "missing_id"} for x in missing_ids[:5]],
                    }
                )
            if extra_ids:
                issues.append(
                    {
                        "type": "extra",
                        "count": len(extra_ids),
                        "examples": [{"id": x, "paths": [], "reason": "extra_id"} for x in extra_ids[:5]],
                    }
                )
            if dup_count:
                issues.append(
                    {
                        "type": "duplicate",
                        "count": dup_count,
                        "examples": dup_examples,
                    }
                )
            if not final_ok:
                issues.append(
                    {
                        "type": "structure",
                        "count": 1,
                        "examples": [{"id": None, "paths": [], "reason": (final_msg or "validation_failed")}],
                    }
                )

            if final_ok:
                status = "ok" if initial_ok else "repaired"
            else:
                status = "needs_review"

            summary = (
                f"status={status}; hard_constraints_ok={final_ok}; "
                f"missing={len(missing_ids)} extra={len(extra_ids)} duplicate={dup_count}"
            )

            report["short_survey"]["taxonomy_diagnostics"] = {
                "status": status,
                "hard_constraints_ok": bool(final_ok),
                "summary": summary,
                "expected_ids_count": len(expected_ids),
                "assigned_ids_count": len(assigned_set),
                "issues": issues,
                "repair": {
                    "steps": repair_steps,
                    "attempts": int(repair_attempts),
                    "success": bool(repair_success and final_ok),
                },
            }
        except Exception as e:
            self.logger.warning(f"Failed to write taxonomy_diagnostics: {e}")

        # Compute simple taxonomy statistics: tree_depth and leaf_count
        def _taxonomy_stats(node: Dict[str, Any], depth: int = 1) -> Tuple[int, int]:
            """
            Compute (tree_depth, leaf_count) for a taxonomy tree.

            - tree_depth: max root-to-leaf level count (root = 1).
            - leaf_count: number of leaves with non-empty 'papers'.
            """
            subtopics = node.get("subtopics") or []
            papers = node.get("papers") or []

            # Leaf node: no subtopics, but has papers
            if not subtopics:
                is_leaf = isinstance(papers, list) and len(papers) > 0
                return depth, 1 if is_leaf else 0

            max_child_depth = depth
            total_leaves = 0
            for ch in subtopics:
                if not isinstance(ch, dict):
                    continue
                d, c = _taxonomy_stats(ch, depth + 1)
                if d > max_child_depth:
                    max_child_depth = d
                total_leaves += c
            return max_child_depth, total_leaves

        try:
            tree_depth, leaf_count = _taxonomy_stats(taxo)
        except Exception as e:
            self.logger.warning(f"Failed to compute taxonomy stats: {e}")
            tree_depth, leaf_count = None, None

        stats = report.setdefault("short_survey", {}).setdefault("statistics", {})
        if tree_depth is not None:
            stats["tree_depth"] = tree_depth
        if leaf_count is not None:
            stats["leaf_count"] = leaf_count

        # mapping: canonical_id -> taxonomy_path
        mapping: Dict[str, List[str]] = {}
        self._collect_leaf_assignments(taxo, [], mapping)
        report["short_survey"]["mapping"] = [
            {"canonical_id": pid, "taxonomy_path": path} for pid, path in mapping.items()
        ]

        # display_index: original paper = 0, others ordered by TopK rank starting from 1
        id_to_rank: Dict[str, int] = {}
        for it in core_items:
            pid = it.get("canonical_id")
            rk = it.get("rank")
            if pid and rk is not None:
                id_to_rank[pid] = rk
        ordered_ids = [pid for pid, _ in sorted(id_to_rank.items(), key=lambda kv: kv[1])]
        display_index: Dict[str, int] = {}
        if original_paper_id:
            display_index[original_paper_id] = 0
        idx_counter = 1
        for pid in ordered_ids:
            if pid == original_paper_id:
                continue
            if pid not in display_index:
                display_index[pid] = idx_counter
                idx_counter += 1
        report["short_survey"]["display_index"] = display_index

        # ui_hints for front-end rendering defaults
        report["short_survey"]["ui_hints"] = {
            "layout": "flowchart",
            "curve": "step",
            "nodeSpacing": DEFAULT_NODE_SPACING,
            "rankSpacing": DEFAULT_RANK_SPACING,
            "label": {"mode": "per_line"},  # or 'comma_nowrap'
        }

        # Write updated report + taxonomy-only file
        survey_dir.mkdir(parents=True, exist_ok=True)
        write_json(survey_dir / "survey_report.json", report)
        taxonomy_data = report.get("short_survey", {}).get("taxonomy", {})
        write_json(survey_dir / "taxonomy.json", taxonomy_data)
        self.logger.info("Phase3: taxonomy + mapping written.")
        return taxo

    # ------------------------------------------------------------------ #
    # Step 3: Narrative (2 paragraphs) + Original Position
    # ------------------------------------------------------------------ #
    def _load_report(self, out_dir: Path) -> Tuple[Dict[str, Any], Path]:
        """Load survey_report.json (new path preferred, old path as fallback)."""
        out_dir = Path(out_dir).resolve()  # Use absolute path to avoid creating dirs in wrong location
        phase3_dir = out_dir / "phase3"
        survey_dir = phase3_dir / "core_task_survey"
        report_path = survey_dir / "survey_report.json"
        if not report_path.exists():
            report_path = phase3_dir / "json_report.json"
        if not report_path.exists():
            raise FileNotFoundError(
                "phase3/core_task_survey/survey_report.json or phase3/json_report.json not found. Run Step 1 first."
            )
        with report_path.open("r", encoding="utf-8") as f:
            return json.load(f), phase3_dir

    def _build_leaf_groups(self, taxonomy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build list of leaf groups from taxonomy tree.

        Each group contains the leaf path and associated paper IDs.
        """
        groups: List[Dict[str, Any]] = []

        def walk(node: Dict[str, Any], path: List[str]) -> None:
            subs = node.get("subtopics")
            papers = node.get("papers")
            name = node.get("name") or ""
            if isinstance(subs, list) and subs:
                new_path = path + [name] if name else list(path)
                for ch in subs:
                    if isinstance(ch, dict):
                        walk(ch, new_path)
                return
            if isinstance(papers, list) and papers:
                leaf_path = path + [name] if name else list(path)
                groups.append({"leaf_path": leaf_path, "papers": papers})

        walk(taxonomy, [])
        return groups

    def _collect_taxonomy_paper_ids(self, taxonomy: Dict[str, Any]) -> Set[str]:
        """
        Collect all paper IDs appearing anywhere in a taxonomy tree.

        Used to scope narrative citations strictly to the papers that actually
        appear in the core-task taxonomy tree.
        """
        ids: Set[str] = set()

        def walk(node: Dict[str, Any]) -> None:
            subs = node.get("subtopics")
            papers = node.get("papers")
            if isinstance(papers, list):
                for pid in papers:
                    if isinstance(pid, str) and pid.strip():
                        ids.add(pid.strip())
            if isinstance(subs, list):
                for ch in subs:
                    if isinstance(ch, dict):
                        walk(ch)

        walk(taxonomy)
        return ids

    def _filter_citation_index_for_taxonomy(
        self,
        citation_index: Dict[str, Dict[str, Any]],
        taxonomy_paper_ids: Set[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Filter a citation_index to only include IDs that appear in the taxonomy."""
        return {pid: info for pid, info in citation_index.items() if pid in taxonomy_paper_ids}

    def _extract_numeric_citation_indices(self, text: str) -> List[int]:
        """
        Extract purely numeric indices from bracketed citation markers like '[12]'.

        This intentionally ignores non-numeric bracket content (e.g., duplicate-id
        hints like '[arxiv:...,openreview:...]').
        """
        if not text:
            return []
        return [int(m.group(1)) for m in re.finditer(r"\[(\d{1,6})\]", text)]

    def _validate_narrative_citations(
        self,
        narrative_text: str,
        allowed_indices: Set[int],
    ) -> Tuple[bool, List[int]]:
        """
        Ensure narrative only cites indices that are present in allowed_indices.

        Returns (ok, invalid_indices_sorted_unique).
        """
        cited = self._extract_numeric_citation_indices(narrative_text)
        invalid = sorted({i for i in cited if i not in allowed_indices})
        return (len(invalid) == 0), invalid

    def _build_citation_index(
        self,
        core_items: List[Dict[str, Any]],
        original_paper_id: Optional[str],
        display_index: Dict[str, int],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a compact citation index mapping paper_id -> alias/index/year/is_original.

        alias will be a short name derived from the title (e.g., 'AgentGym-RL').
        """
        def make_alias(title: str) -> str:
            if not title:
                return "Paper"
            t = title.strip()
            # Prefer part before ':' if present
            if ":" in t:
                head = t.split(":", 1)[0].strip()
                if head:
                    return head
            # Otherwise use first up-to-4 words
            words = t.split()
            return " ".join(words[:4])

        citation_index: Dict[str, Dict[str, Any]] = {}
        for it in core_items:
            pid = it.get("canonical_id")
            if not pid:
                continue
            title = it.get("title") or ""
            year = it.get("year")
            idx = display_index.get(pid)
            citation_index[pid] = {
                "alias": make_alias(title),
                "index": idx,
                "year": year,
                "is_original": (pid == original_paper_id),
            }
        return citation_index

    def _load_citation_index_from_phase2(
        self,
        phase2_dir: Path,
        original_paper_id: Optional[str],
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Load citation_index from Phase2 final/citation_index.json.
        
        This ensures consistency with Phase2's alias generation across the pipeline.
        Returns None if the file doesn't exist or can't be loaded.
        
        Returns:
            Dict mapping paper_id -> {alias, index, year, is_original}, or None on failure
        """
        citation_path = phase2_dir / "final" / "citation_index.json"
        
        if not citation_path.exists():
            self.logger.warning(
                f"Phase2 citation_index.json not found at {citation_path}, "
                "will generate aliases from titles as fallback"
            )
            return None
        
        try:
            data = json.loads(citation_path.read_text(encoding="utf-8"))
            # Phase2 citation_index.json schema: prefer "items" (new), fallback to "papers" (legacy)
            papers = data.get("items", None)
            if papers is None:
                papers = data.get("papers", [])
            
            if not isinstance(papers, list):
                self.logger.warning("Phase2 citation_index.json has unexpected structure")
                return None
            
            citation_index: Dict[str, Dict[str, Any]] = {}
            for paper in papers:
                pid = paper.get("canonical_id")
                if not pid:
                    continue
                
                # Use Phase2's alias and index for consistency
                citation_index[pid] = {
                    "alias": paper.get("alias", "Paper"),
                    "index": paper.get("index"),
                    "year": paper.get("year"),
                    "is_original": (pid == original_paper_id),
                }
            
            self.logger.info(
                f"Loaded {len(citation_index)} paper aliases from Phase2 citation_index.json"
            )
            return citation_index
            
        except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
            self.logger.warning(
                f"Failed to load Phase2 citation_index.json: {e}, "
                "will generate aliases from titles as fallback"
            )
            return None

    def build_narrative_via_llm(
        self,
        phase2_dir: Path,
        out_dir: Path,
        *,
        language: str = "en",
        max_tokens: int = DEFAULT_MAX_TOKENS_NARRATIVE,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a compact 2-paragraph narrative + original-paper position.

        Narrative requirements:
          - EXACTLY two paragraphs (separated by a blank line).
          - No bullet lists or headings.
          - No explicit integer counts of papers/branches (e.g., '14 papers', '3 frameworks').
            Use qualitative phrases such as 'a small handful', 'many', 'a dense branch',
            'only a few works', etc.
          - When referring to specific papers, use Alias[index] form based on the
            provided citation_index (e.g., 'AgentGym-RL[0]').
          - Optionally, you MAY end the SECOND paragraph with a single sentence of the form:
              'Version note: possible duplicates include [idA,idB]; [idC,idD].'
            if you strongly suspect near-duplicate ids (arXiv/OpenReview/published variants).
            In that sentence, keep the format exactly and do not mention numeric counts.
        """
        report, phase3_dir = self._load_report(out_dir)
        idx = read_final_index(phase2_dir)
        core_path = idx.get("core_task_file")
        if not core_path:
            self.logger.error("Phase2 final core_task file missing.")
            return None
        core_items = load_topk_core(Path(core_path))

        # Paper metadata for tree labels
        id_meta: Dict[str, Any] = {
            it.get("canonical_id"): {
                "title": it.get("title"),
                "authors": it.get("authors")
                or it.get("author_names")
                or (it.get("raw_metadata") or {}).get("authors", []),
                "abstract": it.get("abstract_short")
                or (it.get("abstract") or "")[:MAX_ABSTRACT_LENGTH],
                "year": it.get("year"),
                "venue": it.get("venue"),
                "doi": it.get("doi"),
                "url": it.get("url_pref") or it.get("url"),
                "url_pref": it.get("url_pref"),
            }
            for it in core_items
            if it.get("canonical_id")
        }

        original_id = (
            report.get("short_survey", {})
            .get("highlight", {})
            .get("original_paper_id")
        )
        # include original paper meta if missing
        if original_id and original_id not in id_meta:
            pinfo = report.get("paper_info", {}) or {}
            id_meta[original_id] = {
                "title": pinfo.get("title"),
                "authors": pinfo.get("authors") or [],
                "abstract": "",
                "year": pinfo.get("year"),
                "venue": pinfo.get("venue"),
                "doi": pinfo.get("doi"),
                "url": pinfo.get("url"),
                "url_pref": None,
            }

        taxo = report.get("short_survey", {}).get("taxonomy")
        if not isinstance(taxo, dict):
            self.logger.error("taxonomy not found in report. Run Step 2 first.")
            return None

        leaf_groups = self._build_leaf_groups(taxo)

        # mapping & display_index
        mapping_list = report.get("short_survey", {}).get("mapping", []) or []
        display_index: Dict[str, int] = (
            report.get("short_survey", {}).get("display_index", {}) or {}
        )

        # Taxonomy path for original paper
        original_path: List[str] = []
        for m in mapping_list:
            if m.get("canonical_id") == original_id:
                original_path = list(m.get("taxonomy_path") or [])
                break

        # Neighbors: other papers in the same leaf, ordered by display_index
        neighbors_ids: List[str] = []
        if original_id:
            for g in leaf_groups:
                if original_id in g.get("papers", []):
                    # Others in the same leaf
                    leaf_others = [
                        pid for pid in g.get("papers", []) if pid != original_id
                    ]
                    neighbors_ids = sorted(
                        leaf_others, key=lambda x: display_index.get(x, 10**9)
                    )
                    break

        # Core task text (from Phase2 candidates if available)
        core_task_text: Optional[str] = None
        try:
            core_candidates_file = Path(phase2_dir) / "candidates" / "core_task_candidates.json"
            if not core_candidates_file.exists():
                alt = Path(phase2_dir) / "core_task_candidates.json"
                core_candidates_file = alt if alt.exists() else core_candidates_file
            if core_candidates_file.exists():
                with core_candidates_file.open("r", encoding="utf-8") as fct:
                    _cdata = json.load(fct)
                    core_task_text = (_cdata.get("core_task_text") or "").strip() or None
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            core_task_text = None

        # Citation index for Alias[index] usage in LLM narrative
        # Prefer Phase2's citation_index.json for consistency across pipeline
        citation_index = self._load_citation_index_from_phase2(
            phase2_dir, original_id
        )
        
        # Fallback: generate aliases from titles if Phase2 index unavailable
        if citation_index is None:
            self.logger.info("Generating citation aliases from paper titles (fallback)")
            citation_index = self._build_citation_index(
                core_items, original_id, display_index
            )

        # IMPORTANT: scope citations to the taxonomy papers only.
        taxonomy_paper_ids = self._collect_taxonomy_paper_ids(taxo)
        citation_index = self._filter_citation_index_for_taxonomy(citation_index, taxonomy_paper_ids)
        allowed_indices: Set[int] = {
            int(info.get("index"))
            for info in citation_index.values()
            if isinstance(info, dict) and isinstance(info.get("index"), int)
        }

        # Top-level branches (for global structure hints)
        top_level_branches = [
            st.get("name") or ""
            for st in (taxo.get("subtopics") or [])
            if isinstance(st, dict)
        ]

        sys_prompt = (
            "You are writing a SHORT survey-style narrative for domain experts.\n"
            "You will receive:\n"
            "- A core task description (if available).\n"
            "- A taxonomy root name and a list of top-level branches.\n"
            "- The taxonomy path and leaf neighbors of the original paper.\n"
            "- A citation_index mapping each paper_id to {alias,index,year,is_original}.\n\n"
            "**CRITICAL: JSON OUTPUT FORMAT**\n"
            "The text you write will contain citation markers like [0], [3], or [57] (e.g., 'AgentGym-RL[0]'). "
            "These are NOT JSON arrays - they are just text references to papers. "
            "You MUST output the COMPLETE JSON object as specified below, NOT just a number in brackets. "
            "Your entire response must be a valid JSON object starting with '{' and ending with '}'.\n\n"
            "Your job is to return STRICT JSON only (no code fences) with a single key:\n"
            "{\n"
            "  'narrative': '<TWO short paragraphs, plain text>'\n"
            "}\n\n"
            "NARRATIVE REQUIREMENTS:\n"
            "- Write EXACTLY TWO paragraphs, separated by a single blank line.\n"
            "- No headings, no bullet points; use compact, fluent prose.\n"
            "- Overall length should be relatively short (roughly 180–250 words).\n"
            "- In the FIRST paragraph:\n"
            "  * Briefly restate the core task in your own words (you may start with\n"
            "    'Core task: <core_task_text>' when a core_task_text is provided).\n"
            "  * Give a high-level picture of the field structure suggested by the\n"
            "    taxonomy: what the main branches are, what kinds of methods or\n"
            "    problem settings each branch tends to focus on, and how they relate.\n"
            "  * You may mention a few representative works using the Alias[index]\n"
            "    style when it helps to make the structure concrete.\n"
            "- In the SECOND paragraph:\n"
            "  * Zoom in on a few particularly active or contrasting lines of work,\n"
            "    and describe the main themes, trade-offs, or open questions that\n"
            "    appear across these branches.\n"
            "  * Naturally situate the original paper (typically Alias[0]) within this\n"
            "    landscape: describe which branch or small cluster of works it feels\n"
            "    closest to, and how its emphasis compares to one or two nearby\n"
            "    papers (for example Alias[3], Alias[5]), without rewriting a full\n"
            "    taxonomy path.\n"
            "  * Keep the tone descriptive: you are helping the reader see where this\n"
            "    work roughly sits among existing directions, not writing a review\n"
            "    decision.\n\n"
            "NUMERIC STYLE:\n"
            "- Avoid detailed integer counts of papers or branches; instead prefer\n"
            "  qualitative phrases such as 'a small handful of works', 'many studies',\n"
            "  'a dense branch', 'only a few papers'.\n"
            "- You may use numeric indices that are part of citations like Alias[0]\n"
            "  or Alias[3]; this is allowed.\n\n"
            "CITATION STYLE:\n"
            "- When you want to mention a specific paper, use its Alias[index] from\n"
            "  citation_index. For example: 'AgentGym-RL[0] provides ...',\n"
            "  'Webagent-r1[1] focuses on ...'.\n"
            "- Do not invent new aliases; only use the provided ones.\n"
            "- You may ONLY cite indices that are listed in allowed_citation_indices.\n"
            "  Do not cite any other index.\n"
        )

        user_payload = {
            "language": language,
            "core_task_text": core_task_text,
            "taxonomy_root": taxo.get("name"),
            "top_level_branches": top_level_branches,
            "original_paper_id": original_id,
            "original_taxonomy_path": original_path,
            "neighbor_ids": neighbors_ids,
            "citation_index": citation_index,
            "allowed_citation_indices": sorted(allowed_indices),
        }

        client = self.llm_client
        if client is None:
            self.logger.error("LLM client not configured")
            return None
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        # Enable prompt caching for the system prompt (saves ~90% on repeated calls)
        nar = client.generate_json(messages, max_tokens=max_tokens, use_cache=True)
        
        # Save raw narrative LLM response
        self._save_llm_response(nar, "narrative", out_dir)
        
        if not isinstance(nar, dict):
            self.logger.error("LLM did not return narrative JSON")
            return None

        # Basic sanity normalization
        narrative_text = (nar.get("narrative") or "").strip()
        if not narrative_text:
            self.logger.error("Narrative JSON missing non-empty 'narrative' field")
            return None

        # Post-validate: narrative must not cite indices outside the taxonomy.
        ok, invalid = self._validate_narrative_citations(narrative_text, allowed_indices)
        if not ok:
            self.logger.warning(
                "Narrative cites out-of-taxonomy indices %s; retrying once with stricter instruction.",
                invalid,
            )
            sys_prompt_retry = (
                sys_prompt
                + "\n\nSTRICT CITATION CONSTRAINT (HARD RULE):\n"
                + f"- allowed_citation_indices = {sorted(allowed_indices)}\n"
                + "- You MUST NOT cite any index outside allowed_citation_indices.\n"
            )
            messages_retry = [
                {"role": "system", "content": sys_prompt_retry},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ]
            nar2 = client.generate_json(messages_retry, max_tokens=max_tokens, use_cache=True)
            
            # Save raw narrative retry LLM response
            self._save_llm_response(nar2, "narrative_retry", out_dir)
            
            if not isinstance(nar2, dict):
                self.logger.error("Narrative retry did not return narrative JSON")
                return None
            narrative_text2 = (nar2.get("narrative") or "").strip()
            if not narrative_text2:
                self.logger.error("Narrative retry missing non-empty 'narrative' field")
                return None
            ok2, invalid2 = self._validate_narrative_citations(narrative_text2, allowed_indices)
            if not ok2:
                self.logger.error(
                    "Narrative retry still cites out-of-taxonomy indices %s; failing narrative step.",
                    invalid2,
                )
                return None
            nar = nar2
            narrative_text = narrative_text2

        # Merge into report（仅保留 narrative 文本，original_position 不再暴露）
        report.setdefault("short_survey", {})["narrative"] = {
            "narrative": narrative_text,
        }

        # Save report + narrative.json
        survey_dir = phase3_dir / "core_task_survey"
        survey_dir.mkdir(parents=True, exist_ok=True)
        write_json(survey_dir / "survey_report.json", report)
        narrative_data = report.get("short_survey", {}).get("narrative", {})
        write_json(survey_dir / "narrative.json", narrative_data)
        self.logger.info(
            "Phase3: narrative JSON written to %s",
            survey_dir / "narrative.json",
        )

        # Generate markdown file: tree + narrative + original position
        md_path = survey_dir / "reports" / "core_task_survey.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)

        topic = report.get("short_survey", {}).get("topic")
        core_task_text_for_md = core_task_text

        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Core Task Short Survey\n\n")
            if core_task_text_for_md:
                f.write(f"Core task: {core_task_text_for_md}\n\n")
            elif topic:
                f.write(f"Core task: {topic}\n\n")

            # Taxonomy text tree
            try:
                text_tree = self._render_text_tree(
                    taxo,
                    original_id=original_id,
                    display_index=display_index,
                    id_meta=id_meta,
                    max_reprs=None,
                )
                if text_tree.strip():
                    f.write("## Taxonomy (text)\n\n")
                    f.write(text_tree + "\n\n")
            except (KeyError, TypeError, AttributeError) as e:
                self.logger.debug("Failed to render text tree: %s", e)

            # Narrative
            f.write("## Narrative\n\n")
            f.write(narrative_text + "\n\n")
        self.logger.info("Phase3: narrative markdown written to %s", md_path)
        return nar

    # ------------------------------------------------------------------ #
    # Step 4: Appendix one-liners (unchanged)
    # ------------------------------------------------------------------ #
    def build_appendix_via_llm(
        self,
        phase2_dir: Path,
        out_dir: Path,
        *,
        language: str = "en",
        max_tokens: int = DEFAULT_MAX_TOKENS_APPENDIX,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate one-liner summaries for TopK papers and merge into appendix.top50_briefs.
        """
        report, phase3_dir = self._load_report(out_dir)
        idx = read_final_index(phase2_dir)
        core_path = idx.get("core_task_file")
        if not core_path:
            self.logger.error("Phase2 final core_task file missing.")
            return None
        core_items = load_topk_core(Path(core_path))
        payload = [
            {
                "canonical_id": it.get("canonical_id"),
                "title": it.get("title"),
                "abstract": it.get("abstract_short")
                or (it.get("abstract") or "")[:MAX_ABSTRACT_LENGTH],
            }
            for it in core_items
            if it.get("canonical_id")
        ]
        sys_prompt = (
            f"Write a concise one-liner summary for each paper ({ONE_LINER_MIN_WORDS}–{ONE_LINER_MAX_WORDS} words).\n"
            "Return STRICT JSON only: {\"items\": [{\"paper_id\": id, \"brief_one_liner\": text}, ...]} with the SAME\n"
            "order and length as the input list. Do not invent numbers; base only on title/abstract. Language: English.\n\n"
            "JSON FORMAT RULES (CRITICAL):\n"
            "- The entire response must be valid JSON that can be parsed by a standard json.loads implementation.\n"
            "- Do NOT wrap the JSON in code fences such as ```json or ```; return raw JSON only.\n"
            "- Inside JSON string values, do not use unescaped double quotes. If you need quotes inside a string,\n"
            "  either use single quotes or escape double quotes as \\\".\n"
            "- Do not include comments, trailing commas, or any keys beyond 'items', 'paper_id', 'brief_one_liner'.\n"
        )
        client = self.llm_client
        if client is None:
            self.logger.error("LLM client not configured")
            return None
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps({"papers": payload}, ensure_ascii=False)},
        ]
        # Enable prompt caching (may not hit cache if system prompt < 1024 tokens, but no harm)
        js = client.generate_json(messages, max_tokens=max_tokens, use_cache=True)
        
        # Save raw one-liners LLM response
        self._save_llm_response(js, "one_liners", out_dir)

        # Fallback: retry in text mode and coerce to JSON if provider JSON mode fails.
        if not isinstance(js, dict) or not isinstance(js.get("items"), list):
            self.logger.warning(
                "LLM generate_json for one-liners failed; retrying via text mode + manual JSON coercion."
            )
            try:
                raw = client.generate(
                    messages,
                    max_tokens=max_tokens,  # type: ignore[arg-type]
                    use_cache=True  # Enable prompt caching
                )
            except TypeError:
                raw = client.generate(messages, max_tokens=max_tokens, use_cache=True)  # type: ignore[call-arg]

            js_coerced = None
            if isinstance(raw, str) and raw.strip():
                parse_fn = getattr(client, "_parse_json_content", None)
                if callable(parse_fn):
                    try:
                        js_coerced = parse_fn(raw)
                    except Exception as e:
                        self.logger.error(f"Manual JSON coercion for one-liners failed: {e}")
                else:
                    try:
                        js_coerced = json.loads(raw)
                    except Exception as e:
                        self.logger.error(f"json.loads on one-liners text failed: {e}")
            js = js_coerced

        if not isinstance(js, dict) or not isinstance(js.get("items"), list):
            self.logger.error("LLM did not return one-liners JSON")
            return None
        liners = js["items"]
        # Merge into report.appendix.top50_briefs by paper_id
        brief_map = {
            it.get("canonical_id"): (it.get("brief_one_liner") or "").strip()
            for it in liners
            if it.get("canonical_id")
        }
        for ent in report.get("appendix", {}).get("top50_briefs", []):
            pid = ent.get("canonical_id")
            if pid and pid in brief_map:
                ent["brief_one_liner"] = brief_map[pid]
        # Save to new location: core_task_survey/survey_report.json
        survey_dir = phase3_dir / "core_task_survey"
        survey_dir.mkdir(parents=True, exist_ok=True)
        write_json(survey_dir / "survey_report.json", report)
        self.logger.info("Phase3: one-liners merged into survey_report.json")
        return liners
