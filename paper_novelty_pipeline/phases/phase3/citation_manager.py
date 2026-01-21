"""
Phase 3: Citation Manager

Unified citation management for all Phase 3 components.
Handles citation index loading, extension, and reference generation.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set


# ============================================================================
# Citation Manager Class
# ============================================================================

class CitationManager:
    """
    Unified citation manager for Phase 3.
    
    Responsibilities:
    - Load citation_index from core_task_survey
    - Extend citation_index with Contribution Analysis candidates
    - Provide paper_id -> Alias[index] conversion
    - Generate unified reference lists
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        base_dir: Optional[Path] = None,
    ):
        """
        Initialize CitationManager.
        
        Args:
            logger: Logger instance
            base_dir: Base directory containing phase3 outputs
        """
        self.logger = logger
        self.base_dir = base_dir or Path(".")
        
        # Citation index: key is canonical_id (or other stable paper id) ->
        # {alias, index, year, title, url, ...}
        self.citation_index: Dict[str, Dict[str, Any]] = {}
        
        # Track which papers have been cited
        self.cited_papers: Set[str] = set()
        
        # Next available index for new citations
        self.next_index: int = 0
    
    def load_from_phase2_citation(
        self,
        citation_index_path: Optional[Path] = None,
    ) -> bool:
        """
        Load citation_index from Phase2 `final/citation_index.json`.

        This is the preferred, canonical source of citation information.
        Keys are canonical_id strings.
        """
        if citation_index_path is None:
            citation_index_path = (
                self.base_dir / "phase2" / "final" / "citation_index.json"
            )

        if not citation_index_path.exists():
            self.logger.warning(f"Phase2 citation_index.json not found: {citation_index_path}")
            return False

        try:
            with open(citation_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Phase2 citation_index.json schema: prefer "items" (new), fallback to "papers" (legacy)
            papers = data.get("items", None)
            if papers is None:
                papers = data.get("papers", [])
            if not isinstance(papers, list):
                self.logger.warning("Phase2 citation_index.json has no 'items'/'papers' list")
                return False

            index: Dict[str, Dict[str, Any]] = {}
            max_index = -1
            for p in papers:
                if not isinstance(p, dict):
                    continue
                cid = p.get("canonical_id")
                if not isinstance(cid, str) or not cid.strip():
                    continue
                cid = cid.strip()
                idx = p.get("index")
                alias = p.get("alias") or ""
                title = p.get("title") or ""
                year = p.get("year")
                url = p.get("url") or p.get("source_url")
                authors = p.get("authors") or []
                venue = p.get("venue")
                doi = p.get("doi")
                arxiv_id = p.get("arxiv_id")

                # Determine whether this is the original paper (based on roles)
                is_original = False
                roles = p.get("roles") or []
                for r in roles:
                    if isinstance(r, dict) and r.get("type") == "original_paper":
                        is_original = True
                        break

                index[cid] = {
                    "alias": alias or self._make_alias(title),
                    "index": idx,
                    "year": year,
                    "title": title,
                    "url": url,
                    "authors": authors,
                    "venue": venue,
                    "doi": doi,
                    "arxiv_id": arxiv_id,
                    "is_original": is_original,
                }

                if isinstance(idx, int) and idx > max_index:
                    max_index = idx

            self.citation_index = index
            self.next_index = max_index + 1 if max_index >= 0 else 0

            self.logger.info(
                f"Loaded {len(self.citation_index)} citations from Phase2 "
                f"({citation_index_path}, max_index={max_index}, next_index={self.next_index})"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to load citation_index from Phase2: {e}")
            return False

    def load_from_survey_report(
        self,
        survey_report_path: Optional[Path] = None,
    ) -> bool:
        """
        Load citation_index for Phase3.

        Preferred source is Phase2 `final/citation_index.json`. If that fails,
        falls back to loading from `core_task_survey/survey_report.json`
        (legacy format used by early experiments).
        
        Args:
            survey_report_path: Optional path to survey_report.json.
                               If None, uses default location.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        # 1) Preferred: Phase2 citation_index.json (new pipeline)
        if self.load_from_phase2_citation():
            return True

        # 2) Fallback: legacy citation_index embedded in survey_report.json
        if survey_report_path is None:
            survey_report_path = (
                self.base_dir / "phase3" / "core_task_survey" / "survey_report.json"
            )

        if not survey_report_path.exists():
            self.logger.warning(f"Survey report not found: {survey_report_path}")
            return False

        try:
            with open(survey_report_path, "r", encoding="utf-8") as f:
                survey_report = json.load(f)

            citation_index = survey_report.get("short_survey", {}).get("citation_index", {})

            if not citation_index:
                self.logger.warning("No citation_index found in survey_report")
                return False

            # Load into self.citation_index (keys are whatever the legacy report used)
            self.citation_index = citation_index.copy()

            # Find maximum index to determine next_index
            max_index = -1
            for info in self.citation_index.values():
                idx = info.get("index")
                if isinstance(idx, int) and idx > max_index:
                    max_index = idx

            self.next_index = max_index + 1 if max_index >= 0 else 0

            self.logger.info(
                f"Loaded {len(self.citation_index)} citations from survey_report "
                f"(max_index={max_index}, next_index={self.next_index})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to load citation_index from survey_report: {e}")
            return False
    
    def _make_alias(self, title: str) -> str:
        """
        Generate a short alias from paper title.
        
        Strategy:
        - If title contains ':', use part before ':'
        - Otherwise, use first 4 words
        - Clean up special characters
        """
        if not title:
            return "Paper"
        
        t = title.strip()
        
        # Prefer part before ':' if present
        if ":" in t:
            head = t.split(":", 1)[0].strip()
            if head:
                t = head
        
        # Use first up-to-4 words
        words = t.split()
        alias = " ".join(words[:4])
        
        # Clean up: remove special chars, keep alphanumeric and spaces
        alias = re.sub(r'[^\w\s-]', '', alias)
        alias = re.sub(r'\s+', ' ', alias).strip()
        
        return alias or "Paper"
    
    def add_paper(
        self,
        paper_id: str,
        title: str,
        year: Optional[int] = None,
        url: Optional[str] = None,
        authors: Optional[List[str]] = None,
        venue: Optional[str] = None,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
    ) -> int:
        """
        Add a paper to citation_index if not already present.
        
        Args:
            paper_id: Paper ID
            title: Paper title
            year: Publication year
            url: Paper URL
            authors: List of authors
            venue: Publication venue
            doi: DOI
            arxiv_id: arXiv ID
        
        Returns:
            Citation index assigned to this paper
        """
        # Check if already exists
        if paper_id in self.citation_index:
            return self.citation_index[paper_id]["index"]
        
        # Generate alias
        alias = self._make_alias(title)
        
        # Assign new index
        index = self.next_index
        self.next_index += 1
        
        # Add to citation_index
        self.citation_index[paper_id] = {
            "alias": alias,
            "index": index,
            "year": year,
            "title": title,
            "url": url,
            "authors": authors or [],
            "venue": venue,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "is_original": False,  # Contribution Analysis papers are not original
        }
        
        self.logger.debug(f"Added citation: {alias}[{index}] for paper {paper_id}")
        return index
    
    def cite(self, paper_id: str) -> str:
        """
        Get citation string in format 'Alias[index]'.
        
        Args:
            paper_id: Paper ID
        
        Returns:
            Citation string like 'AgentGym-RL[3]' or 'Unknown[?]' if not found
        """
        info = self.citation_index.get(paper_id)
        if not info:
            self.logger.warning(f"Paper {paper_id} not found in citation_index")
            return f"Unknown[?]"
        
        alias = info.get("alias", "Paper")
        index = info.get("index")
        
        if index is None:
            return f"{alias}[?]"
        
        # Mark as cited
        self.cited_papers.add(paper_id)
        
        return f"{alias}[{index}]"
    
    def get_citation_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full citation information for a paper.
        
        Args:
            paper_id: Paper ID
        
        Returns:
            Citation info dict or None if not found
        """
        return self.citation_index.get(paper_id)
    
    def generate_references_markdown(
        self,
        cited_paper_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Generate markdown-formatted reference list.
        
        Args:
            cited_paper_ids: Optional list of paper IDs to include.
                           If None, includes all papers in citation_index.
        
        Returns:
            Markdown string with reference list
        """
        if cited_paper_ids is None:
            # Use all papers in citation_index, sorted by index
            cited_paper_ids = sorted(
                self.citation_index.keys(),
                key=lambda pid: self.citation_index[pid].get("index", 999999)
            )
        
        lines = ["## References\n"]
        
        for paper_id in cited_paper_ids:
            info = self.citation_index.get(paper_id)
            if not info:
                continue
            
            index = info.get("index")
            alias = info.get("alias", "Paper")
            year = info.get("year")
            title = info.get("title", "")
            url = info.get("url", "")
            authors = info.get("authors", [])
            
            # Format: [index] Alias (Year). Title. URL
            year_str = f"({year})" if year else ""
            authors_str = ", ".join(authors[:3])  # First 3 authors
            if len(authors) > 3:
                authors_str += " et al."
            authors_str = f"{authors_str}. " if authors_str else ""
            
            ref_line = f"[{index}] {authors_str}{title}"
            if year_str:
                ref_line += f" {year_str}"
            if url:
                ref_line += f" {url}"
            
            lines.append(ref_line)
        
        return "\n".join(lines)
    
    def generate_references_json(
        self,
        cited_paper_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate JSON-formatted reference list.
        
        Args:
            cited_paper_ids: Optional list of paper IDs to include.
                           If None, includes all papers in citation_index.
        
        Returns:
            List of reference dictionaries
        """
        if cited_paper_ids is None:
            cited_paper_ids = sorted(
                self.citation_index.keys(),
                key=lambda pid: self.citation_index[pid].get("index", 999999)
            )
        
        references = []
        for paper_id in cited_paper_ids:
            info = self.citation_index.get(paper_id)
            if not info:
                continue
            
            # Skip papers without valid canonical_id
            if not paper_id or not str(paper_id).strip():
                continue
            
            references.append({
                "canonical_id": paper_id,
                "index": info.get("index"),
                "alias": info.get("alias", "Paper"),
                "title": info.get("title", ""),
                "authors": info.get("authors", []),
                "year": info.get("year"),
                "venue": info.get("venue"),
                "url": info.get("url", ""),
                "doi": info.get("doi"),
                "arxiv_id": info.get("arxiv_id"),
            })
        
        return references
    
    def save_extended_index(self, output_path: Path) -> None:
        """
        Save extended citation_index to file.
        
        Args:
            output_path: Path to save citation_index JSON
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.citation_index, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved extended citation_index to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save citation_index: {e}")

