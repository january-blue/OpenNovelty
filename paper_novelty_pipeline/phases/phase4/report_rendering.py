"""
Phase 4: Lightweight PDF/Markdown Novelty Assessment Report Generator

A simplified report generator:
- Generates Markdown report and optionally converts to PDF
- Template-based: Most content generated from JSON data + templates
"""

import json
import logging
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from paper_novelty_pipeline.utils.report_artifacts import (
    phase4_lightweight_report_basename_from_phase3_report,
)

class LightweightReportGenerator:
    """
    Lightweight report generator for Phase 4.

    Generates a concise novelty assessment report from phase3_complete_report.json.
    Uses template-based generation only; overall novelty summary is taken from Phase3:
    contribution_analysis.overall_novelty_assignment.summary_paragraph
    """
    
    def __init__(self, generate_pdf: bool = True):
        """
        Initialize the lightweight report generator.
        
        Args:
            generate_pdf: If True, generate PDF in addition to Markdown
        """
        self.logger = logging.getLogger(__name__)
        self.generate_pdf = generate_pdf
    
    def generate_report(
        self,
        phase3_complete_report_path: Path,
        output_dir: Path
    ) -> Dict[str, str]:
        """
        Generate lightweight novelty assessment report.
        
        Args:
            phase3_complete_report_path: Path to phase3_complete_report.json
            output_dir: Output directory for report
            
        Returns:
            Dict with "markdown" and optionally "pdf" keys pointing to file paths
        """
        self.logger.info(f"Generating lightweight report from: {phase3_complete_report_path}")
        
        # 1. Load JSON data
        try:
            with open(phase3_complete_report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._current_report_path = phase3_complete_report_path
        except Exception as e:
            self.logger.error(f"Failed to load phase3_complete_report.json: {e}")
            return {}
        
        # 2. Build all sections (template-based, no LLM)
        sections = []
        sections.append(self._build_header(data))  # 1) Original Paper Info
        sections.append(self._build_core_task_section(data))  # 2) Core Task Survey (Tree+Narrative)
        sections.append(self._build_core_task_comparisons(data))  # 3) Core Task Comparisons (Siblings)
        sections.append(self._build_contributions_section(data))  # 4) Contribution Analysis（Include Phase3 overall_novelty_assignment）
        
        # 3. Add text similarity detection appendix (if any similarities found)
        similarity_appendix = self._build_similarity_appendix(data)
        if similarity_appendix:
            sections.append(similarity_appendix)
        
        # 3.5 Add References section based on Phase3 references (if available)
        references_section = self._build_references_section(data)
        if references_section:
            sections.append(references_section)
        
        # 4. Combine into complete Markdown
        markdown_content = "\n\n".join(sections)

        # 4.1 Add OpenNovelty link in the top-right corner as a subtle header line.
        # Use inline HTML so that PDF renderers treat it as a clickable link without default underline/border.
        header_link = (
            '<p style="text-align: right; font-size: 0.85em; color: #666666; '
            'margin: 0;">'
            '<a href="https://opennovelty.org/" '
            'style="color: #666666; text-decoration: none; border: none;">'
            'https://opennovelty.org/'
            '</a></p>\n\n'
        )
        markdown_content = header_link + markdown_content
        
        # 5. Save Markdown
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Deterministic filename: prefer Phase3-provided artifacts; otherwise compute from Phase3 metadata.generated_at
        meta = (data or {}).get("metadata", {}) or {}
        artifacts = meta.get("artifacts") or {}
        md_filename = artifacts.get("phase4_lightweight_md_filename")
        if not isinstance(md_filename, str) or not md_filename.strip():
            # Compute a deterministic basename from Phase3 report (fallback keeps old behavior if metadata missing)
            base = phase4_lightweight_report_basename_from_phase3_report(data)
            md_filename = f"{base}.md"
        md_path = output_dir / md_filename
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        self.logger.info(f"Markdown report saved to: {md_path}")
        
        result = {"markdown": str(md_path)}
        
        # 6. Generate PDF if requested
        if self.generate_pdf:
            # Keep PDF name in sync with Markdown basename.
            pdf_path = md_path.with_suffix(".pdf")
            if self._generate_pdf_from_markdown(md_path, pdf_path):
                result["pdf"] = str(pdf_path)
                self.logger.info(f"PDF report saved to: {pdf_path}")
            else:
                self.logger.warning("PDF generation failed, but Markdown report is available")
        
        return result
    
    # ==================== Section Builders (Template-Based) ====================
    
    def _build_header(self, data: Dict) -> str:
        """Build header section with paper information."""
        original = data.get("original_paper", {})
        metadata = data.get("metadata", {})
        
        lines = [
            "# Novelty Assessment Report",
            "",
            "---",
            "",
            f"**Paper**: {original.get('title', 'N/A')}",
            ""
        ]
        
        # PDF URL (try paper_id first, fallback to url if paper_id is missing)
        pdf_url = original.get("paper_id") or original.get("url")
        if isinstance(pdf_url, str) and pdf_url.strip():
            safe_url = pdf_url.strip()
            lines.append(
                f"**PDF URL**: <a href=\"{safe_url}\" "
                f"style=\"color:#1a73e8; text-decoration:none;\">{safe_url}</a>"
            )
            lines.append("")
        
        authors = original.get("authors")
        if authors:
            if isinstance(authors, list):
                lines.append(f"**Authors**: {', '.join(authors)}")
            else:
                lines.append(f"**Authors**: {authors}")
            lines.append("")
        
        if original.get("venue"):
            lines.append(f"**Venue**: {original.get('venue')}")
            lines.append("")
        
        if original.get("year"):
            lines.append(f"**Year**: {original.get('year')}")
            lines.append("")
        
        generated_at = metadata.get("generated_at", datetime.now().isoformat())
        lines.append(f"**Report Generated**: {generated_at[:10] if len(generated_at) >= 10 else generated_at}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Abstract
        abstract = original.get("abstract")
        if abstract:
            lines.append("## Abstract")
            lines.append("")
            lines.append(abstract)
            lines.append("")
            
            # Yellow disclaimer box (HTML block) directly under Abstract
            disclaimer_html = (
                '<div style="background-color:#FFF9E8; border:1px solid #F1DEAA; '
                'border-radius:8px; padding:8px 12px; margin:6px 0 10px 0; '
                'font-size:0.8em; line-height:1.45;">'
                '<p style="margin:0 0 4px 0; font-weight:600; color:#6D5A2A;">'
                'Disclaimer'
                '</p>'
                '<p style="margin:0 0 4px 0; color:#5A4A3A;">'
                'This report is <strong>AI-GENERATED</strong> using Large Language Models and '
                'WisPaper (a scholar search engine). It analyzes academic papers&#39; tasks and '
                'contributions against retrieved prior work. While this system identifies '
                '<strong>POTENTIAL</strong> overlaps and novel directions, '
                '<strong>ITS COVERAGE IS NOT EXHAUSTIVE AND JUDGMENTS ARE APPROXIMATE</strong>. '
                'These results are intended to assist human reviewers and '
                '<strong>SHOULD NOT</strong> be relied upon as a definitive verdict on novelty.'
                '</p>'
                '<p style="margin:0; color:#5A4A3A;">'
                'Note that some papers exist in multiple, slightly different versions (e.g., with '
                'different titles or URLs). The system may retrieve several versions of the same '
                'underlying work. The current automated pipeline does not reliably align or '
                'distinguish these cases, so human reviewers will need to disambiguate them '
                'manually.'
                '</p>'
                '<p style="margin:6px 0 0 0; color:#5A4A3A; font-style:italic;">'
                'If you have any questions, please contact: '
                '<a href="mailto:mingzhang23@m.fudan.edu.cn" '
                'style="color:inherit; text-decoration:none; font:inherit; font-style:inherit;">'
                'mingzhang23@m.fudan.edu.cn'
                '</a>'
                '</p>'
                '</div>'
            )
            lines.append(disclaimer_html)
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def _build_core_task_section(self, data: Dict) -> str:
        """Build Core Task Landscape section with complete taxonomy tree."""
        survey = data.get("core_task_survey", {})
        if not survey:
            return "## Core Task Landscape\n\n*No survey data available.*\n"
        
        taxonomy = survey.get("taxonomy", {})
        mapping = survey.get("mapping", [])
        papers = survey.get("papers", {})
        statistics = survey.get("statistics", {})
        display_index_map = survey.get("display_index", {})  #   获取 display_index mapping
        
        lines = [
            "## Core Task Landscape",
            ""
        ]
        
        # Core task description
        taxonomy_name = taxonomy.get("name", "")
        if taxonomy_name:
            # Extract core task from taxonomy name (remove "Survey Taxonomy" suffix)
            core_task = taxonomy_name.replace(" Survey Taxonomy", "")
            lines.append(f"This paper addresses: **{core_task}**")
            lines.append("")
        
        # Statistics
        top50_count = statistics.get("top50_count", len(papers))
        leaf_count = self._count_leaf_nodes(taxonomy)
        
        lines.append(f"A total of **{top50_count} papers** were analyzed and organized into a taxonomy with **{leaf_count} categories**.")
        lines.append("")
        

        # Brief taxonomy overview (top-level categories only)
        subtopics = taxonomy.get("subtopics", [])
        if subtopics:
            lines.append("### Taxonomy Overview")
            lines.append("")
            lines.append("The research landscape has been organized into the following main categories:")
            lines.append("")
            for subtopic in subtopics[:10]:  # Limit to top 10
                name = subtopic.get("name", "")
                if name:
                    lines.append(f"- **{name}**")
            if len(subtopics) > 10:
                lines.append(f"- *... and {len(subtopics) - 10} more categories*")
            lines.append("")
        
        lines.append("### Complete Taxonomy Tree")
        lines.append("")
        
        taxonomy_from_md = self._extract_taxonomy_from_survey_md(data)
        if taxonomy_from_md:
            lines.append(taxonomy_from_md)
        else:
            # Fallback: render full tree recursively
            root_name = taxonomy.get("name", "")
            if root_name:
                lines.append(f"**{root_name}**")
                lines.append("")
            
            subtopics = taxonomy.get("subtopics", [])
            rendered_urls = set()
            if subtopics:
                for subtopic_idx, subtopic in enumerate(subtopics):
                    is_last_subtopic = (subtopic_idx == len(subtopics) - 1)
                    self._render_taxonomy_tree(subtopic, papers, lines, display_index_map, rendered_urls, indent_level=1, prefix="", is_last=is_last_subtopic)
        
        lines.append("")
        
        #   Add Narrative (directly use overview from phase3_complete_report.json)
        narrative = survey.get("narrative", {})
        overview = narrative.get("overview", "")
        if overview:
            lines.append("### Narrative")
            lines.append("")
            lines.append(overview)
            lines.append("")
        
        return "\n".join(lines)
    
    def _build_contributions_section(self, data: Dict) -> str:
        """Build Contributions Analysis section with ALL detailed comparisons."""
        contrib_analysis = data.get("contribution_analysis", {})
        contribs_with_results = contrib_analysis.get("contributions_with_results", [])
        overall_novelty = (contrib_analysis.get("overall_novelty_assignment") or {}).get("summary_paragraph", "").strip()
        
        if not contribs_with_results:
            lines = ["## Contributions Analysis", ""]
            if overall_novelty:
                lines.append(overall_novelty)
                lines.append("")
            else:
                lines.append("*No contribution analysis available.*")
                lines.append("")
            return "\n".join(lines)
        
        lines = [
            "## Contributions Analysis",
            "",
        ]
        
        if overall_novelty:
            # Add an explicit lead-in so readers know this is the overall novelty summary.
            lines.append(f"**Overall novelty summary.** {overall_novelty}")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        lines.append(f"This paper presents **{len(contribs_with_results)} main contributions**, each analyzed against relevant prior work:")
        lines.append("")
        
        for idx, contrib in enumerate(contribs_with_results, 1):
            name = contrib.get("contribution_name", f"Contribution {idx}")
            desc = contrib.get("contribution_description", "")
            comparisons = contrib.get("comparisons", [])
            
            lines.append(f"### Contribution {idx}: {name}")
            lines.append("")
            
            if desc:
                lines.append(f"**Description**: {desc}")
                lines.append("")
            
            lines.append(f"This contribution was assessed against **{len(comparisons)} related papers** from the literature.")
            lines.append(f"Papers with potential prior art are analyzed in detail with textual evidence; others receive brief assessments.")
            lines.append("")
            lines.append("---")
            lines.append("")
            
            #   Display all comparisons in detail (supporting new refutation-based format)
            if comparisons:
                for comp_idx, comp in enumerate(comparisons, 1):
                    # Note: Contribution analysis uses simple field names (title, url)
                    title = comp.get("title", "Unknown Paper")
                    authors = comp.get("authors", [])
                    url = comp.get("url", "")
                    
                    # Check for new refutation-based format
                    refutation_status = comp.get("refutation_status", "")
                    refutation_evidence = comp.get("refutation_evidence", {})
                    brief_note = comp.get("brief_note", "")
                    
                    # Fallback to legacy format if refutation_status not present
                    novelty_summary = comp.get("novelty_summary", "")
                    novelty_evidence = comp.get("novelty_evidence", [])
                    
                    lines.append(f"#### {comp_idx}. {title}")
                    lines.append("")
                    
                    # Metadata
                    # Note: authors field may not exist in contribution analysis
                    if authors and len(authors) > 0:
                        author_str = ", ".join(authors[:5])
                        if len(authors) > 5:
                            author_str += f", et al. ({len(authors)} authors total)"
                        lines.append(f"**Authors**: {author_str}")
                        lines.append("")
                    
                    if url:
                        lines.append(f"**URL**: [{url}]({url})")
                        lines.append("")
                    
                    #   Display based on refutation_status (Refutation-based V2 logic)
                    if refutation_status == "can_refute":
                        # Can refute: show refutation evidence (prior art analysis)
                        lines.append("**Prior Art Analysis**")
                        lines.append("")
                        
                        # Use refutation_evidence if available
                        if isinstance(refutation_evidence, dict) and refutation_evidence.get("summary"):
                            summary = refutation_evidence.get("summary", "")
                            evidence_pairs = refutation_evidence.get("evidence_pairs", [])
                            
                            if summary:
                                lines.append(summary)
                                lines.append("")
                            
                            # Display evidence pairs
                            if evidence_pairs and isinstance(evidence_pairs, list) and len(evidence_pairs) > 0:
                                lines.append("**Evidence**")
                                lines.append("")
                                for ev_idx, evidence in enumerate(evidence_pairs[:5], 1):
                                    if isinstance(evidence, dict):
                                        orig_quote = evidence.get("original_quote", "")
                                        cand_quote = evidence.get("candidate_quote", "")
                                        rationale = evidence.get("rationale", "")
                                        
                                        if orig_quote or cand_quote:
                                            lines.append(f"*Evidence {ev_idx}*")
                                            if rationale:
                                                lines.append(f"- **Rationale**: {rationale}")
                                            if orig_quote:
                                                lines.append(f"- **Original**: {orig_quote[:300]}{'...' if len(orig_quote) > 300 else ''}")
                                            if cand_quote:
                                                lines.append(f"- **Candidate**: {cand_quote[:300]}{'...' if len(cand_quote) > 300 else ''}")
                                            lines.append("")
                                
                                if len(evidence_pairs) > 5:
                                    lines.append(f"*... and {len(evidence_pairs) - 5} more evidence pairs*")
                                    lines.append("")
                        else:
                            lines.append("*Prior art detected but evidence details are missing.*")
                            lines.append("")
                    
                    elif refutation_status in ("cannot_refute", "unclear"):
                        # Cannot refute or unclear: show brief note
                        lines.append("**Brief Assessment**")
                        lines.append("")
                        if brief_note:
                            lines.append(brief_note)
                        else:
                            lines.append("*No significant overlap with this contribution.*")
                        lines.append("")
                    
                    else:
                        # Safety fallback for older reports if any still exist
                        lines.append("*Analysis status unknown.*")
                        lines.append("")
                    
                    lines.append("---")
                    lines.append("")
        
        return "\n".join(lines)
    
    def _build_core_task_comparisons(self, data: Dict) -> str:
        """Build Related Works in Same Category section (sibling comparisons)."""
        core = data.get("core_task_comparisons", {}) or {}
        comparisons = core.get("comparisons", []) or []
        structural = core.get("structural_position", {}) or {}
        position_type = structural.get("position_type") or ("has_siblings" if comparisons else "")
        note_en = structural.get("note_en", "")

        # Load subtopic comparison if available
        subtopic_data = core.get("subtopic_comparison", {}) or {}
        if not subtopic_data:
            pointer = core.get("subtopic_comparison_file")
            if pointer and hasattr(self, "_current_report_path"):
                base_dir = Path(self._current_report_path).parent / "core_task_comparisons"
                spath = base_dir / pointer
                if spath.exists():
                    try:
                        subtopic_data = json.loads(spath.read_text(encoding="utf-8"))
                    except Exception as e:
                        self.logger.warning(f"Failed to load subtopic_comparison via pointer: {e}")

        lines = [
            "## Related Works in Same Category",
            "",
        ]

        # Case: no siblings but have sibling subtopics -> show taxonomy-level summary
        if position_type == "no_siblings_but_subtopic_siblings":
            if note_en:
                lines.append(note_en)
                lines.append("")
            if subtopic_data:
                llm = subtopic_data.get("llm_summary_en", {}) or {}
                if llm:
                    lines.append("### Taxonomy-Level Summary")
                    if llm.get("overall"):
                        lines.append(llm["overall"])
                    if llm.get("similarities"):
                        lines.append("")
                        lines.append("**Similarities:**")
                        for item in llm["similarities"]:
                            lines.append(f"- {item}")
                    if llm.get("differences"):
                        lines.append("")
                        lines.append("**Differences:**")
                        for item in llm["differences"]:
                            lines.append(f"- {item}")
                    if llm.get("suggested_search_directions"):
                        lines.append("")
                        lines.append("**Suggested Search Directions:**")
                        for item in llm["suggested_search_directions"]:
                            lines.append(f"- {item}")
                    lines.append("")

                sibs = subtopic_data.get("sibling_subtopics", []) or []
                if sibs:
                    lines.append("### Sibling Subtopics")
                    for sub in sibs:
                        name = sub.get("subtopic_name", "Unnamed Subtopic")
                        pcnt = sub.get("paper_count", 0)
                        lcnt = sub.get("leaf_count", 0)
                        lines.append(f"- **{name}** (leaves: {lcnt}, papers: {pcnt})")
                        scope = sub.get("scope_note")
                        exclude = sub.get("exclude_note")
                        if scope:
                            lines.append(f"  - Scope: {scope}")
                        if exclude:
                            lines.append(f"  - Exclude: {exclude}")
                    lines.append("")

            return "\n".join(lines)

        # Case: no siblings and no sibling subtopics -> structural isolation note
        if position_type == "no_siblings_no_subtopic_siblings":
            if note_en:
                lines.append(note_en)
            else:
                lines.append("*No sibling papers or subtopics were found in the taxonomy.*")
            return "\n".join(lines)

        # Default: has siblings (paper-level comparisons)
        if not comparisons:
            lines.append("*No comparison data available.*")
            return "\n".join(lines)

        lines.append(f"The following **{len(comparisons)} sibling papers** share the same taxonomy leaf node with the original paper:")
        lines.append("")

        for idx, comp in enumerate(comparisons, 1):
            title = comp.get("candidate_paper_title", "Unknown Paper")
            authors = comp.get("candidate_paper_authors", [])
            url = comp.get("candidate_paper_url", "")
            year = comp.get("candidate_paper_year")
            venue = comp.get("candidate_paper_venue", "")
            abstract = comp.get("candidate_paper_abstract", "")
            
            # New fields (simplified structure)
            is_duplicate = comp.get("is_duplicate_variant", False)
            brief_comparison = comp.get("brief_comparison", "")
            
            if idx > 1:
                lines.append("---")
                lines.append("")
            
            lines.append(f"### {idx}. {title}")
            lines.append("")
            
            meta_parts = []
            if authors:
                author_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    author_str += f", et al. ({len(authors)} authors total)"
                meta_parts.append(f"**Authors**: {author_str}")
            if year:
                year_venue = str(year)
                if venue:
                    year_venue += f" • {venue}"
                meta_parts.append(f"**Year/Venue**: {year_venue}")
            if url:
                meta_parts.append(f"**URL**: [{url}]({url})")
            
            if meta_parts:
                lines.append(" | ".join(meta_parts))
                lines.append("")
            
            if abstract:
                lines.append("#### Abstract")
                lines.append("")
                abstract_display = abstract[:500] + "..." if len(abstract) > 500 else abstract
                lines.append(abstract_display)
                lines.append("")
            
            #   New: Show duplicate warning or relationship analysis
            if is_duplicate:
                lines.append("#### ⚠️ Similarity Notice")
                lines.append("")
                lines.append(brief_comparison)
                lines.append("")
            elif brief_comparison:
                lines.append("#### Relationship Analysis")
                lines.append("")
                lines.append(brief_comparison)
                lines.append("")
        
        return "\n".join(lines)
    

    
    # ==================== Taxonomy Tree Rendering (ASCII Art) ====================
    
    def _render_taxonomy_tree(self, node: Dict, papers: Dict, lines: List[str], 
                               display_index_map: Dict = None, rendered_urls: set = None,
                               indent_level: int = 0, prefix: str = "", is_last: bool = True):
        """Recursively render complete taxonomy tree with ASCII Art style.
        
        Args:
            node: Taxonomy node dictionary
            papers: Dictionary mapping paper_id to paper metadata
            lines: List to append output lines to
            display_index_map: Dictionary mapping paper_id to display index (序号)
            rendered_urls: Set of already rendered paper URLs (for deduplication)
            indent_level: Current indentation level (0 for root)
            prefix: Prefix string for tree drawing (built recursively)
            is_last: Whether this is the last child at current level
        """
        if display_index_map is None:
            display_index_map = {}
        if rendered_urls is None:
            rendered_urls = set()
        name = node.get("name", "")

        subtopics = node.get("subtopics", [])
        papers_list = node.get("papers", [])
        
        # Render current node (skip root node name if already shown in header)
        if indent_level == 0:
            # Root node - don't show name again (already in section header)
            current_line = ""
        else:
            # Build tree connector: use └── for last item, ├── for others
            connector = "└── " if is_last else "├── "
            current_line = f"{prefix}{connector}**{name}**"
            
            # Add paper count if leaf node
            if papers_list:
                paper_count = len(papers_list)
                current_line += f" ★ ({paper_count} paper{'s' if paper_count != 1 else ''})"
            
            lines.append(current_line)
            
            
            # Render papers in this node (leaf node)
            if papers_list:
                paper_prefix = prefix + ("    " if is_last else "│   ")
                for paper_idx, paper_id in enumerate(papers_list):
                    paper_info = papers.get(paper_id, {})
                    
                    if not paper_info and "arxiv" in paper_id.lower():
                        import re
                        arxiv_match = re.search(r'arxiv[:\.]?(\d+\.\d+)', paper_id, re.IGNORECASE)
                        if arxiv_match:
                            arxiv_id = arxiv_match.group(1)
                            paper_info = papers.get(arxiv_id, {})
                            if not paper_info:
                                paper_info = papers.get(f"arxiv:{arxiv_id}", {})
                            if not paper_info:
                                for key, info in papers.items():
                                    if isinstance(info, dict):
                                        url = info.get("url", "")
                                        if arxiv_id in url:
                                            paper_info = info
                                            break
                    
                    if not paper_info and paper_id.startswith("title:"):
                        import re
                        search_title = paper_id[6:].lower().strip()  # 提取 title 部分
                        stop_words = {'a', 'an', 'the', 'of', 'to', 'for', 'in', 'on', 'and', 'or', 'via', 'with', 'by'}
                        normalized_search = re.sub(r'[^a-z0-9\s]', '', search_title).split()
                        key_words = [w for w in normalized_search if w not in stop_words and len(w) > 2]
                        
                        if key_words:
                            best_match = None
                            best_score = 0
                            for key, info in papers.items():
                                if isinstance(info, dict):
                                    candidate_title = info.get("title", "").lower()
                                    normalized_candidate = re.sub(r'[^a-z0-9\s]', '', candidate_title)
                                    match_count = sum(1 for word in key_words if word in normalized_candidate)
                                    score = match_count / len(key_words) if key_words else 0
                                    if score > best_score and score >= 0.5: 
                                        best_score = score
                                        best_match = info
                            if best_match:
                                paper_info = best_match
                    
                    if not paper_info and paper_id.startswith("doi:"):
                        search_doi = paper_id[4:].lower().strip()
                        for key, info in papers.items():
                            if isinstance(info, dict):
                                candidate_doi = str(info.get("doi", "")).lower()
                                if search_doi and candidate_doi and search_doi in candidate_doi:
                                    paper_info = info
                                    break
                    
                    title = paper_info.get("title", paper_id) if paper_info else paper_id
                    authors = paper_info.get("authors", []) if paper_info else []
                    year = paper_info.get("year") if paper_info else None
                    url = paper_info.get("url", "") if paper_info else ""
                    
                    dedup_key = url if url else title  
                    if dedup_key in rendered_urls:
                        continue  
                    rendered_urls.add(dedup_key)
                    
                    display_index = display_index_map.get(paper_id)
                    if display_index is None and paper_info:
                        display_index = paper_info.get("display_index")
                    
                    is_last_paper = (paper_idx == len(papers_list) - 1)
                    paper_connector = "└── " if is_last_paper else "├── "
                    
                    # Build paper line
                    paper_line = f"{paper_prefix}{paper_connector}"
                    
                    # Add display index if available
                    if display_index is not None:
                        paper_line += f"[{display_index}] "
                    
                    # Add title
                    paper_line += title
                    
                    # Add authors and year
                    if authors:
                        first_author = authors[0].split(",")[0].strip() if "," in authors[0] else authors[0].strip()
                        if len(authors) > 1:
                            paper_line += f" ({first_author} et al."
                        else:
                            paper_line += f" ({first_author}"
                        if year:
                            paper_line += f", {year})"
                        else:
                            paper_line += ")"
                    
                    # Add URL as Markdown link
                    if url:
                        paper_line += f" [View paper]({url})"
                    
                    lines.append(paper_line)
        
        # Prepare prefix for children
        if indent_level == 0:
            # Root level - start with empty prefix
            child_prefix = ""
        else:
            # Extend prefix: add vertical line if not last, spaces if last
            child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Recursively render subtopics
        if subtopics:
            for subtopic_idx, subtopic in enumerate(subtopics):
                is_last_subtopic = (subtopic_idx == len(subtopics) - 1) and not papers_list
                self._render_taxonomy_tree(
                    subtopic, papers, lines,
                    display_index_map, rendered_urls,
                    indent_level + 1, 
                    child_prefix, 
                    is_last_subtopic
                )
        
        # Add blank line after root node for readability
        if indent_level == 0:
            lines.append("")
    
    # ==================== Helper Functions ====================
    
    def _extract_taxonomy_from_survey_md(self, data: Dict) -> Optional[str]:
        """Extract taxonomy tree section from core_task_survey.md if available.
        
        This method tries to read the pre-generated survey markdown file and
        extract the taxonomy tree section, which already has correct formatting
        and includes the original paper with proper indexing.
        """
        try:
            from pathlib import Path
            
            report_path = getattr(self, '_current_report_path', None)
            if not report_path:
                return None
            
            phase3_dir = Path(report_path).parent
            
            full_md_path = phase3_dir / "core_task_survey" / "reports" / "core_task_survey.md"
            
            if not full_md_path.exists():
                self.logger.warning(f"Survey markdown not found: {full_md_path}")
                return None
            
            content = full_md_path.read_text(encoding='utf-8')
            
            lines = content.split('\n')
            taxonomy_lines = []
            in_taxonomy = False
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('## Taxonomy'):
                    in_taxonomy = True
                    continue  
                elif in_taxonomy and (
                    stripped.startswith('# ')  # 下一个一级标题
                    or (stripped.startswith('## ') and not stripped.startswith('## Taxonomy'))  # 其他二级标题，如 \"## Narrative\"
                ):
                    break
                elif in_taxonomy:
                    taxonomy_lines.append(line)
            
            while taxonomy_lines and not taxonomy_lines[0].strip():
                taxonomy_lines.pop(0)
            while taxonomy_lines and not taxonomy_lines[-1].strip():
                taxonomy_lines.pop()
            
            import re
            processed_lines = []
            for line in taxonomy_lines:
                processed_line = re.sub(
                    r'\[(https?://[^\]]+)\](?!\()', 
                    r'[View paper](\1)',
                    line
                )
                processed_lines.append(processed_line)
            
            if processed_lines:
                return '\n'.join(processed_lines)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract taxonomy from survey md: {e}")
            return None
    
    def _count_leaf_nodes(self, taxonomy: Dict) -> int:
        """Recursively count leaf nodes (nodes with papers) in taxonomy."""
        count = 0
        subtopics = taxonomy.get("subtopics", [])
        papers = taxonomy.get("papers", [])
        
        if papers:
            # This is a leaf node
            return 1
        elif subtopics:
            # Recursively count leaf nodes in subtopics
            for subtopic in subtopics:
                count += self._count_leaf_nodes(subtopic)
        
        return count
    
    def _safe_filename(self, text: str) -> str:
        """Generate a safe filename from text."""
        # Remove special characters, keep only alphanumeric and spaces
        safe = re.sub(r'[^\w\s-]', '', text)
        # Replace spaces with underscores
        safe = re.sub(r'[-\s]+', '_', safe)
        # Limit length
        return safe[:50]
    
    def _build_similarity_appendix(self, data: Dict) -> str:
        """
        Build appendix for text similarity detection results.
        
        Collects all papers with high textual similarity from both core task comparisons
        and contribution analysis, and generates an appendix section.
        """
        # Preferred path: use the unified plagiarism_detection index from Phase3.
        # Support both new field name (plagiarism_detection) and legacy (textual_similarity) for backward compatibility
        ts = data.get("plagiarism_detection") or data.get("textual_similarity") or {}
        stats = ts.get("statistics") or {}
        items = ts.get("items") or []
        
        # Generate note_en based on statistics if not provided
        total_checked = stats.get("total_papers_checked", 0)
        papers_with_plagiarism = stats.get("papers_with_plagiarism", 0)
        total_segments = stats.get("total_unique_segments", 0)
        
        if papers_with_plagiarism > 0:
            note_en = stats.get(
                "note_en",
                f"Textual similarity detection checked {total_checked} papers and found {total_segments} similarity segment(s) across {papers_with_plagiarism} paper(s)."
            )
        else:
            note_en = stats.get(
                "note_en",
                "No high-similarity text segments were detected across any compared papers."
            )

        # If textual_similarity is present (even with empty items), render based on it.
        if ts:
            lines: List[str] = []
            lines.append("## Appendix: Text Similarity Detection")
            lines.append("")
            lines.append(note_en)
            lines.append("")

            if not items:
                # No segments found; just return the note.
                return "\n".join(lines)

            lines.append(
                f"The following **{len(items)} paper(s)** were detected to have high textual similarity with the original paper."
            )
            lines.append(
                "These may represent different versions of the same work, duplicate submissions, "
                "or papers with substantial textual overlap. Readers are advised to verify these "
                "relationships independently."
            )
            lines.append("")

            for idx, entry in enumerate(items, 1):
                title = entry.get("title", "Unknown")
                url = entry.get("url", "")
                
                # Support both new aggregated 'sources' and legacy 'source'
                sources = entry.get("sources") or []
                if not sources and "source" in entry:
                    sources = [entry["source"]]
                
                source_text = ", ".join(sources) if sources else "Unknown source"
                segments = entry.get("segments") or []

                lines.append(f"### {idx}. {title}")
                lines.append("")

                if url:
                    lines.append(f"**URL**: [{url}]({url})")
                    lines.append("")

                lines.append(f"**Detected in**: {source_text}")
                lines.append("")

                if segments:
                    lines.append(f"**Number of similar text segments**: {len(segments)}")
                    lines.append("")

                lines.append(
                    "⚠️ **Note**: This paper shows substantial textual similarity with the original paper. "
                    "It may be a different version, a duplicate submission, or contain significant overlapping content. "
                    "Please review carefully to determine the nature of the relationship."
                )
                lines.append("")

                if idx < len(items):
                    lines.append("---")
                    lines.append("")

            return "\n".join(lines)

        # Collect all papers with high textual similarity from both core task comparisons
        # and contribution analysis, using a dictionary to aggregate by URL/title.
        aggregated_legacy: Dict[str, Dict[str, Any]] = {}

        def add_legacy(title: str, url: str, source: str, count: int):
            key = url if url else title
            if key not in aggregated_legacy:
                aggregated_legacy[key] = {
                    "title": title,
                    "url": url,
                    "sources": [],
                    "segments_count": 0
                }
            entry = aggregated_legacy[key]
            if source not in entry["sources"]:
                entry["sources"].append(source)
            # Legacy path doesn't have segment de-duplication, just take max or sum?
            # Let's take the max to be safe as segments often overlap.
            entry["segments_count"] = max(entry["segments_count"], count)

        # Collect from core task comparisons
        core_comparisons = data.get("core_task_comparisons", {}).get("comparisons", [])
        for comp in core_comparisons:
            is_duplicate = comp.get("is_duplicate_variant", False)
            segments = comp.get("textual_similarity_segments") or comp.get("similarity_segments") or []
            if is_duplicate or (segments and len(segments) > 0):
                add_legacy(
                    title=comp.get("candidate_paper_title", "Unknown"),
                    url=comp.get("candidate_paper_url", ""),
                    source="Core Task Comparison (Same Category)",
                    count=len(segments) if segments else 1 # 1 if is_duplicate but no segments
                )
        
        # Collect from contribution analysis
        contrib_analysis = data.get("contribution_analysis", {})
        contribs_with_results = contrib_analysis.get("contributions_with_results", [])
        for contrib in contribs_with_results:
            c_name = contrib.get("contribution_name", "Unknown")
            for comp in contrib.get("comparisons", []):
                segments = comp.get("textual_similarity_segments") or comp.get("similarity_segments") or []
                if segments and len(segments) > 0:
                    add_legacy(
                        title=comp.get("title", comp.get("candidate_paper_title", "Unknown")),
                        url=comp.get("url", comp.get("candidate_paper_url", "")),
                        source=f"Contribution Analysis ({c_name})",
                        count=len(segments)
                    )
        
        # Convert aggregated to list
        similar_papers = []
        for key in sorted(aggregated_legacy.keys()):
            entry = aggregated_legacy[key]
            similar_papers.append({
                "title": entry["title"],
                "url": entry["url"],
                "source": ", ".join(entry["sources"]),
                "segments_count": entry["segments_count"]
            })
        
        if not similar_papers:
            return ""
        
        lines = [
            "## Appendix: Text Similarity Detection",
            "",
            f"The following **{len(similar_papers)} paper(s)** were detected to have high textual similarity with the original paper. ",
            "These may represent different versions of the same work, duplicate submissions, or papers with substantial textual overlap. ",
            "Readers are advised to verify these relationships independently.",
            ""
        ]
        
        for idx, paper in enumerate(similar_papers, 1):
            lines.append(f"### {idx}. {paper['title']}")
            lines.append("")
            
            if paper['url']:
                lines.append(f"**URL**: [{paper['url']}]({paper['url']})")
                lines.append("")
            
            lines.append(f"**Detected in**: {paper['source']}")
            lines.append("")
            
            if paper.get('segments_count'):
                lines.append(f"**Number of similar text segments**: {paper['segments_count']}")
                lines.append("")
            
            lines.append("⚠️ **Note**: This paper shows substantial textual similarity with the original paper. "
                        "It may be a different version, a duplicate submission, or contain significant overlapping content. "
                        "Please review carefully to determine the nature of the relationship.")
            lines.append("")
            
            if idx < len(similar_papers):
                lines.append("---")
                lines.append("")
        
        return "\n".join(lines)
    
    def _build_references_section(self, data: Dict) -> str:
        """
        Build a simple References section using the Phase3 references field.

        Only uses index, title, and url (if available), sorted by index.
        """
        refs = data.get("references") or {}
        items = refs.get("items") or []
        if not items or not isinstance(items, list):
            return ""

        # Keep only dict items and sort stably by index
        cleaned_items = [it for it in items if isinstance(it, dict) and "index" in it]
        if not cleaned_items:
            return ""

        cleaned_items.sort(key=lambda it: it.get("index", 0))

        lines: List[str] = []
        lines.append("## References")
        lines.append("")

        for it in cleaned_items:
            idx = it.get("index")
            title = (it.get("title") or "").strip() or "(no title)"
            url = (it.get("url") or "").strip()

            if url:
                # Use a bullet list with a clickable Markdown link for nicer layout
                lines.append(f"- [{idx}] {title} [View paper]({url})")
            else:
                lines.append(f"- [{idx}] {title}")

        lines.append("")
        return "\n".join(lines)
    
    def _generate_pdf_from_markdown(self, md_path: Path, pdf_path: Path) -> bool:
        """
        Generate PDF from markdown file.
        
        Reuses the PDF generation logic from this module.
        Tries multiple methods: markdown-pdf, weasyprint, pdfkit.
        """
        try:
            # Read markdown content
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            
            # Fix markdown for PDF (same flow as this module)
            fixed_content = self._fix_markdown_for_pdf(md_content)
            
            # Write fixed content to temporary file
            temp_md = md_path.parent / f"{md_path.stem}_temp_fixed.md"
            with open(temp_md, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            
            # Try markdown-pdf (npm package) first
            css_path = Path(__file__).parent / "pdf-styles-compact.css"
            markdown_pdf_cmd = shutil.which("markdown-pdf")
            
            if markdown_pdf_cmd:
                self.logger.info("Using markdown-pdf to generate PDF...")
                cmd = [
                    markdown_pdf_cmd,
                    str(temp_md),
                    "-f", "A4",
                    "-b", "1cm",
                    "-d", "1000",
                    "-o", str(pdf_path)
                ]
                
                if css_path.exists():
                    cmd.extend(["-s", str(css_path)])
                else:
                    self.logger.warning(f"CSS file not found: {css_path}")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    temp_md.unlink()
                    return True
                else:
                    self.logger.warning(f"markdown-pdf failed: {result.stderr}")
            
            # Fallback: try weasyprint
            try:
                from weasyprint import HTML, CSS
                import markdown
                
                self.logger.info("Using weasyprint to generate PDF...")
                html_content = markdown.markdown(fixed_content, extensions=['extra', 'codehilite'])
                
                css_content = ""
                if css_path.exists():
                    with open(css_path, "r", encoding="utf-8") as f:
                        css_content = f.read()
                
                html_doc = HTML(string=html_content)
                # Add zoom parameter to match markdown-pdf rendering size
                html_doc.write_pdf(
                    pdf_path, 
                    stylesheets=[CSS(string=css_content)],
                    zoom=1.0,  # Adjust if needed (smaller = more compact)
                    presentational_hints=True
                )
                temp_md.unlink()
                self.logger.info(f"✅ PDF report saved to: {pdf_path}")
                return True
            except ImportError:
                pass
            except Exception as e:
                self.logger.warning(f"weasyprint failed: {e}")
            
            # Fallback: try pdfkit
            try:
                import pdfkit
                import markdown
                
                self.logger.info("Using pdfkit to generate PDF...")
                html_content = markdown.markdown(fixed_content, extensions=['extra', 'codehilite'])
                
                options = {
                    'page-size': 'A4',
                    'margin-top': '1cm',
                    'margin-right': '1cm',
                    'margin-bottom': '1cm',
                    'margin-left': '1cm',
                }
                
                pdfkit.from_string(
                    html_content,
                    str(pdf_path),
                    options=options,
                    css=str(css_path) if css_path.exists() else None
                )
                temp_md.unlink()
                return True
            except ImportError:
                pass
            except Exception as e:
                self.logger.warning(f"pdfkit failed: {e}")
            
            # All methods failed
            self.logger.error(
                "PDF generation failed. Please install one of:\n"
                "  - markdown-pdf (npm): npm install -g markdown-pdf\n"
                "  - weasyprint: pip install weasyprint markdown\n"
                "  - pdfkit: pip install pdfkit (requires wkhtmltopdf)"
            )
            temp_md.unlink()
            return False
            
        except Exception as e:
            self.logger.error(f"PDF generation error: {e}", exc_info=True)
            return False
    
    def _fix_markdown_for_pdf(self, md_content: str) -> str:
        """Fix markdown content for PDF generation (shared helper in this module)."""
        # Detect language
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', md_content))
        link_text = "查看论文" if has_chinese else "View paper"
        
        fixed_content = md_content
        
        # Fix links where text is still a URL: [url](url) -> [View paper](url)
        link_pattern1 = r'\[(https?://[^\]]+)\]\((https?://[^\)]+)\)'
        def fix_link1(match):
            url = match.group(2)
            return f'[{link_text}]({url})'
        
        fixed_content = re.sub(link_pattern1, fix_link1, fixed_content)
        
        # Fix [→ View paper] to [View paper] (remove arrow if present)
        fixed_content = re.sub(r'\[→\s*', '[', fixed_content)
        
        # Fix math formulas: $A$ -> A (preserve case)
        def fix_math_formula(match):
            letter = match.group(1)
            return letter.upper() if letter.islower() else letter
        
        math_pattern = r'\$([A-Za-z])\$'
        fixed_content = re.sub(math_pattern, fix_math_formula, fixed_content)
        
        return fixed_content


# Backward compatibility alias
LightweightNoveltyReportPhase = LightweightReportGenerator
# Back-compat alias so existing scripts importing NoveltyReportGenerator keep working
NoveltyReportGenerator = LightweightReportGenerator
