"""
Phase 3: LLM Analyzer

Handles LLM-based analysis for contribution comparison and core task distinction.
Includes prompt construction, response parsing, and result validation.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from paper_novelty_pipeline.models import (
    RetrievedPaper,
    ExtractedContent,
    ContributionAnalysis,
    EvidencePair,
    EvidenceLocation,
    TextualSimilaritySegment,
    ASPECT,
    ContributionClaim,
    ContributionSlot,
)

from paper_novelty_pipeline.phases.phase3.evidence_verifier import EvidenceVerifier
from paper_novelty_pipeline.phases.phase3.utils import parse_json_flexible, sanitize_id
from paper_novelty_pipeline.config import (
    LLM_MAX_TOKENS,
    LLM_MAX_PROMPT_CHARS,
    MAX_CONTEXT_CHARS,
    PHASE3_MIN_CONTEXT_CHARS,
    PHASE3_MIN_LLM_TOKENS,
    PHASE3_MAX_LLM_TOKENS,
    PHASE3_MIN_SIMILARITY_WORDS,
    PHASE3_MAX_SIMILARITY_SEGMENTS,
)


# ============================================================================
# Constants
# ============================================================================

MIN_CONTEXT_CHARS = PHASE3_MIN_CONTEXT_CHARS
MIN_LLM_TOKENS = PHASE3_MIN_LLM_TOKENS
MAX_LLM_TOKENS = PHASE3_MAX_LLM_TOKENS
DEFAULT_LLM_TOKENS = LLM_MAX_TOKENS

# Similarity thresholds
MIN_SIMILARITY_WORDS = PHASE3_MIN_SIMILARITY_WORDS  # Minimum word count for similarity segments
MAX_SIMILARITY_SEGMENTS = PHASE3_MAX_SIMILARITY_SEGMENTS


# ============================================================================
# Contribution Wrapper (for backward compatibility)
# ============================================================================

class ContributionWrapper:
    """
    Wrapper to adapt ContributionClaim to ContributionSlot-like interface.
    Used for backward compatibility with legacy code paths.
    """
    def __init__(self, claim: ContributionClaim):
        self.name = claim.name
        self.description = claim.description or claim.author_claim_text or ""
        self.prior_work_query = claim.prior_work_query or ""
        self.source_hint = getattr(claim, 'source_hint', 'unknown')
        # For compatibility with legacy code that checks aspect
        self.aspect = "contribution"


# ============================================================================
# LLM Analyzer Class
# ============================================================================

class LLMAnalyzer:
    """
    Handles LLM-based analysis for contribution comparison and core task distinction.
    
    Responsibilities:
    - Construct prompts for batch contribution comparison
    - Parse and validate LLM responses
    - Extract evidence pairs and similarity segments
    - Save raw LLM responses for debugging
    """
    
    def __init__(
        self,
        llm_client: Any,
        logger: logging.Logger,
        output_base_dir: Optional[str] = None,
        evidence_verifier: Optional[EvidenceVerifier] = None,
    ):
        """
        Initialize LLM analyzer.
        
        Args:
            llm_client: LLM client instance (from services.llm_client)
            logger: Logger instance
            output_base_dir: Base directory for saving outputs
            evidence_verifier: Optional EvidenceVerifier instance (creates one if not provided)
        """
        self.llm_client = llm_client
        self.logger = logger
        self.output_base_dir = output_base_dir
        self.evidence_verifier = evidence_verifier or EvidenceVerifier(logger)
    
    def _get_max_tokens(self, default: int = DEFAULT_LLM_TOKENS) -> int:
        """Get max tokens for LLM calls, clamped to safe range."""
        max_tokens = int(LLM_MAX_TOKENS) if LLM_MAX_TOKENS else default
        return max(MIN_LLM_TOKENS, min(max_tokens, MAX_LLM_TOKENS))
    
    def analyze_candidate_contributions(
        self,
        candidate: RetrievedPaper,
        contribution_tuples: List[Tuple[ASPECT, str, str]],  # (aspect, contribution_name, source_query)
        extracted_content: ExtractedContent,
        orig_full_text: str,
        cand_full_text: str,
        candidate_citation: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> List[ContributionAnalysis]:
        """
        Analyze multiple contributions for a single candidate in one LLM call.
        
        Textual similarity detection now runs separately in Part 3
        (`run_phase3_textual_similarity.py`) before the comparison steps.
        Results are stored at `phase3/textual_similarity_detection/results.json`
        and reused by later parts. This method only builds contribution
        comparison prompts; it does not perform similarity detection.
        
        Args:
            candidate: The candidate paper to compare against
            contribution_tuples: List of (aspect, contribution_name, source_query) tuples to compare
            extracted_content: Original paper's extracted content with contributions
            orig_full_text: Original paper full text
            cand_full_text: Candidate paper full text
            candidate_citation: Optional citation reference (e.g., "Alias[51]") to include in prompt
            timeout: Optional timeout override
            
        Returns:
            List of ContributionAnalysis objects, one per contribution
        """
        if not self.llm_client:
            self.logger.error("LLM client not initialized")
            return []
        
        if not contribution_tuples:
            return []
        
        # Collect complete ContributionClaim objects from ExtractedContent
        # Support both new structure (contributions) and legacy structure (slots) for backward compatibility
        original_contributions = []
        for aspect, contribution_name, _ in contribution_tuples:
            found_contribution = None
            
            # Try new structure first: extracted_content.contributions (List[ContributionClaim])
            if hasattr(extracted_content, 'contributions') and extracted_content.contributions:
                for contribution in extracted_content.contributions:
                    if contribution.name == contribution_name:
                        found_contribution = ContributionWrapper(contribution)
                        break
            
            # Fallback to legacy structure: extracted_content.slots (for backward compatibility)
            if not found_contribution and hasattr(extracted_content, 'slots') and extracted_content.slots:
                for group in extracted_content.slots:
                    if group.aspect == aspect:
                        for contribution in group.contributions:
                            if contribution.name == contribution_name:
                                found_contribution = contribution
                                break
                        if found_contribution:
                            break
            
            if not found_contribution:
                self.logger.warning(
                    f"Phase3: Cannot find contribution '{contribution_name}' under aspect '{aspect}' "
                    f"in original paper's ExtractedContent. Skipping."
                )
                continue
            
            original_contributions.append(found_contribution)
        
        if not original_contributions:
            self.logger.warning("Phase3: No valid contributions found for batch comparison")
            return []
        
        # Truncate full texts for context
        # At this point orig_full_text / cand_full_text have already been passed
        # through truncate_at_references and MAX_CONTEXT_CHARS in _get_*_full_text.
        # Here we only enforce a per-call budget derived from LLM_MAX_PROMPT_CHARS.
        orig_text_no_refs = orig_full_text or ""
        cand_text_no_refs = cand_full_text or ""
        
        # Apply per-call character limit (20W characters), leaving headroom for prompt.
        raw_limit = int(LLM_MAX_PROMPT_CHARS) - 50000  # Leave space for prompt other parts
        max_context_chars = min(raw_limit, MAX_CONTEXT_CHARS)
        if max_context_chars < MIN_CONTEXT_CHARS:
            max_context_chars = MIN_CONTEXT_CHARS
        
        orig_text_truncated = (
            orig_text_no_refs[:max_context_chars]
            if len(orig_text_no_refs) > max_context_chars
            else orig_text_no_refs
        )
        cand_text_truncated = (
            cand_text_no_refs[:max_context_chars]
            if len(cand_text_no_refs) > max_context_chars
            else cand_text_no_refs
        )
        
        # Normalize texts before passing to LLM (for consistent matching)
        orig_text_normalized = self.evidence_verifier.normalize_text(orig_text_truncated)
        cand_text_normalized = self.evidence_verifier.normalize_text(cand_text_truncated)
        
        # Build contribution descriptions section with prior_work_query
        # Note: Now only handling 'contribution' aspect
        contributions_section = []
        for idx, contribution in enumerate(original_contributions, 1):
            contributions_section.append(
                f"### Contribution {idx}: {contribution.name}\n"
                f"**Description**:\n```\n{contribution.description}\n```\n\n"
                f"**Prior Work Query** (used to retrieve this candidate):\n```\n{contribution.prior_work_query}\n```\n\n"
                f"**Context**: This candidate was retrieved using the above query, suggesting potential similarity "
                f"on this contribution. Please analyze both novelty and overlap by searching the candidate's full text.\n"
            )
        
        contributions_text = "\n\n---\n\n".join(contributions_section)
        
        # Call LLM for Task 1: Refutation Assessment with Audit-Retry Loop
        contribution_analyses = []
        try:
            max_toks = self._get_max_tokens()
            scaled_max_tokens = min(max_toks + (len(original_contributions) - 1) * 2000, 16000)

            ref_system_prompt = self._build_refutation_system_prompt()
            ref_user_prompt = self._build_refutation_user_prompt(
                candidate, len(original_contributions), contributions_text,
                orig_text_normalized, cand_text_normalized, candidate_citation
            )
            # Fresh messages for each attempt to avoid Token overflow or model bias
            ref_initial_messages = [
                {"role": "system", "content": ref_system_prompt},
                {"role": "user", "content": ref_user_prompt},
            ]

            # Call LLM for Task 1: Refutation Assessment with retry on ANY error
            # Network retries are handled by the LLM client itself (MAX_RETRIES).
            # Enable prompt caching for the system prompt (saves ~90% on repeated calls).
            max_parse_retries = 3
            parse_retry_count = 0
            parsing_succeeded = False
            
            while parse_retry_count < max_parse_retries and not parsing_succeeded:
                try:
                    if parse_retry_count > 0:
                        self.logger.info(
                            f"Phase3: Task 1 (Refutation) retry {parse_retry_count}/{max_parse_retries} "
                            f"for candidate {candidate.paper_id}"
                        )
                    else:
                        self.logger.info(f"Phase3: Task 1 (Refutation) call for candidate {candidate.paper_id}")
                    
                    ref_raw = self.llm_client.generate_json(ref_initial_messages, max_tokens=scaled_max_tokens, use_cache=True)
                    
                    # Detect [N] bug: API sometimes returns just a citation index like [62] instead of proper JSON
                    # This happens when the API's JSON mode incorrectly parses citation markers in the prompt
                    if isinstance(ref_raw, list) and len(ref_raw) == 1 and isinstance(ref_raw[0], (int, float)):
                        self.logger.warning(
                            f"Detected [N] bug for candidate {candidate.paper_id} (got {ref_raw}), "
                            "retrying with text mode..."
                        )
                        try:
                            text_response = self.llm_client.generate(ref_initial_messages, max_tokens=scaled_max_tokens, use_cache=True)
                            if text_response:
                                ref_raw = self.llm_client._parse_json_content(text_response)
                                self.logger.info(f"Text mode retry succeeded for candidate {candidate.paper_id}")
                        except Exception as retry_err:
                            self.logger.error(f"Text mode retry failed for candidate {candidate.paper_id}: {retry_err}")
                            raise  # Re-raise to trigger outer retry loop
                    
                    self._save_llm_response(ref_raw, "refutation", candidate.paper_id, context="contribution")
                    ref_parsed = parse_json_flexible(ref_raw)

                    if isinstance(ref_parsed, dict) and "contribution_analyses" in ref_parsed:
                        contribution_analyses = self._parse_contribution_analyses(
                            ref_parsed, original_contributions, contribution_tuples,
                            orig_text_norm=orig_text_normalized,
                            cand_text_norm=cand_text_normalized
                        )
                        
                        # Optional audit logging (no retry)
                        audit_failures = []
                        for analysis in contribution_analyses:
                            if analysis.refutation_status == "can_refute":
                                has_verified = any(
                                    pair.original_location.found and pair.candidate_location.found 
                                    for pair in (analysis.refutation_evidence.get("evidence_pairs", []) if analysis.refutation_evidence else [])
                                )
                                if not has_verified:
                                    audit_failures.append(analysis.contribution_name)
                        
                        if audit_failures:
                            self.logger.warning(
                                f"Audit: No verified evidence for contributions {audit_failures}. "
                                f"Accepting result without retry as per configuration."
                            )
                        
                        # Parsing succeeded!
                        parsing_succeeded = True
                        self.logger.info(f"Successfully parsed refutation response for {candidate.paper_id}")
                    else:
                        # JSON parsed but structure is wrong - this should trigger a retry
                        raise ValueError(f"Response structure invalid: missing 'contribution_analyses' field")
                
                except Exception as parse_err:
                    parse_retry_count += 1
                    if parse_retry_count < max_parse_retries:
                        self.logger.warning(
                            f"Failed to parse refutation response for {candidate.paper_id} "
                            f"(attempt {parse_retry_count}/{max_parse_retries}): {parse_err}. Retrying..."
                        )
                    else:
                        self.logger.error(
                            f"Failed to parse refutation response for {candidate.paper_id} "
                            f"after {max_parse_retries} attempts: {parse_err}. Using empty analyses."
                        )
                        contribution_analyses = []  # Final fallback

        except Exception as e:
            self.logger.error(f"Task 1 (Refutation) failed for candidate {candidate.paper_id}: {e}")

        # ==================== Task 2: Textual Similarity Detection ====================
        # NOTE: Task 2 has been moved to the independent module run_textual_similarity.py
        # This ensures each paper is only analyzed once (by canonical_id) and results
        # are loaded from phase3/textual_similarity_detection/results.json
        
        return contribution_analyses

    def analyze_core_task_subtopics(
        self,
        *,
        core_task_text: str,
        original_leaf: Dict[str, Any],
        sibling_subtopics: List[Dict[str, Any]],
        original_abstract: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Compare the original leaf with sibling subtopics (taxonomy-level).

        Inputs are lightweight: leaf/subtopic names, scope/exclude notes,
        counts, and representative papers (title/abstract/rank/id).
        """
        if not self.llm_client:
            self.logger.error("LLM client not initialized")
            return None

        # Build prompt
        system_prompt = (
            "You are an expert academic research analyst specializing in comparative literature reviews. "
            "Your goal is to precisely situate a specific research paper within its immediate taxonomic neighborhood. "
            "\n\nAnalytical Guidelines:\n"
            "1. **Boundary Analysis**: Examine the 'scope_note' and 'exclude_note' of the original leaf vs. sibling subtopics to identify the formal conceptual boundaries.\n"
            "2. **Technical Differentiation**: Compare the methodology and core task of the original paper (from its abstract) with the representative papers of sibling subtopics. Look for differences in approach, modality, optimization targets, or theoretical grounding.\n"
            "3. **Relationship Mapping**: Identify shared foundations (similarities) and distinct branching points (differences).\n"
            "4. **Grounded Reasoning**: Every claim must be supported by the provided text. If a sibling subtopic is broad, treat its representative papers as a sample of its focus.\n"
            "5. **Academic Tone**: Use precise technical terminology and maintain a neutral, analytical tone. Do not invent details or assume information not present in the abstracts."
        )

        def _format_subtopic(st: Dict[str, Any]) -> Dict[str, Any]:
            papers = []
            for rp in st.get("papers", []) or []:
                if not isinstance(rp, dict):
                    continue
                papers.append(
                    {
                        "title": rp.get("title", ""),
                        "abstract": rp.get("abstract", ""),
                        "id": rp.get("id", ""),
                    }
                )
            return {
                "name": st.get("subtopic_name", ""),
                "scope_note": st.get("scope_note", ""),
                "exclude_note": st.get("exclude_note", ""),
                "leaf_count": st.get("leaf_count", 0),
                "paper_count": st.get("paper_count", 0),
                "papers": papers,
            }

        formatted_subtopics = [_format_subtopic(st) for st in sibling_subtopics]

        user_payload = {
            "original_paper": {
                "core_task": core_task_text,
                "abstract": original_abstract,
            },
            "original_leaf": {
                "name": original_leaf.get("leaf_name", ""),
                "scope_note": original_leaf.get("scope_note", ""),
                "exclude_note": original_leaf.get("exclude_note", ""),
                "paper_ids": original_leaf.get("paper_ids", []),
            },
            "sibling_subtopics": formatted_subtopics,
            "instructions": (
                "Return concise JSON with keys: "
                "overall (2-4 sentences), similarities (list), differences (list), "
                "suggested_search_directions (optional list). "
                "Base reasoning on the original paper's abstract/core_task, "
                "scope/exclude notes, and the provided sibling subtopic papers "
                "(titles/abstracts). Do not invent facts."
            ),
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        try:
            result = self.llm_client.generate_json(
                messages, max_tokens=500
            )
            if isinstance(result, dict):
                overall = str(result.get("overall", "")).strip()
                similarities = result.get("similarities", [])
                differences = result.get("differences", [])
                suggested = result.get("suggested_search_directions") or []

                return {
                    "overall": overall,
                    "similarities": similarities if isinstance(similarities, list) else [],
                    "differences": differences if isinstance(differences, list) else [],
                    "suggested_search_directions": suggested if isinstance(suggested, list) else [],
                }

            self.logger.warning(f"LLM returned invalid format for subtopic comparison: {result}")
            return None
        except Exception as e:
            self.logger.error(f"LLM call failed for subtopic comparison: {e}")
            return None
    
    def analyze_core_task_distinction(
        self,
        core_task_text: str,
        original_title: str,
        original_abstract: str,
        original_paper_id: str,
        candidate_title: str,
        candidate_abstract: str,
        candidate_paper_id: str,
        taxonomy_context: Optional[Dict[str, Any]] = None,
        relationship: str = "sibling",
        candidate_fulltext: Optional[str] = None,
        original_fulltext: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """
        Generate similarity and difference analysis between original and candidate papers
        within the core_task domain context, optionally using taxonomy information.
        
        This method handles the brief distinction/duplicate check used in
        core-task comparisons (2–3 sentence summary + duplicate/variant check).
        Textual similarity detection runs separately in Part 3
        (`run_phase3_textual_similarity.py`) and its results are reused later
        when generating the final report.
        
        NOTE: Currently only sibling comparison is implemented. Cousin comparison (papers
        sharing a common ancestor but in different leaves) could be added as a fallback
        when no siblings exist. See .cursor/plans/20251210_cousins.plan.md for design.
        
        Args:
            core_task_text: Core task description from Phase1
            original_title: Original paper title
            original_abstract: Original paper abstract
            original_paper_id: Original paper ID
            candidate_title: Candidate paper title
            candidate_abstract: Candidate paper abstract
            candidate_paper_id: Candidate paper ID
            taxonomy_context: Optional taxonomy context information
            relationship: Taxonomy relationship - currently only "sibling" (same leaf) is supported
            candidate_fulltext: Optional full text of candidate paper (for detailed sibling comparison)
            original_fulltext: Optional full text of original paper (for detailed sibling comparison)
            
        Returns:
            Dictionary with keys: "is_duplicate_variant", "brief_comparison"
            or None if LLM call fails
        """
        if not self.llm_client:
            self.logger.error("LLM client not initialized")
            return None
        
        # Format taxonomy context if available
        taxonomy_context_text = ""
        if taxonomy_context:
            taxonomy_context_text = self._format_taxonomy_context(taxonomy_context)
            if taxonomy_context_text:
                taxonomy_context_text = f"\n{taxonomy_context_text}\n"
        
        # Build system prompt based on relationship
        system_prompt = self._build_core_task_distinction_system_prompt(
            core_task_text, taxonomy_context_text, relationship
        )
        
        # Prepare text content: use fulltext if available, otherwise use abstract
        # Apply truncation for LLM context limits (similar to contribution analysis)
        max_context_per_paper = 30000  # Conservative limit per paper for core task comparison
        
        # Prepare candidate text
        if candidate_fulltext and len(candidate_fulltext.strip()) > 100:
            cand_text = candidate_fulltext[:max_context_per_paper]
            cand_text_type = "fulltext"
        else:
            cand_text = candidate_abstract or ""
            cand_text_type = "abstract"
        
        # Prepare original text
        if original_fulltext and len(original_fulltext.strip()) > 100:
            orig_text = original_fulltext[:max_context_per_paper]
            orig_text_type = "fulltext"
        else:
            orig_text = original_abstract or ""
            orig_text_type = "abstract"
        
        # Currently only sibling comparison is implemented
        # For sibling papers (same taxonomy leaf), provide detailed comparison
        analysis_instruction = (
            "These papers are classified in the SAME leaf category in the taxonomy (sibling papers). "
            "First check if they are likely duplicates/variants by comparing titles, authors, and content. "
            "If not duplicates, provide a concise 2-3 sentence comparison covering: "
            "(1) their shared taxonomy position, (2) overlapping areas with original paper, (3) key differences."
        )
        max_tokens = 400  

        # Textual similarity detection is handled separately in Part 3
        # (run_phase3_textual_similarity.py); this method only performs brief distinction.

        user_prompt = {
            "core_task_domain": core_task_text,
            "original_paper": {
                "title": original_title,
                "content": orig_text,
                "content_type": orig_text_type
            },
            "candidate_paper": {
                "title": candidate_title,
                "content": cand_text,
                "content_type": cand_text_type
            },
            "analysis_instruction": analysis_instruction
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}
        ]
        
        max_parse_retries = 3
        parse_retry_count = 0
        parsing_succeeded = False
        payload = None
        
        while parse_retry_count < max_parse_retries and not parsing_succeeded:
            try:
                if parse_retry_count > 0:
                    self.logger.info(
                        f"Core task distinction retry {parse_retry_count}/{max_parse_retries} "
                        f"for {original_paper_id} vs {candidate_paper_id}"
                    )
                
                # ==================== Task 1: Brief Distinction ====================
                # Enable prompt caching for the system prompt (saves ~90% on repeated calls)
                result = self.llm_client.generate_json(messages, max_tokens=max_tokens, use_cache=True)
                
                # Detect [N] bug: API sometimes returns just a citation index like [62] instead of proper JSON
                if isinstance(result, list) and len(result) == 1 and isinstance(result[0], (int, float)):
                    self.logger.warning(
                        f"Detected [N] bug for core task distinction (original={original_paper_id}, "
                        f"candidate={candidate_paper_id}, got {result}), retrying with text mode..."
                    )
                    try:
                        text_response = self.llm_client.generate(messages, max_tokens=max_tokens, use_cache=True)
                        if text_response:
                            result = self.llm_client._parse_json_content(text_response)
                            self.logger.info(f"Text mode retry succeeded for core task distinction")
                    except Exception as retry_err:
                        self.logger.error(f"Text mode retry failed for core task distinction: {retry_err}")
                        raise  # Re-raise to trigger outer retry loop
                
                # Extract Task 1 results
                is_duplicate = False
                brief_comparison = ""
                
                if isinstance(result, dict):
                    # Try multiple possible key names for duplicate detection
                    is_duplicate = (
                        result.get("is_duplicate_variant", False) or
                        result.get("is_duplicate", False)
                    )
                    
                    if "brief_comparison" in result:
                        # Expected format
                        brief_comparison = str(result.get("brief_comparison", "")).strip()
                    elif "comparison" in result:
                        # Alternative format: LLM used "comparison" instead of "brief_comparison"
                        brief_comparison = str(result.get("comparison", "")).strip()
                    elif "analysis" in result:
                        # Alternative format: LLM used "analysis" instead of "brief_comparison"
                        brief_comparison = str(result.get("analysis", "")).strip()
                    elif "shared_taxonomy_position" in result or "key_differences" in result:
                        # Structured format: combine fields into brief_comparison
                        parts = []
                        if result.get("shared_taxonomy_position"):
                            parts.append(str(result["shared_taxonomy_position"]))
                        if result.get("overlapping_areas"):
                            parts.append(str(result["overlapping_areas"]))
                        if result.get("key_differences"):
                            parts.append(str(result["key_differences"]))
                        brief_comparison = " ".join(parts).strip()
                    elif all(key in result for key in ["similarities", "differences", "novelty_summary"]):
                        # Legacy format (backward compatibility)
                        brief_comparison = f"{result['differences']} {result['novelty_summary']}".strip()
                    else:
                        # Invalid format - trigger retry
                        raise ValueError(f"LLM returned invalid format: {result}")
                else:
                    # Not a dict - trigger retry
                    raise ValueError(f"LLM returned non-dict result: {type(result)}")
                
                # ==================== Build Final Payload ====================
                # If we reached here, parsing was successful
                payload = {
                    "is_duplicate_variant": is_duplicate,
                    "brief_comparison": brief_comparison,
                    # For backward compatibility, also populate legacy fields
                    "similarities": "",
                    "differences": brief_comparison if not is_duplicate else "",
                }
                parsing_succeeded = True
                self.logger.info(f"Successfully parsed core task distinction response")
                
            except Exception as parse_err:
                parse_retry_count += 1
                if parse_retry_count < max_parse_retries:
                    self.logger.warning(
                        f"Failed to parse core task distinction response "
                        f"(attempt {parse_retry_count}/{max_parse_retries}): {parse_err}. Retrying..."
                    )
                else:
                    self.logger.error(
                        f"Failed to parse core task distinction response "
                        f"after {max_parse_retries} attempts: {parse_err}. Returning None."
                    )
        
        # ==================== Task 2: Textual Similarity Detection ====================
        # NOTE: Task 2 has been moved to the independent module run_textual_similarity.py
        # This ensures each paper is only analyzed once (by canonical_id) and results
        # are loaded from phase3/textual_similarity_detection/results.json
        
        return payload
    
    def _build_refutation_system_prompt(self) -> str:
        """Build the system prompt for Task 1: Contribution Refutation Assessment."""
        return (
            "You are a meticulous comparative reviewer for research papers. "
            "Your task is to determine whether a CANDIDATE paper refutes the novelty claims of an ORIGINAL paper.\n\n"
            "**CRITICAL: CITATION REFERENCE USAGE**\n"
            "When a citation reference (e.g., AgentGym-RL[1], Pilotrl[2]) is provided in the candidate paper title, "
            "you MUST use it throughout your analysis instead of phrases like 'The candidate paper'.\n\n"
            "**TASK: Contribution Refutation Assessment**\n"
            "For each contribution from the ORIGINAL paper, determine whether the CANDIDATE paper "
            "can refute the author's novelty claim (i.e., prove that the author was NOT the first "
            "to propose this contribution).\n\n"
            "**REFUTATION STATUS DETERMINATION**:\n"
            "1. **can_refute**: The candidate demonstrates that similar prior work exists.\n"
            "   - Provide detailed `refutation_evidence` with summary (3-5 sentences) and evidence_pairs.\n"
            "   - Evidence pairs should show specific quotes from both papers that support this refutation.\n"
            "   - If you observe that large portions of text are nearly identical, you should set this status.\n"
            "2. **cannot_refute**: The candidate does NOT challenge the novelty.\n"
            "   - Provide ONLY a `brief_note` (1-2 sentences) explaining technical differences.\n"
            "   - Do NOT repeat the original paper's content. Be concise (e.g., 'This candidate focuses on web navigation tasks, not general RL frameworks.').\n"
            "3. **unclear**: Comparison is difficult due to lack of detail.\n\n"
            "**CRITICAL: JSON OUTPUT FORMAT**\n"
            "The user prompt may contain citation markers like [63] or [57] (e.g., 'VideoLLM[63]'). "
            "These are NOT JSON arrays - they are just text references to papers. "
            "You MUST output the COMPLETE JSON object as specified below, NOT just a number in brackets. "
            "Your entire response must be a valid JSON object starting with '{' and ending with '}'.\n\n"
            "**CRITICAL CONSTRAINTS**:\n"
            "- Do NOT create artificial refutations by focusing on minor differences.\n"
            "- Evidence quotes MUST be verbatim excerpts from the **Full Text Context** sections (≤ 90 words per quote).\n"
            "- For each quote, provide a `paragraph_label` (e.g., 'Abstract', 'Introduction, Para 2', 'Methodology').\n"
            "- Every quote MUST be found word-for-word in the provided context.\n\n"
            "Output EXACTLY one JSON object:\n"
            "{\n"
            '  "contribution_analyses": [\n'
            "    {\n"
            '      "aspect": "contribution",\n'
            '      "contribution_name": "...",\n'
            '      "refutation_status": "can_refute" | "cannot_refute" | "unclear",\n'
            '      "refutation_evidence": {\n'
            '        "summary": "3-5 sentences explaining HOW the candidate demonstrates prior work exists.",\n'
            '        "evidence_pairs": [\n'
            "          {\n"
            '            "original_quote": "...",\n'
            '            "original_paragraph_label": "...",\n'
            '            "candidate_quote": "...",\n'
            '            "candidate_paragraph_label": "...",\n'
            '            "rationale": "Explain how this pair supports refutation."\n'
            "          }\n"
            "        ]\n"
            '      },\n'
            '      "brief_note": "1-2 sentences explaining why novelty is not challenged."\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

    def _build_unified_similarity_prompt(
        self,
        orig_text_normalized: str,
        cand_text_normalized: str,
        candidate_title: str
    ) -> str:
        """
        Build unified similarity detection prompt with XML-wrapped data.
        Combines role definition, constraints, and data in a single user message.
        
        This new version:
        - Uses a single user message instead of system+user split
        - Wraps paper texts in XML tags for clarity
        - Includes stricter constraints (30-word threshold)
        - Emphasizes "Ctrl+F-like" verification
        - Supports LaTeX formula handling with [FORMULA] replacement
        
        Note: Uses string concatenation instead of f-string to safely handle
        any special characters (including triple quotes) in paper texts.
        """
        prompt_parts = [
            "# Role\n\n",
            "You are an extremely rigorous academic plagiarism detection system, focused on **verbatim, character-level** text comparison. ",
            "Your core capability is to precisely identify plagiarism segments between two input texts and extract them **without any alteration**.\n\n",

            "# Task\n\n",
            "Compare the contents of **[Paper A]** and **[Paper B]**, ",
            "and identify all paragraphs where the **continuous text overlap is very high**.\n\n",

            "# Input Format\n\n",
            "The user will provide two text sections, labeled as `<Paper_A>` and `<Paper_B>`.\n\n",

            "# Strict Constraints (must be strictly followed)\n\n",
            "1. **Absolutely no rewriting**: The `original_text` and `candidate_text` fields in the output must be **exactly identical** ",
            "to the input texts, including punctuation and whitespace. ",
            "It is strictly forbidden to summarize, polish, reorder, or replace words with synonyms.\n\n",

            "2. **No hallucination**: Before outputting any segment, you must perform an internal \"Ctrl+F-like\" search. ",
            "Every extracted segment must occur verbatim in the provided inputs. ",
            "If the segment does not exist, you must not output it under any circumstances.\n\n",

            "3. **Length threshold**: Only report segments where the number of **consecutive words is ≥ 30**. ",
            "Ignore short phrases or coincidental overlaps of common terms.\n\n",

            "4. **Plagiarism gate: Report a segment only when it meets both\n",
            "   (i) **Verbatim overlap**: ≥30 consecutive-word verbatim overlap (after formula normalization) , The unique wording pattern is substantially the same as the source.\n",
            "   (ii) **Strict semantic/structural equivalence**: same claim, same technical entities, and same logical direction.\n\n",

            "5. **Prefer omission over error**: If no overlapping segments that meet the criteria are found, return an empty list directly. ",
            "Do not fabricate or force results.\n\n",

            "# Definition of Plagiarism\n\n",
            "- **Direct Plagiarism**: Copying text word-for-word from a source without quotation marks and citation.\n\n",
            "- **Paraphrasing Plagiarism**: Rewording a source's content without changing its core meaning or providing attribution.\n\n",

            "# Output Format\n\n",
            "Output only a valid JSON object, with no Markdown code block markers (such as ```json), ",
            "and no opening or closing remarks.\n\n",
            "The JSON structure is as follows:\n\n",
            "**When plagiarism segments are found:**\n\n",
            "{\n",
            '  "plagiarism_segments": [\n',
            "    {\n",
            '      "segment_id": 1,\n',
            '      "location": "Use an explicit section heading if it appears in the provided text near the segment (e.g., Abstract/Introduction/Method/Experiments). If no explicit heading is present, output \\"unknown\\".",\n',
            '      "original_text": "This must be an exact quote that truly exists in Paper A, no modifications allowed...",\n',
            '      "candidate_text": "This must be an exact quote that truly exists in Paper B, no modifications allowed...",\n',
            '      "plagiarism_type": "Direct/Paraphrase",\n',
            '      "rationale": "Brief explanation (1-2 sentences) of why these texts are plagiarism."\n',
            "    }\n",
            "  ]\n",
            "}\n\n",
            "**When NO plagiarism is detected:**\n\n",
            "{\n",
            '  "plagiarism_segments": []\n',
            "}\n\n",
            "**CRITICAL**: Always return the JSON object with the 'plagiarism_segments' key. ",
            "Do NOT return a bare empty array [] - always wrap it in the object structure shown above.\n\n",

            "# Now, please process the following inputs:\n\n",
            "<Paper_A>\n",
            orig_text_normalized,
            "\n</Paper_A>\n\n",
            "<Paper_B>\n",
            cand_text_normalized,
            "\n</Paper_B>"
        ]

        
        return "".join(prompt_parts)

    def _build_refutation_user_prompt(
        self,
        candidate: RetrievedPaper,
        num_contributions: int,
        contributions_text: str,
        orig_text_normalized: str,
        cand_text_normalized: str,
        candidate_citation: Optional[str] = None,
    ) -> str:
        """Build the user prompt for Task 1: Refutation Assessment."""
        citation_info = f" ({candidate_citation})" if candidate_citation else ""
        return (
            f"**Candidate Paper Title**: {candidate.title}{citation_info}\n\n"
            f"**Number of Contributions to Compare**: {num_contributions}\n\n"
            "**[Contributions to Compare]**\n"
            f"{contributions_text}\n\n"
            "**[Full Text Context: ORIGINAL]** (extract evidence from here)\n"
            f"```\n{orig_text_normalized}\n```\n\n"
            "**[Full Text Context: CANDIDATE]** (extract evidence from here)\n"
            f"```\n{cand_text_normalized}\n```\n\n"
            "**CRITICAL RULE**: You MUST extract quotes EXACTLY as they appear above. "
            "Copy character-by-character, including punctuation and spacing. "
            "If you cannot find a quote word-for-word in the context, do NOT use it. "
            "Evidence MUST be extracted from 'Full Text Context', NOT from Contribution Descriptions."
        )

    def _parse_contribution_analyses(
        self,
        parsed: dict,
        original_contributions: List[Any],
        contribution_tuples: List[Tuple[ASPECT, str, str]],
        orig_text_norm: Optional[str] = None,
        cand_text_norm: Optional[str] = None,
    ) -> List[ContributionAnalysis]:
        """
        Parse contribution analyses from LLM response.
        """
        contribution_analyses = []
        analyses_list = parsed.get("contribution_analyses", [])
        
        # Build contribution map for lookup
        contribution_map = {}
        for contribution in original_contributions:
            key = (getattr(contribution, 'aspect', 'contribution'), contribution.name)
            contribution_map[key] = contribution
        
        for analysis_data in analyses_list:
            if not isinstance(analysis_data, dict):
                continue
            
            aspect = analysis_data.get("aspect")
            contribution_name = analysis_data.get("contribution_name")
            
            if not aspect or not contribution_name:
                self.logger.warning(
                    f"Batch analysis missing aspect or contribution_name: {analysis_data}"
                )
                continue
            
            # Find corresponding contribution to get description
            contribution_desc = ""
            contribution_obj = contribution_map.get((aspect, contribution_name))
            if contribution_obj:
                contribution_desc = contribution_obj.description or ""
            
            # Check if new refutation-based format is present
            refutation_status = analysis_data.get("refutation_status", "")
            
            if refutation_status:
                # New refutation-based format
                refutation_evidence = None
                brief_note = ""
                
                if refutation_status == "can_refute":
                    # Extract refutation evidence
                    refutation_data = analysis_data.get("refutation_evidence", {})
                    if isinstance(refutation_data, dict):
                        summary = (refutation_data.get("summary") or "").strip()
                        evidence_pairs = self._parse_evidence_pairs(
                            refutation_data.get("evidence_pairs", []),
                            orig_text_norm=orig_text_norm,
                            cand_text_norm=cand_text_norm
                        )
                        refutation_evidence = {
                            "summary": summary,
                            "evidence_pairs": evidence_pairs
                        }
                elif refutation_status in ("cannot_refute", "unclear"):
                    # Extract brief note
                    brief_note = (analysis_data.get("brief_note") or "").strip()
                
                contribution_analysis = ContributionAnalysis(
                    aspect=aspect,  # type: ignore
                    contribution_name=contribution_name,
                    contribution_description=contribution_desc,
                    refutation_status=refutation_status,
                    refutation_evidence=refutation_evidence,
                    brief_note=brief_note,
                )
            else:
                # Legacy format (backward compatibility)
                novelty_data = analysis_data.get("novelty_analysis", {})
                
                novelty_evidence = self._parse_evidence_pairs(
                    novelty_data.get("evidence_pairs", []),
                    orig_text_norm=orig_text_norm,
                    cand_text_norm=cand_text_norm
                )
                
                # Try to infer refutation_status from legacy data
                # If novelty_evidence exists, likely can_refute; otherwise unclear
                inferred_status = "can_refute" if novelty_evidence else "unclear"
                
                contribution_analysis = ContributionAnalysis(
                    aspect=aspect,  # type: ignore
                    contribution_name=contribution_name,
                    contribution_description=contribution_desc,
                    refutation_status=inferred_status,
                    refutation_evidence={
                        "summary": (novelty_data.get("summary") or "").strip(),
                        "evidence_pairs": novelty_evidence
                    } if novelty_evidence else None,
                    brief_note="",  # Legacy format doesn't have brief_note
                )
            
            contribution_analyses.append(contribution_analysis)
        
        return contribution_analyses
    
    def _parse_similarity_segments(
        self,
        parsed: dict,
        candidate: RetrievedPaper,
        orig_text_norm: Optional[str] = None,
        cand_text_norm: Optional[str] = None,
    ) -> List[TextualSimilaritySegment]:
        """
        Parse plagiarism/similarity segments from LLM response and validate them.
        
        Supports both new format (plagiarism_segments) and legacy format (similarity_segments)
        for backward compatibility.
        """
        similarity_segments: List[TextualSimilaritySegment] = []
        segments_list = parsed.get("plagiarism_segments") or parsed.get("similarity_segments", [])
        self._log_similarity_list(candidate, segments_list)

        for segment_data in segments_list:
            if not isinstance(segment_data, dict):
                continue

            try:
                texts = self._extract_segment_texts(segment_data, candidate)
                if not texts:
                    continue
                orig_text, cand_text = texts

                verification = self._verify_segment_sources(
                    candidate, orig_text, cand_text, orig_text_norm, cand_text_norm
                )
                if not verification:
                    continue
                found_orig, conf_orig, found_cand, conf_cand = verification

                word_counts = self._calculate_word_counts(segment_data, orig_text, cand_text, candidate)
                if not word_counts:
                    continue
                word_count, word_count_orig, word_count_cand, llm_word_count = word_counts

                locations = self._extract_locations(segment_data)
                orig_location, cand_location = self._build_locations(locations, found_orig, conf_orig, found_cand, conf_cand)

                segment_id = segment_data.get("segment_id")
                plagiarism_type = segment_data.get("plagiarism_type", "Direct")
                self._log_segment_check(candidate, word_count, word_count_orig, word_count_cand, orig_text, cand_text, found_orig, found_cand, plagiarism_type)

                similarity_segment = TextualSimilaritySegment(
                    original_text=orig_text[:500],
                    candidate_text=cand_text[:500],
                    word_count=word_count,
                    original_location=orig_location,
                    candidate_location=cand_location,
                    rationale=(segment_data.get("rationale") or "").strip(),
                    segment_id=segment_id,
                    plagiarism_type=plagiarism_type
                )

                if word_count >= MIN_SIMILARITY_WORDS:
                    similarity_segments.append(similarity_segment)
                    self.logger.info(
                        f"[Plagiarism] Candidate {candidate.paper_id}: "
                        f"Segment ACCEPTED - words={word_count}, type={plagiarism_type}"
                    )
                else:
                    self.logger.info(
                        f"[Plagiarism] Candidate {candidate.paper_id}: "
                        f"Segment REJECTED - word_count {word_count} < {MIN_SIMILARITY_WORDS}"
                    )
            except (ValueError, TypeError) as e:
                self.logger.warning(
                    f"[Plagiarism] Candidate {candidate.paper_id}: "
                    f"Failed to parse segment: {e}. "
                    f"Segment data keys: {list(segment_data.keys()) if isinstance(segment_data, dict) else 'N/A'}"
                )
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                continue
            except Exception as e:
                self.logger.error(
                    f"[Plagiarism] Candidate {candidate.paper_id}: "
                    f"Unexpected error parsing segment: {e}"
                )
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                continue
        
        # If multiple segments, keep only top 3 by word count (as representative evidence)
        if len(similarity_segments) > MAX_SIMILARITY_SEGMENTS:
            similarity_segments.sort(key=lambda s: s.word_count, reverse=True)
            similarity_segments = similarity_segments[:MAX_SIMILARITY_SEGMENTS]
            self.logger.info(
                f"Found {len(segments_list)} segments, keeping top {MAX_SIMILARITY_SEGMENTS} (by word count) as evidence"
            )
        
        if similarity_segments:
            self.logger.info(
                f"[Plagiarism] Candidate {candidate.paper_id}: "
                f"[Plagiarism] Candidate {candidate.paper_id}: Final result: {len(similarity_segments)} segments ACCEPTED "
                f"(out of {len(segments_list)} reported by LLM)"
            )
        else:
            if segments_list:
                self.logger.info(
                    f"[Plagiarism] Candidate {candidate.paper_id}: "
                f"[Plagiarism] Candidate {candidate.paper_id}: Final result: 0 segments ACCEPTED "
                    f"(LLM reported {len(segments_list)} segments, but all were filtered out)"
                )
            else:
                self.logger.debug(
                    f"[Plagiarism] Candidate {candidate.paper_id}: "
                    f"No segments reported by LLM"
                )
        
        return similarity_segments

    def _log_similarity_list(self, candidate: RetrievedPaper, segments_list: List[Any]) -> None:
        """Log initial similarity segment list status."""
        if segments_list:
            self.logger.info(
                f"[Plagiarism] Candidate {candidate.paper_id}: "
                f"LLM identified {len(segments_list)} segments, starting validation..."
            )
        else:
            self.logger.debug(
                f"[Plagiarism] Candidate {candidate.paper_id}: No segments reported by LLM"
            )

    def _extract_segment_texts(
        self,
        segment_data: Dict[str, Any],
        candidate: RetrievedPaper,
    ) -> Optional[Tuple[str, str]]:
        """Extract and validate original/candidate text from a segment payload."""
        orig_text = (segment_data.get("original_text") or "").strip()
        cand_text = (segment_data.get("candidate_text") or "").strip()

        if not orig_text or not cand_text:
            self.logger.debug(
                f"Skipping segment: empty text "
                f"(orig_len={len(orig_text)}, cand_len={len(cand_text)}) for candidate {candidate.paper_id}"
            )
            return None
        return orig_text, cand_text

    def _verify_segment_sources(
        self,
        candidate: RetrievedPaper,
        orig_text: str,
        cand_text: str,
        orig_text_norm: Optional[str],
        cand_text_norm: Optional[str],
    ) -> Optional[Tuple[bool, float, bool, float]]:
        """
        Normalize and verify that segment texts exist in the provided normalized full texts.
        Returns (found_orig, conf_orig, found_cand, conf_cand) or None if rejected.
        """
        if not orig_text_norm or not cand_text_norm:
            # If no normalized texts are provided, skip strict verification.
            return True, 0.0, True, 0.0

        orig_text_for_verify = self.evidence_verifier.normalize_text(orig_text)
        cand_text_for_verify = self.evidence_verifier.normalize_text(cand_text)

        found_orig, conf_orig = self.evidence_verifier.verify_quote_in_fulltext(
            orig_text_for_verify, orig_text_norm
        )
        found_cand, conf_cand = self.evidence_verifier.verify_quote_in_fulltext(
            cand_text_for_verify, cand_text_norm
        )

        if not found_orig or not found_cand:
            # Fallback: substring match as last resort
            if orig_text_for_verify in orig_text_norm and cand_text_for_verify in cand_text_norm:
                found_orig = found_cand = True
                self.logger.info(
                    f"[Plagiarism] Candidate {candidate.paper_id}: Substring match succeeded (fallback)"
                )

        if not found_orig or not found_cand:
            self.logger.warning(
                f"[Plagiarism] Candidate {candidate.paper_id}: "
                f"REJECTED - segment not found in source papers. "
                f"Original: {found_orig} ({conf_orig:.2f}), Candidate: {found_cand} ({conf_cand:.2f})"
            )
            return None

        return found_orig, conf_orig, found_cand, conf_cand

    def _calculate_word_counts(
        self,
        segment_data: Dict[str, Any],
        orig_text: str,
        cand_text: str,
        candidate: RetrievedPaper,
    ) -> Optional[Tuple[int, int, int, Any]]:
        """Calculate word counts and enforce minimum similarity word threshold."""
        word_count_orig = len(orig_text.split()) if orig_text else 0
        word_count_cand = len(cand_text.split()) if cand_text else 0
        word_count = min(word_count_orig, word_count_cand)
        llm_word_count = segment_data.get("word_count", 0)

        if word_count < MIN_SIMILARITY_WORDS:
            self.logger.info(
                f"[Plagiarism] Candidate {candidate.paper_id}: "
                f"Skipping segment - word_count={word_count} "
                f"(orig={word_count_orig}, cand={word_count_cand}, LLM_reported={llm_word_count}) < {MIN_SIMILARITY_WORDS}"
            )
            return None

        return word_count, word_count_orig, word_count_cand, llm_word_count

    def _extract_locations(self, segment_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Parse location fields from segment data, handling both new (location) and legacy (original/candidate) formats.
        Returns (orig_location_str, cand_location_str).
        """
        if "location" in segment_data:
            location_str = segment_data.get("location", "").strip()
            return location_str, location_str

        orig_location_str = (segment_data.get("original_location") or "").strip()
        cand_location_str = (segment_data.get("candidate_location") or "").strip()
        return orig_location_str, cand_location_str

    def _build_locations(
        self,
        locations: Tuple[str, str],
        found_orig: bool,
        conf_orig: float,
        found_cand: bool,
        conf_cand: float,
    ) -> Tuple[EvidenceLocation, EvidenceLocation]:
        """Construct EvidenceLocation objects for original and candidate papers."""
        orig_location_str, cand_location_str = locations
        orig_location = EvidenceLocation(
            paragraph_label=orig_location_str,
            found=found_orig,
            match_score=conf_orig
        )
        cand_location = EvidenceLocation(
            paragraph_label=cand_location_str,
            found=found_cand,
            match_score=conf_cand
        )
        return orig_location, cand_location

    def _log_segment_check(
        self,
        candidate: RetrievedPaper,
        word_count: int,
        word_count_orig: int,
        word_count_cand: int,
        orig_text: str,
        cand_text: str,
        found_orig: bool,
        found_cand: bool,
        plagiarism_type: str,
    ) -> None:
        """Log detailed segment check info."""
        self.logger.info(
            f"[Plagiarism] Candidate {candidate.paper_id}: "
            f"Segment check - word_count={word_count} "
            f"(orig={word_count_orig}, cand={word_count_cand}), "
            f"orig_len={len(orig_text)} chars, cand_len={len(cand_text)} chars, "
            f"orig_found={found_orig}, cand_found={found_cand}, "
            f"type={plagiarism_type}"
        )
    
    def _parse_evidence_pairs(
        self, 
        evidence_data: List[Dict],
        orig_text_norm: Optional[str] = None,
        cand_text_norm: Optional[str] = None,
    ) -> List[EvidencePair]:
        """Parse evidence pairs and verify them against normalized full texts."""
        evidence_pairs: List[EvidencePair] = []
        for ep in evidence_data:
            orig_quote = (ep.get("original_quote") or "").strip()
            cand_quote = (ep.get("candidate_quote") or "").strip()
            
            orig_loc = EvidenceLocation(paragraph_label=ep.get("original_paragraph_label"))
            cand_loc = EvidenceLocation(paragraph_label=ep.get("candidate_paragraph_label"))
            
            # Verify quotes if text is provided
            if orig_text_norm and orig_quote:
                found, conf = self.evidence_verifier.verify_quote_in_fulltext(orig_quote, orig_text_norm)
                orig_loc.found = found
                orig_loc.confidence = conf
            
            if cand_text_norm and cand_quote:
                found, conf = self.evidence_verifier.verify_quote_in_fulltext(cand_quote, cand_text_norm)
                cand_loc.found = found
                cand_loc.confidence = conf

            evidence_pairs.append(
                EvidencePair(
                    original_quote=orig_quote,
                    candidate_quote=cand_quote,
                    rationale=(ep.get("rationale") or "").strip(),
                    original_location=orig_loc,
                    candidate_location=cand_loc,
                )
            )
        return evidence_pairs
    
    def _build_core_task_distinction_system_prompt(
        self,
        core_task_text: str,
        taxonomy_context_text: str,
        relationship: str = "sibling",
    ) -> str:
        """
        Build the system prompt for core task distinction analysis.
        
        Args:
            core_task_text: Core task description
            taxonomy_context_text: Formatted taxonomy context
            relationship: currently only "sibling" is supported
        """
        system_prompt = (
            "**CRITICAL: You MUST respond with valid JSON only. No markdown, no code fences, no explanations outside JSON.**\n\n"
            "**CRITICAL: JSON OUTPUT FORMAT**\n"
            "The user prompt may contain citation markers like [63] or [57] (e.g., 'VideoLLM[63]'). "
            "These are NOT JSON arrays - they are just text references to papers. "
            "You MUST output the COMPLETE JSON object as specified below, NOT just a number in brackets. "
            "Your entire response must be a valid JSON object starting with '{' and ending with '}'.\n\n"
            "You are an expert in assessing research novelty within a specific research domain.\n\n"
            "Your task is to compare the ORIGINAL paper against a CANDIDATE paper within the context "
            "of the core_task domain, to assess the ORIGINAL paper's novelty.\n\n"
            "Context Information:\n"
            f"- Core Task Domain: {core_task_text}\n"
        )
        
        if taxonomy_context_text:
            system_prompt += (
                "- Taxonomy Structure: The domain has been organized into a hierarchical taxonomy.\n"
                f"{taxonomy_context_text}\n"
            )
        
        # Simplified comparison for siblings with duplicate detection
            system_prompt += (
                "**RELATIONSHIP**: These papers are in the SAME taxonomy category (sibling papers).\n"
                "They address very similar aspects of the core_task.\n\n"
                "**CRITICAL: DUPLICATE DETECTION**\n"
                "First, determine if these papers are likely the same paper or different versions/variants:\n"
                "- Check if the titles are extremely similar or identical\n"
                "- Check if the abstracts describe essentially the same system/method/contribution\n"
                "- Check if the core technical content and approach are nearly identical\n\n"
                "If you determine they are likely duplicates/variants, output:\n"
                "{\n"
                '  "is_duplicate_variant": true,\n'
                '  "brief_comparison": "This paper is highly similar to the original paper; it may be a variant or near-duplicate. Please manually verify."\n'
                "}\n\n"
                "If they are clearly different papers (despite being in the same category), provide a CONCISE comparison (2-3 sentences):\n"
                "{\n"
                '  "is_duplicate_variant": false,\n'
                '  "brief_comparison": "2-3 sentences covering: (1) how they belong to the same parent category, '
                '(2) overlapping areas with the original paper regarding the core_task, '
                '(3) key differences from the original paper."\n'
                "}\n\n"
                "**Requirements for brief_comparison (when is_duplicate_variant=false)**:\n"
                "- Sentence 1: Explain how both papers belong to the same taxonomy category (shared focus/approach)\n"
                "- Sentence 2-3: Describe the overlapping areas and the key differences from the original paper\n"
                "- Be CONCISE: 2-3 sentences ONLY\n"
                "- Focus on: shared parent category, overlapping areas, and key distinctions\n"
                "- Do NOT include quotes or detailed evidence\n"
                "- Do NOT repeat extensive details from the original paper\n\n"
                "**Output format (STRICT JSON only, no markdown, no code fences, no extra text)**:\n"
                "{\n"
                '  "is_duplicate_variant": true/false,\n'
                '  "brief_comparison": "2-3 sentences as described above"\n'
                "}\n\n"
                "IMPORTANT: Your ENTIRE response must be valid JSON starting with { and ending with }. "
                "Do NOT include any text before or after the JSON object."
            )
        
        return system_prompt
    
    def _extract_taxonomy_context(
        self,
        taxonomy: Dict[str, Any],
        original_paper_id: str,
        candidate_paper_id: str,
        mapping: List[Dict[str, Any]],  # List of {"paper_id": str, "taxonomy_path": List[str]}
    ) -> Optional[Dict[str, Any]]:
        """
        Extract taxonomy context information for original and candidate papers.
        
        Args:
            taxonomy: Taxonomy tree structure
            original_paper_id: Original paper ID
            candidate_paper_id: Candidate paper ID
            mapping: List of paper_id to taxonomy_path mappings from survey report
            
        Returns:
            Dictionary with taxonomy context info, or None if not found
        """
        try:
            # Build paper_id -> taxonomy_path mapping
            # NOTE: mapping uses "canonical_id" as key, not "paper_id"
            paper_to_path: Dict[str, List[str]] = {}
            for item in mapping:
                # Try both canonical_id and paper_id for compatibility
                pid = item.get("canonical_id") or item.get("paper_id")
                path = item.get("taxonomy_path", [])
                if pid and path:
                    paper_to_path[pid] = path
            
            original_path = paper_to_path.get(original_paper_id, [])
            candidate_path = paper_to_path.get(candidate_paper_id, [])
            
            if not original_path and not candidate_path:
                return None
            
            # Find common ancestor path
            common_prefix_len = 0
            min_len = min(len(original_path), len(candidate_path))
            for i in range(min_len):
                if original_path[i] == candidate_path[i]:
                    common_prefix_len = i + 1
                else:
                    break
            
            # Recursively find leaf node information
            def find_leaf_node(taxo_node: Dict[str, Any], target_path: List[str], current_path: List[str]) -> Optional[Dict[str, Any]]:
                if not target_path:
                    return taxo_node
                
                if len(current_path) >= len(target_path):
                    # Reached target depth, check if it's a leaf node
                    if "papers" in taxo_node:
                        return taxo_node
                    return None
                
                next_name = target_path[len(current_path)]
                subtopics = taxo_node.get("subtopics", [])
                for st in subtopics:
                    if st.get("name") == next_name:
                        return find_leaf_node(st, target_path, current_path + [next_name])
                return None
            
            original_leaf = None
            candidate_leaf = None
            
            if original_path:
                original_leaf = find_leaf_node(taxonomy, original_path, [taxonomy.get("name", "")])
            if candidate_path:
                candidate_leaf = find_leaf_node(taxonomy, candidate_path, [taxonomy.get("name", "")])
            
            # Determine if both papers are in the same leaf node
            same_leaf = False
            if original_path and candidate_path:
                # If paths are identical, they are in the same leaf node
                same_leaf = original_path == candidate_path
            elif original_leaf and candidate_leaf:
                # Or compare leaf node names
                same_leaf = original_leaf.get("name") == candidate_leaf.get("name")
            
            # Extract other papers in the same leaf node (up to 5)
            leaf_neighbors = []
            if same_leaf and original_leaf:
                papers_in_leaf = original_leaf.get("papers", [])
                for pid in papers_in_leaf:
                    if pid not in [original_paper_id, candidate_paper_id]:
                        leaf_neighbors.append(pid)
                        if len(leaf_neighbors) >= 5:
                            break
            
            # Extract common ancestor information
            common_ancestor = None
            if common_prefix_len > 0:
                common_path = original_path[:common_prefix_len]
                current_node = taxonomy
                for name in common_path[1:]:  # Skip root name
                    subtopics = current_node.get("subtopics", [])
                    found = False
                    for st in subtopics:
                        if st.get("name") == name:
                            current_node = st
                            found = True
                            break
                    if not found:
                        break
                common_ancestor = current_node
            
            return {
                "original_path": original_path,
                "candidate_path": candidate_path,
                "common_ancestor": common_ancestor,
                "original_leaf": original_leaf,
                "candidate_leaf": candidate_leaf,
                "same_leaf": same_leaf,
                "leaf_neighbors": leaf_neighbors,
            }
        except Exception as e:
            self.logger.warning(f"Failed to extract taxonomy context: {e}")
            return None
    
    def _format_taxonomy_context(self, taxonomy_info: Dict[str, Any]) -> str:
        """Format taxonomy information as text for prompt inclusion."""
        if not taxonomy_info:
            return ""
        
        original_path = taxonomy_info.get("original_path", [])
        candidate_path = taxonomy_info.get("candidate_path", [])
        same_leaf = taxonomy_info.get("same_leaf", False)
        original_leaf = taxonomy_info.get("original_leaf", {})
        candidate_leaf = taxonomy_info.get("candidate_leaf", {})
        common_ancestor = taxonomy_info.get("common_ancestor", {})
        
        context_parts = []
        
        # Original paper position
        if original_path and original_leaf:
            context_parts.append(
                f"- ORIGINAL paper taxonomy path: {' > '.join(original_path)}\n"
                f"  Category: {original_leaf.get('name', '')}\n"
                f"  Scope: {original_leaf.get('scope_note', '')}"
            )
        
        # Candidate paper position
        if candidate_path and candidate_leaf:
            context_parts.append(
                f"- CANDIDATE paper taxonomy path: {' > '.join(candidate_path)}\n"
                f"  Category: {candidate_leaf.get('name', '')}\n"
                f"  Scope: {candidate_leaf.get('scope_note', '')}"
            )
        
        # Position relationship description
        if same_leaf:
            context_parts.append(
                "- Both papers are in the SAME taxonomy category. "
                "Focus on detailed distinctions within this category."
            )
        elif common_ancestor:
            context_parts.append(
                f"- Common ancestor category: {common_ancestor.get('name', '')}\n"
                "- Papers are in different sub-categories within this broader category. "
                "Highlight how they diverge in their approaches."
            )
        elif original_path or candidate_path:
            context_parts.append(
                "- Papers are in different taxonomy branches, indicating different research directions."
            )
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def _detect_formula_density(self, text: str) -> float:
        """
        Detect formula density (ratio of special math symbols).
        
        Args:
            text: Text to analyze
            
        Returns:
            Formula density ratio (0.0 to 1.0)
        """
        if not text:
            return 0.0
        special_chars = sum(1 for c in text if c in '∑∫∏αβγδλΣΠ∥≥≤∈⊆⊂∪∩∅^_\\$')
        return special_chars / max(len(text), 1)
    
    def _save_llm_response(
        self,
        raw_response: Any,
        file_prefix: str,
        candidate_id: Optional[str] = None,
        context: str = "contribution",
        extra_suffix: str = ""
    ) -> Optional[str]:
        """
        Save raw LLM response to file and return file path.
        
        Args:
            raw_response: Raw response from LLM
            file_prefix: Prefix for filename
            candidate_id: Optional candidate ID for filename
            context: Context subdirectory - "contribution", "core_task", or "survey" (default: "contribution")
            extra_suffix: Optional extra suffix for filename
            
        Returns:
            File path if successful, None otherwise
        """
        try:
            base_dir = Path(self.output_base_dir if self.output_base_dir else os.getcwd()).resolve()  # Use absolute path to avoid creating dirs in wrong location
            outdir = base_dir / "phase3" / "raw_llm_responses" / context
            outdir.mkdir(parents=True, exist_ok=True)
            safe_cid = sanitize_id(candidate_id or "unknown_candidate")
            ts = int(time.time())
            suffix = f"_{extra_suffix}" if extra_suffix else ""
            fname = outdir / f"{file_prefix}_{safe_cid}{suffix}_{ts}.json"
            with open(fname, "w", encoding="utf-8") as fh:
                try:
                    json_dump = raw_response if raw_response is not None else {"error": "no_response"}
                    json.dump(json_dump, fh, ensure_ascii=False, indent=2)
                except Exception:
                    fh.write(str(raw_response))
            self.logger.info(f"Saved raw LLM response to: {fname}")
            return fname
        except Exception as e:
            self.logger.warning(f"Failed to save raw LLM response: {e}")
            return None
