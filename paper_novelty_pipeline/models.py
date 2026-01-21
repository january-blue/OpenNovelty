"""
Data models for the paper novelty evaluation pipeline.
Defines all data structures used throughout the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal

ASPECT = Literal["task", "setting", "problem", "method", "contribution"]


@dataclass
class PaperInput:
    """Input data for a paper to be evaluated."""
    paper_id: str
    title: str
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    original_pdf_url: Optional[str] = None  # URL to the original PDF if available
    openreview_id: Optional[str] = None  # Short OpenReview ID (e.g., "7dvYWzOiEu")

    # OpenReview extension fields (flattened structure; all fields live at the top level)
    openreview_forum_id: Optional[str] = None  # Source: OpenReview forum ID
    keywords: Optional[List[str]] = None  # Source: OpenReview submission.keywords
    primary_area: Optional[str] = None  # Source: OpenReview submission.primary_area
    submission_number: Optional[str] = None  # Source: OpenReview submission.submission_number
    openreview_venueid: Optional[str] = None  # Source: OpenReview submission.venueid (ID form of venue)
    openreview_pdf_path: Optional[str] = None  # Source: OpenReview submission.pdf
    bibtex: Optional[str] = None  # Source: OpenReview submission._bibtex
    openreview_rating_mean: Optional[float] = None  # Source: mean rating across OpenReview reviews
    openreview_ratings: Optional[List[float]] = None  # Source: all ratings from OpenReview reviews


@dataclass
class ContributionSlot:
    """
    Represents a single contribution claim from the original paper.
    
    This is the new name for what was previously called "Subslot".
    Since we now only handle contributions (not task/setting/problem/method),
    the naming is more direct and clear.
    
    Fields:
    - aspect: Always "contribution" for current implementation
    - name: Short label for this contribution (e.g., "Residual Weight Adaptation")
    - description: 1-2 sentences describing the contribution
    - prior_work_query: Search query used in Phase2 to find related prior work
    - source_hint: Location in paper where this contribution was mentioned
    """

    aspect: ASPECT          # Always "contribution" in current implementation
    name: str               # Short, human-readable label for this contribution
    description: str        # 1-2 sentences describing this contribution
    prior_work_query: str   # Search query to find prior work
    source_hint: str = "unknown"  # Location hint (e.g., "Introduction", "Abstract")


@dataclass
class ContributionGroup:
    """
    Groups multiple ContributionSlots under a single aspect.
    
    This is the new name for what was previously called "Slot".
    Since we now only use aspect="contribution", this is mainly a container structure.
    
    Note: In current implementation, there's typically only ONE ContributionGroup
    with aspect="contribution" containing all contributions (1-3 items).
    """

    aspect: ASPECT  # Always "contribution" in current implementation
    contributions: List[ContributionSlot]


# Backward compatibility aliases (to be deprecated)
Subslot = ContributionSlot  # Deprecated: use ContributionSlot
Slot = ContributionGroup  # Deprecated: use ContributionGroup


@dataclass
class CoreTask:
    """
    One-sentence description of the core task/capability.
    Example: "general reasoning of vision-language models"
    """
    text: str = ""
    query_variants: List[str] = field(default_factory=list)
    """
    Query expansion list for Phase2 search. Following standard Information Retrieval
    practices, this list includes the original query (text) as variants[0], plus 2-3
    paraphrased variants for broader recall.
    
    Example:
        text = "neural machine translation quality estimation"
        query_variants = [
            "neural machine translation quality estimation",  # variants[0] = original
            "NMT quality prediction",                         # variants[1] = paraphrase
            "translation quality assessment"                  # variants[2] = paraphrase
        ]
    
    Note: Phase2 searches only query_variants (not text separately) to avoid duplication.
    """


@dataclass
class CoreTaskSelectedPaper:
    """
    A representative paper in the Core Task Survey.
    Phase1 only creates placeholders; actual content is filled by Phase2/3.
    """
    paper_id: str
    title: str
    venue: Optional[str] = None
    year: Optional[int] = None

    # canonical | closest_competitor | variant
    role: Optional[str] = None

    summary: str = ""
    comparison_to_target: str = ""


@dataclass
class CoreTaskSurvey:
    """
    A lightweight survey scaffold around the core_task, aligned with the design document.
    Phase1 only initializes it empty; Phase2/3 progressively fill it in.
    """
    query_variants: List[str] = field(default_factory=list)
    selected_papers: List[CoreTaskSelectedPaper] = field(default_factory=list)
    survey_text: str = ""
    per_paper_bullets: List[str] = field(default_factory=list)


@dataclass
class ContributionClaim:
    """
    An author-claimed "new contribution" at the claim level (method / data / finding / theory, etc.).
    """
    id: str
    name: str
    author_claim_text: str
    description: str = ""
    prior_work_query: str = ""
    query_variants: List[str] = field(default_factory=list)
    source_hint: str = "unknown"  # e.g. "Section 3.1", "Abstract", if missing then mark as "unknown" and show warning
    

@dataclass
class ExtractedContent:
    """
    Key paper information extracted in Phase1.
    """
    # One-sentence core_task + Core Task Survey scaffold
    core_task: CoreTask = field(default_factory=CoreTask)
    # Phase1 no longer outputs core_task_survey; Phase2 is responsible for generating and persisting it
    core_task_survey: Optional[CoreTaskSurvey] = None

    # Claim-level contribution list
    contributions: List[ContributionClaim] = field(default_factory=list)

    # OpenReview ID for tracking (format: "openreview:XXXXX")
    openreview_id: Optional[str] = None


@dataclass
class SearchQuery:
    """
    A search query generated from extracted content.
    
    In current implementation, this represents a query for finding papers
    related to a specific contribution from the original paper.
    
    Fields:
    - query: The actual search query text
    - aspect: Always "contribution" in current implementation
    - contribution_name: Name of the contribution this query is for
    - source_description: Description of where this query came from
    """
    query: str
    aspect: ASPECT  # Always "contribution" in current implementation
    contribution_name: str  # Name of the contribution (previously: subslot_name)
    source_description: str
    
    # Backward compatibility property
    @property
    def subslot_name(self) -> str:
        """Deprecated: use contribution_name instead."""
        return self.contribution_name 


@dataclass
class RetrievedPaper:
    """A paper retrieved from search."""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    venue: str
    year: int
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    relevance_score: float = 0.0
    # Optional original/raw fields preserved from the search backend
    pdf_url: Optional[str] = None
    source_url: Optional[str] = None
    raw_metadata: Optional[Dict[str, Any]] = None
    # Notes or diagnosis from Phase2 (e.g., why no pdf_url was found)
    note: Optional[str] = None
    # Canonical ID for global paper identification
    canonical_id: Optional[str] = None


@dataclass
class SearchResult:
    """Results from a search query."""
    query: SearchQuery
    retrieved_papers: List[RetrievedPaper]
    total_results: int

    # Number of verification events for this search query
    #verification_events: int = 0 # Abandoned since we are not using streaming, but keeping the field for future reference


@dataclass
class EvidenceLocation:
    """Verification details for an evidence quote."""
    paragraph_label: Optional[str] = None  # e.g., "Introduction, Para 1" (from LLM)
    found: bool = False                    # Verified locally?
    match_score: float = 0.0               # Fuzzy match score (0.0 - 1.0)


@dataclass
class EvidencePair:
    """A pair of aligned evidence snippets from Original and Candidate papers."""
    original_quote: str
    candidate_quote: str
    rationale: str
    original_location: EvidenceLocation = field(default_factory=EvidenceLocation)
    candidate_location: EvidenceLocation = field(default_factory=EvidenceLocation)


@dataclass
class ContributionAnalysis:
    """
    Analysis result for comparing a specific contribution between original and candidate papers.
    
    NEW STRUCTURE (refutation-based):
    - refutation_status: Whether the candidate can refute the author's novelty claim
    - If can_refute: refutation_evidence contains detailed evidence
    - If cannot_refute: brief_note contains 1-2 sentence explanation
    - If unclear: brief_note explains why unclear
    
    LEGACY STRUCTURE (backward compatibility):
    - novelty_summary/novelty_evidence: For can_refute cases
    - overlap_summary/overlap_evidence: Deprecated, kept for compatibility
    
    Fields:
    - aspect: Always "contribution" in current implementation
    - contribution_name: Name of the contribution being compared
    - contribution_description: Full description of the contribution
    - refutation_status: "can_refute" | "cannot_refute" | "unclear"
    - refutation_evidence: Optional detailed evidence (when can_refute)
      - summary: How candidate shows similar prior work exists
      - evidence_pairs: Specific evidence quotes
    - brief_note: Optional brief explanation (when cannot_refute or unclear)
    - novelty_summary: Legacy field (maps to refutation_evidence.summary when can_refute)
    - novelty_evidence: Legacy field (maps to refutation_evidence.evidence_pairs when can_refute)
    - overlap_summary: Deprecated, kept for backward compatibility
    - overlap_evidence: Deprecated, kept for backward compatibility
    """
    aspect: ASPECT  # Always "contribution" in current implementation
    contribution_name: str  # Name of the contribution (previously: subslot_name)
    contribution_description: str = ""  # Full description (previously: subslot_description)
    
    # New refutation-based fields
    refutation_status: str = ""  # "can_refute" | "cannot_refute" | "unclear"
    refutation_evidence: Optional[Dict[str, Any]] = None  # {summary: str, evidence_pairs: List[EvidencePair]}
    brief_note: str = ""  # 1-2 sentences when cannot_refute or unclear
    
    # Legacy fields (for backward compatibility)
    # DEPRECATED: These fields are no longer populated by new Phase 3 runs.
    # They are kept for reading old reports only. Use refutation_status/refutation_evidence instead.
    novelty_summary: str = ""          
    novelty_evidence: List[EvidencePair] = field(default_factory=list)
    overlap_summary: str = ""          
    overlap_evidence: List[EvidencePair] = field(default_factory=list)
    
    # Backward compatibility properties
    @property
    def subslot_name(self) -> str:
        """Deprecated: use contribution_name instead."""
        return self.contribution_name
    
    @property
    def subslot_description(self) -> str:
        """Deprecated: use contribution_description instead."""
        return self.contribution_description


# Backward compatibility alias (to be deprecated)
SubslotAnalysis = ContributionAnalysis  # Deprecated: use ContributionAnalysis

@dataclass
class TextualSimilaritySegment:
    """
    A segment of highly similar text / plagiarism between original and candidate papers.
    
    Updated to support plagiarism detection format with:
    - segment_id: Sequential ID for tracking
    - plagiarism_type: Classification as "Direct" or "Paraphrase"
    - Maintains backward compatibility with original_location and candidate_location
    """
    original_text: str                    # Text from original paper (up to 500 chars)
    candidate_text: str                   # Matching text from candidate paper (up to 500 chars)
    word_count: int                       # Word count of matched segment
    original_location: EvidenceLocation   # Location in original (structured, with verification)
    candidate_location: EvidenceLocation  # Location in candidate (structured, with verification)
    rationale: str = ""                   # LLM explanation of similarity/plagiarism (1-2 sentences)
    
    # New fields for plagiarism detection format
    segment_id: Optional[int] = None      # Sequential ID (1, 2, 3...)
    plagiarism_type: Optional[str] = None # "Direct" or "Paraphrase"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with unified location format for output.
        
        Returns dict matching professor's plagiarism detection format:
        - Single 'location' field (using original_location's label)
        - Includes segment_id and plagiarism_type
        - Keeps internal location details with _ prefix for debugging
        
        Returns:
            Dictionary in professor's format
        """
        # Use original_location's label as the unified location
        # (In most plagiarism cases, both papers have similar section structure)
        unified_location = self.original_location.paragraph_label if self.original_location else ""
        
        result = {
            "segment_id": self.segment_id or 1,
            "location": unified_location,
            "original_text": self.original_text,
            "candidate_text": self.candidate_text,
            "plagiarism_type": self.plagiarism_type or "Direct",
            "rationale": self.rationale
        }
        
        # Add internal debugging fields (prefixed with _)
        if self.original_location:
            result["_original_location_detail"] = {
                "label": self.original_location.paragraph_label,
                "found": self.original_location.found,
                "match_score": self.original_location.match_score
            }
        
        if self.candidate_location:
            result["_candidate_location_detail"] = {
                "label": self.candidate_location.paragraph_label,
                "found": self.candidate_location.found,
                "match_score": self.candidate_location.match_score
            }
        
        result["_word_count"] = self.word_count
        
        return result

@dataclass
class ComparisonResult:
    """Final result of comparing target paper with a retrieved paper."""
    # Metadata
    original_paper_id: str         # ID for tracking
    original_paper_url: Optional[str]  # User input URL (or local path)
    
    original_paper_title: str = ""
    original_paper_abstract: str = ""
    original_paper_authors: Optional[List[str]] = None
    original_paper_venue: str = ""
    original_paper_year: Optional[int] = None
    original_paper_keywords: Optional[List[str]] = None
    original_paper_primary_area: str = ""
    original_paper_openreview_rating_mean: Optional[float] = None
    
    candidate_paper_title: str = ""
    candidate_paper_url: Optional[str] = None
    retrieved_paper_id: str = ""
    # Candidate metadata (for richer downstream rendering)
    candidate_paper_abstract: Optional[str] = None
    candidate_paper_authors: Optional[List[str]] = None
    candidate_paper_venue: Optional[str] = None
    candidate_paper_year: Optional[int] = None
    
    # Processing Info
    comparison_mode: str = ""
    comparison_note: Optional[str] = None
    
    # Core Analysis
    analyzed_contributions: List[ContributionAnalysis] = field(default_factory=list)
    
    # Textual Similarity Detection (Phase 3 Enhancement)
    textual_similarity_segments: List[TextualSimilaritySegment] = field(default_factory=list)
    
    # Processing context
    query_source: Optional[str] = None  # The user query that triggered this search
    
    # Backward compatibility property
    @property
    def analyzed_slots(self) -> List[ContributionAnalysis]:
        """Deprecated: use analyzed_contributions instead."""
        return self.analyzed_contributions

@dataclass
class Phase3Report:
    """Complete Phase 3 evaluation report with statistics."""
    # Original paper metadata
    original_paper: Dict[str, Any]  
    
    # Comparison results
    comparisons: List[ComparisonResult] = field(default_factory=list)
    
    # Statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    generated_at: str = ""  # ISO timestamp
    pipeline_version: str = "1.0"


@dataclass
class PaperEvaluation:
    """Complete evaluation of a paper's novelty."""
    paper_id: str
    paper_title: str
    extracted_content: ExtractedContent
    search_results: List[SearchResult]
    comparison_results: List[ComparisonResult]
    
    overall_novelty_score: float = 0.0
    overall_assessment: str = ""
    top_concerns: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.top_concerns is None:
            self.top_concerns = []
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class UploadResult:
    """Result of uploading evaluation to website."""
    success: bool
    upload_id: Optional[str] = None
    message: str = ""
    response_data: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingStatus:
    """Status of pipeline processing for a paper."""
    paper_id: str
    phase: str  # 'extraction', 'searching', 'comparison', 'uploading', 'completed'
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    error_message: Optional[str] = None
    
    def duration_seconds(self) -> Optional[float]:
        """Calculate processing duration in seconds."""
        return None
