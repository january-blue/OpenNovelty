"""
Phase 3: Evidence Verifier

Handles evidence verification and textual similarity detection.
Verifies that evidence quotes from LLM actually exist in the source papers.
"""

import re
import difflib
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from paper_novelty_pipeline.models import ContributionAnalysis, EvidencePair
from paper_novelty_pipeline.config import (
    EVIDENCE_COVERAGE_WEIGHT,
    EVIDENCE_HIT_RATIO_WEIGHT,
    EVIDENCE_COMPACT_PENALTY,
    EVIDENCE_MIN_CONFIDENCE_THRESHOLD,
    EVIDENCE_PARTIAL_MATCH_THRESHOLD,
    EVIDENCE_CACHE_MAX_SIZE,
    EVIDENCE_CACHE_KEEP_SIZE,
    EVIDENCE_MIN_ANCHOR_CHARS,
    EVIDENCE_MIN_ANCHOR_COVERAGE,
    EVIDENCE_MAX_GAP_TOKENS,
    EVIDENCE_MIN_BLOCK_TOKENS,
)


# ============================================================================
# Constants
# ============================================================================

# Confidence calculation weights (configurable)
COVERAGE_WEIGHT = EVIDENCE_COVERAGE_WEIGHT
HIT_RATIO_WEIGHT = EVIDENCE_HIT_RATIO_WEIGHT
COMPACT_PENALTY = EVIDENCE_COMPACT_PENALTY
MIN_CONFIDENCE_THRESHOLD = EVIDENCE_MIN_CONFIDENCE_THRESHOLD
PARTIAL_MATCH_THRESHOLD = EVIDENCE_PARTIAL_MATCH_THRESHOLD

# Cache configuration (configurable)
CACHE_MAX_SIZE = EVIDENCE_CACHE_MAX_SIZE
CACHE_KEEP_SIZE = EVIDENCE_CACHE_KEEP_SIZE

# Anchor matching configuration (configurable)
MIN_ANCHOR_CHARS = EVIDENCE_MIN_ANCHOR_CHARS
MIN_ANCHOR_COVERAGE = EVIDENCE_MIN_ANCHOR_COVERAGE
MAX_GAP_TOKENS = EVIDENCE_MAX_GAP_TOKENS

# Segment similarity configuration (configurable)
MIN_BLOCK_TOKENS = EVIDENCE_MIN_BLOCK_TOKENS


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AnchorMatch:
    """Result of matching an anchor against full text."""
    coverage: float
    full_token_start: int
    full_token_end: int
    anchor_token_len: int


# ============================================================================
# Evidence Verifier Class
# ============================================================================

class EvidenceVerifier:
    """
    Verifies evidence pairs and detects textual similarity.
    
    Uses fuzzy matching to verify that quotes extracted by LLM actually
    exist in the source papers, handling PDF extraction artifacts and
    minor variations.
    """
    
    # Unicode character mappings for normalization
    # Maps Unicode variants to ASCII equivalents
    CHAR_REPLACEMENTS = {
        # Multiplication signs (various Unicode representations)
        '\u00d7': 'x',  # U+00D7 multiplication sign
        '\u2715': 'x',  # U+2715 multiplication x
        # Dashes and hyphens
        '\u2013': '-',  # En dash U+2013
        '\u2014': '-',  # Em dash U+2014
        '\u2212': '-',  # Minus sign U+2212
        # Ellipsis
        '\u2026': '...',  # Horizontal ellipsis
        # Smart quotes → plain quotes
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        # Degree sign
        '\u00b0': 'o',
    }
    
    def __init__(self, logger):
        """
        Initialize the evidence verifier.
        
        Args:
            logger: Logger instance for logging
        """
        self.logger = logger
        # Use OrderedDict for true FIFO cache behavior
        self._normalized_text_cache: OrderedDict[str, str] = OrderedDict()
    
    def normalize_text(self, text: str) -> str:
        """
        Enhanced text normalization for comparison.
        
        Handles Unicode variants, special characters, and common PDF extraction
        artifacts. This normalization is critical for accurate quote matching.
        
        Steps:
        1. Convert to lowercase
        2. Replace Unicode variants with ASCII equivalents
        3. Remove zero-width characters
        4. Merge hyphenation from PDF line breaks
        5. Normalize whitespace
        6. Trim
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Step 1: Lowercase
        text = text.lower()
        
        # Step 2: Replace Unicode variants using dictionary mapping
        for unicode_char, ascii_char in self.CHAR_REPLACEMENTS.items():
            text = text.replace(unicode_char, ascii_char)
        
        # PDF soft hyphen (U+00AD)
        text = text.replace('\u00ad', '')
        
        # Step 3: Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e]', '', text)
        
        # Step 4: Merge hyphenation from PDF line breaks
        # Example: "quantifi- able" -> "quantifiable"
        text = re.sub(r'(\w)-\s+(\w)', r"\1\2", text)
        
        # Step 5: Normalize whitespace (all types to regular space)
        text = re.sub(r'[\s\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+', ' ', text)
        
        # Step 6: Trim
        text = text.strip()
        
        return text
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for normalized text.
        
        Uses a combination of length and hash to create a reliable key
        that minimizes collisions while being efficient.
        
        Args:
            text: Text to generate key for
            
        Returns:
            Cache key string
        """
        # Use SHA256 for more reliable hashing (first 16 chars)
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        return f"{len(text)}_{text_hash}"
    
    def _get_normalized_text(self, text: str) -> str:
        """
        Get normalized text, using cache if available.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        cache_key = self._get_cache_key(text)
        
        # Check cache (move to end if found for LRU behavior)
        if cache_key in self._normalized_text_cache:
            # Move to end (most recently used)
            normalized = self._normalized_text_cache.pop(cache_key)
            self._normalized_text_cache[cache_key] = normalized
            return normalized
        
        # Normalize and cache
        normalized = self.normalize_text(text)
        self._normalized_text_cache[cache_key] = normalized
        
        # Enforce cache size limit (FIFO: remove oldest)
        if len(self._normalized_text_cache) > CACHE_MAX_SIZE:
            # Remove oldest entries (keep only CACHE_KEEP_SIZE most recent)
            while len(self._normalized_text_cache) > CACHE_KEEP_SIZE:
                self._normalized_text_cache.popitem(last=False)  # Remove oldest
        
        return normalized
    
    def verify_evidence(
        self,
        contribution_analysis: ContributionAnalysis,
        orig_full_text: str,
        cand_full_text: str,
    ) -> None:
        """
        Verify evidence pairs using enhanced fuzzy matching against full texts.
        
        Updates the EvidenceLocation objects in place with verification results.
        Uses caching to avoid re-normalizing the same text multiple times.
        
        Args:
            contribution_analysis: Contribution analysis containing evidence pairs
            orig_full_text: Original paper full text
            cand_full_text: Candidate paper full text
        """
        if not contribution_analysis:
            return
        
        # Get normalized texts (with caching)
        norm_orig = self._get_normalized_text(orig_full_text)
        norm_cand = self._get_normalized_text(cand_full_text)
        
        # Verify evidence pairs based on refutation_status
        if contribution_analysis.refutation_status == "can_refute" and contribution_analysis.refutation_evidence:
            # New refutation-based structure: verify refutation_evidence
            ref_ev = contribution_analysis.refutation_evidence
            evidence_pairs = ref_ev.get("evidence_pairs") or []
            self._verify_evidence_list(evidence_pairs, norm_orig, norm_cand)
        else:
            # Legacy structure or no evidence: verify novelty_evidence (for backward compatibility)
            self._verify_evidence_list(contribution_analysis.novelty_evidence, norm_orig, norm_cand)
        
        # Legacy: verify overlap_evidence if present (for backward compatibility)
        if contribution_analysis.overlap_evidence:
            self._verify_evidence_list(contribution_analysis.overlap_evidence, norm_orig, norm_cand)
    
    def _verify_evidence_list(
        self,
        pairs: List[EvidencePair],
        norm_orig: str,
        norm_cand: str,
    ) -> None:
        """
        Verify a list of evidence pairs.
        
        Args:
            pairs: List of evidence pairs to verify
            norm_orig: Normalized original paper text
            norm_cand: Normalized candidate paper text
        """
        for pair in pairs:
            # Verify Original Quote
            found, score = self.verify_quote_in_fulltext(
                pair.original_quote,
                norm_orig,
            )
            pair.original_location.found = found
            pair.original_location.match_score = score
            
            if not found and score > PARTIAL_MATCH_THRESHOLD:
                self.logger.debug(
                    f"[Evidence] Original quote partial match: score={score:.2f}, "
                    f"quote_preview={pair.original_quote[:80]}..."
                )
            
            # Verify Candidate Quote
            found, score = self.verify_quote_in_fulltext(
                pair.candidate_quote,
                norm_cand,
            )
            pair.candidate_location.found = found
            pair.candidate_location.match_score = score
            
            if not found and score > PARTIAL_MATCH_THRESHOLD:
                self.logger.debug(
                    f"[Evidence] Candidate quote partial match: score={score:.2f}, "
                    f"quote_preview={pair.candidate_quote[:80]}..."
                )
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity score between two texts using difflib.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize both texts
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        # Use difflib.SequenceMatcher to calculate similarity
        matcher = difflib.SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()
    
    def split_into_anchors(self, quote_norm: str, min_chars: int = MIN_ANCHOR_CHARS) -> List[str]:
        """
        Split a normalized quote into anchor fragments using ellipsis as separators.
        
        Very short anchors are discarded to avoid noise.
        
        Args:
            quote_norm: Normalized quote text
            min_chars: Minimum characters per anchor (default: MIN_ANCHOR_CHARS)
            
        Returns:
            List of anchor fragments
        """
        if not quote_norm:
            return []
        
        # Use "..." as the main separator; treat 3+ dots as ellipsis
        raw_parts = re.split(r"\s*\.{3,}\s*", quote_norm)
        anchors: List[str] = []
        for part in raw_parts:
            part = part.strip()
            if len(part) >= min_chars:
                anchors.append(part)
        return anchors
    
    def match_anchor_in_fulltext(
        self,
        anchor_norm: str,
        full_text_norm: str,
    ) -> AnchorMatch:
        """
        Match a single anchor against the full normalized text using token-level SequenceMatcher.
        
        Args:
            anchor_norm: Normalized anchor text
            full_text_norm: Normalized full text to search in
            
        Returns:
            AnchorMatch object with coverage and token span information
        """
        if not anchor_norm or not full_text_norm:
            return AnchorMatch(
                coverage=0.0,
                full_token_start=0,
                full_token_end=0,
                anchor_token_len=0,
            )
        
        anchor_tokens = anchor_norm.split()
        full_tokens = full_text_norm.split()
        if not anchor_tokens or not full_tokens:
            return AnchorMatch(
                coverage=0.0,
                full_token_start=0,
                full_token_end=0,
                anchor_token_len=len(anchor_tokens),
            )
        
        # Use SequenceMatcher on token sequences
        sm = difflib.SequenceMatcher(None, anchor_tokens, full_tokens, autojunk=False)
        blocks = sm.get_matching_blocks()
        if not blocks:
            return AnchorMatch(
                coverage=0.0,
                full_token_start=0,
                full_token_end=0,
                anchor_token_len=len(anchor_tokens),
            )
        
        # Find best matching block
        best_block = max(blocks, key=lambda b: b.size)
        coverage = best_block.size / float(len(anchor_tokens) or 1)
        
        return AnchorMatch(
            coverage=coverage,
            full_token_start=best_block.b,
            full_token_end=best_block.b + best_block.size,
            anchor_token_len=len(anchor_tokens),
        )
    
    def verify_quote_in_fulltext(
        self,
        quote: str,
        full_text_norm: str,
        min_anchor_chars: int = MIN_ANCHOR_CHARS,
        min_anchor_coverage: float = MIN_ANCHOR_COVERAGE,
        max_gap_tokens: int = MAX_GAP_TOKENS,
    ) -> Tuple[bool, float]:
        """
        Verify whether a quote is supported by the normalized full text.
        
        Strategy:
        1. Normalize the quote
        2. Split into anchors using ellipsis ("...") as separators
        3. For each anchor, find the best token-level match in full_text_norm
        4. Compute coverage (matched_tokens / anchor_tokens)
        5. If anchors have sufficient coverage and appear in a compact region,
           mark as found=True
        
        Args:
            quote: Quote text to verify
            full_text_norm: Normalized full text to search in
            min_anchor_chars: Minimum characters per anchor
            min_anchor_coverage: Minimum coverage threshold for good matches
            max_gap_tokens: Maximum gap between anchors for compact region
            
        Returns:
            Tuple of (found, confidence_score)
        """
        if not quote or not full_text_norm:
            return False, 0.0
        
        quote_norm = self.normalize_text(quote)
        if not quote_norm:
            return False, 0.0
        
        # Split into anchors by ellipsis; fall back to whole quote if needed
        anchors = self.split_into_anchors(quote_norm, min_chars=min_anchor_chars)
        if not anchors:
            anchors = [quote_norm]
        
        # Match each anchor
        anchor_matches: List[AnchorMatch] = []
        for anchor in anchors:
            match_info = self.match_anchor_in_fulltext(anchor, full_text_norm)
            anchor_matches.append(match_info)
        
        # Filter anchors with good coverage
        good_matches = [
            m for m in anchor_matches if m.coverage >= min_anchor_coverage
        ]
        if not good_matches:
            best = max((m.coverage for m in anchor_matches), default=0.0)
            return False, float(best)
        
        # Sort by location in full text
        good_matches.sort(key=lambda m: m.full_token_start)
        
        # Check that good anchors are not too far apart (compact region)
        compact = True
        for i in range(len(good_matches) - 1):
            gap = good_matches[i + 1].full_token_start - good_matches[i].full_token_end
            if gap > max_gap_tokens:
                compact = False
                break
        
        # Calculate confidence: combines average coverage and anchor hit ratio
        avg_cov = sum(m.coverage for m in good_matches) / float(len(good_matches))
        hit_ratio = len(good_matches) / float(len(anchors) or 1)
        confidence = avg_cov * COVERAGE_WEIGHT + hit_ratio * HIT_RATIO_WEIGHT
        if not compact:
            confidence *= COMPACT_PENALTY
        
        found = confidence >= MIN_CONFIDENCE_THRESHOLD
        return found, confidence
    
    def analyze_segment_similarity(
        self,
        orig_text: str,
        cand_text: str,
        min_block_tokens: int = MIN_BLOCK_TOKENS,
    ) -> Tuple[float, float, float]:
        """
        Analyze structural similarity between two text segments using token-level matching.
        
        Returns:
            Tuple of (score, coverage_orig, coverage_cand), where:
            - coverage_orig: fraction of original tokens covered by long common blocks
            - coverage_cand: fraction of candidate tokens covered by long common blocks
            - score: F1-style combined score of the two coverages
        """
        if not orig_text or not cand_text:
            return 0.0, 0.0, 0.0
        
        # Normalize and tokenize
        orig_norm = self.normalize_text(orig_text)
        cand_norm = self.normalize_text(cand_text)
        orig_tokens = orig_norm.split()
        cand_tokens = cand_norm.split()
        if not orig_tokens or not cand_tokens:
            return 0.0, 0.0, 0.0
        
        # Use SequenceMatcher on token sequences
        sm = difflib.SequenceMatcher(None, orig_tokens, cand_tokens, autojunk=False)
        blocks = sm.get_matching_blocks()
        if not blocks:
            return 0.0, 0.0, 0.0
        
        # Only keep sufficiently long common blocks (to avoid noise)
        long_blocks = [b for b in blocks if b.size >= min_block_tokens]
        if not long_blocks:
            return 0.0, 0.0, 0.0
        
        matched_tokens = sum(b.size for b in long_blocks)
        coverage_orig = matched_tokens / float(len(orig_tokens))
        coverage_cand = matched_tokens / float(len(cand_tokens))
        
        # F1-style score
        if coverage_orig + coverage_cand > 0.0:
            score = 2.0 * coverage_orig * coverage_cand / (coverage_orig + coverage_cand)
        else:
            score = 0.0
        
        return score, coverage_orig, coverage_cand

