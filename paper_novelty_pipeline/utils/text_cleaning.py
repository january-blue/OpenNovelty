# utils/text_cleaning.py

import json
import logging
import re
from typing import Optional, List, Dict, Any


def truncate_at_references(text: str) -> Optional[str]:
    """
    Heuristic truncation: find common headings that mark the start of references
    or bibliography sections and return the text before that heading.

    Recognizes English headings like 'References', 'Bibliography', 'Acknowledgements'
    and Chinese '参考文献' as common markers. Case-insensitive and matches on
    a line that begins with the marker.

    Returns None if no truncation point found or input is invalid.
    """
    if not text or not isinstance(text, str):
        return None

    headings = [
        # Match 'References' or 'Bibliography' at start of line, 
        # allowing for concatenated page/line numbers often found in PDF extractions (e.g. References546)
        r"^\s*references?(\d+|\b)",
        r"^\s*references?\s+and\s+notes(\d+|\b)",
        r"^\s*references?\s+and\s+bibliography(\d+|\b)",
        r"^\s*bibliograph(y|ies)(\d+|\b)",
        r"^\s*works\s+cited(\d+|\b)",
        r"^\s*literature\s+cited(\d+|\b)",
        r"^\s*参考文献",
        r"^\s*参考资料",
        r"^\s*参考(\d+|\b)",
        r"^\s*acknowledg(e|ement)s?(\d+|\b)",
        r"^\s*致谢",
    ]

    pattern = re.compile(r"(?mi)" + r"|".join(headings))
    m = pattern.search(text)
    if m:
        idx = m.start()
        return text[:idx].strip()
    return None


def clean_extracted_text(text: str) -> str:
    """
    Light-weight cleanup for PDF extracted text:
    - Drop lines that are purely digits (likely line numbers/page numbers)
    - Collapse consecutive blank lines into a single blank line
    - Strip trailing whitespace on each line
    """
    if not isinstance(text, str):
        return ""

    lines = text.splitlines()
    cleaned: List[str] = []
    blank_pending = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not blank_pending:
                cleaned.append("")
                blank_pending = True
            continue
        if stripped.isdigit() and len(stripped) <= 4:
            # treat as line/page number; skip
            continue

        blank_pending = False
        cleaned.append(line.rstrip())

    return "\n".join(cleaned).strip()


def sanitize_unicode(text: str) -> str:
    """
    Remove surrogate characters that cause JSON encoding errors
    (e.g., 'invalid low surrogate in string').
    """
    if not isinstance(text, str):
        return ""
    # Use regex to remove lone surrogates (U+D800 to U+DFFF)
    # These often appear in malformed PDF extractions.
    return re.sub(r'[\ud800-\udfff]', ' ', text)


def sanitize_for_llm(text: str) -> str:
    """
    Normalize and strip problematic control characters to ensure safe UTF-8 JSON transport.

    This function prepares text for LLM consumption by:
    - Applying Unicode NFKC normalization (canonical composition)
    - Removing control characters (except tabs and newlines)
    - Removing invalid UTF-8 surrogate pairs

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text safe for JSON encoding and LLM processing
    """
    try:
        import unicodedata
        if not isinstance(text, str):
            text = str(text or "")
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        # Remove control chars except tabs/newlines
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
        # Drop invalid surrogates by round-tripping through utf-8
        text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
        return text
    except Exception:
        return text or ""


def strip_code_fence(text: str) -> str:
    """
    Remove markdown code fences from text (typically LLM output).

    Handles formats like:
    - ```json\n{...}\n```
    - ```\n{...}\n```

    Args:
        text: Input text potentially wrapped in code fences

    Returns:
        Text with code fences removed
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.lstrip("\n")
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.rstrip()
    return stripped


def parse_json_flexible(raw_text: Optional[str], logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    """
    Multi-strategy JSON parsing for LLM outputs.

    Attempts multiple strategies to parse potentially malformed JSON from LLM responses:
    1. Direct JSON parsing
    2. Strip code fences and retry
    3. Fix common LLM mistakes (closing array with } instead of ])
    4. Aggressive brace/bracket counting and fixing

    Args:
        raw_text: Raw text output from LLM (potentially containing JSON)
        logger: Optional logger for debug messages (uses root logger if None)

    Returns:
        Parsed dictionary if successful, None otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not raw_text:
        return None

    candidate = raw_text.strip()
    if not candidate:
        return None

    candidate = strip_code_fence(candidate)

    # First attempt: direct parse
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed: {e}")

    # Second attempt: fix common LLM mistake - using } instead of ] to close arrays
    # This happens when LLM writes: {"contributions": [{...}, {...}}} instead of {"contributions": [{...}, {...}]}
    try:
        # Pattern: array items followed by }} instead of ]}
        # Fix pattern: }\n  }\n} -> }\n  ]\n} (for contributions array)
        fixed = re.sub(r'\}\s*\}\s*\}$', '}\n  ]\n}', candidate)
        if fixed != candidate:
            result = json.loads(fixed)
            logger.info("Fixed JSON array closing brace and parsed successfully")
            return result
    except Exception:
        pass

    # Third attempt: more aggressive brace fixing
    try:
        # Count opening and closing braces/brackets
        open_braces = candidate.count('{')
        close_braces = candidate.count('}')
        open_brackets = candidate.count('[')
        close_brackets = candidate.count(']')

        # If we have more } than expected (one extra } where ] should be)
        if close_braces == open_braces + 1 and close_brackets == open_brackets - 1:
            # Find the second-to-last } and replace with ]
            last_brace = candidate.rfind('}')
            if last_brace > 0:
                second_last_brace = candidate.rfind('}', 0, last_brace)
                if second_last_brace > 0:
                    fixed = candidate[:second_last_brace] + ']' + candidate[second_last_brace+1:]
                    result = json.loads(fixed)
                    logger.info("Fixed mismatched brace/bracket and parsed successfully")
                    return result
    except Exception:
        pass

    # All attempts failed
    preview = candidate[:200].replace("\n", " ")
    logger.warning(f"Failed to parse JSON after all attempts; content head={preview!r}")
    return None
