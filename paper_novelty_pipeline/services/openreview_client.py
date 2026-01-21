# services/openreview_client.py
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import openreview


TITLE_KEYS = ["title"]
ABSTRACT_KEYS = ["abstract"]
KEYWORDS_KEYS = ["keywords"]
PRIMARY_AREA_KEYS = ["primary_area", "Primary Area", "primary_subject"]
SUBMISSION_NUMBER_KEYS = ["submission_number", "Submission Number", "submission_id"]
REVIEW_RATING_KEYS = ["rating", "recommendation"]


@dataclass
class OpenReviewMeta:
    forum_id: str
    title: Optional[str]
    abstract: Optional[str]
    keywords: List[str]
    authors: List[str]
    primary_area: Optional[str]
    submission_number: Optional[str]
    ratings: List[float]
    rating_mean: Optional[float]
    url: str
    raw_submission_content: Dict[str, Any] = field(default_factory=dict)  # 保存所有原始 submission content 字段


def get_val(d: Dict[str, Any], keys: List[str], default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, dict) and "value" in v:
                return v.get("value", default)
            return v
    return default


def normalize_keywords(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v)
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]
    return [s.strip()] if s.strip() else []


def extract_all_content_values(content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the actual values of all fields from a submission content dict
    (handling the `{ "value": ... }` wrapped structure).

    Args:
        content: The `content` dictionary of a submission note.

    Returns:
        Dict[str, Any]: Mapping from field name -> actual value.
    """

    result = {}
    for key, value in content.items():
        if isinstance(value, dict) and "value" in value:
            result[key] = value["value"]
        else:
            result[key] = value
    return result


def invitation_matches(inv: str, suffixes: List[str]) -> bool:
    return isinstance(inv, str) and any(inv.endswith(s) for s in suffixes)


def extract_first_matching_note(notes, suffixes: List[str]):
    for n in notes:
        inv = getattr(n, "invitation", None) or (
            getattr(n, "invitations", None) or [None]
        )[0]
        if invitation_matches(inv, suffixes):
            return n
    return None


def extract_all_matching_notes(notes, suffixes: List[str]):
    out = []
    for n in notes:
        inv = getattr(n, "invitation", None) or (
            getattr(n, "invitations", None) or [None]
        )[0]
        if invitation_matches(inv, suffixes):
            out.append(n)
    return out

def make_openreview_client_for_year(year: int):
    """Select API base URL by year; do not inherit local proxy env settings."""
    if year <= 2023:
        client = openreview.Client(baseurl="https://api.openreview.net")
    else:
        client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
    client.session.trust_env = False
    return client

# Common invitation suffixes
SUBMISSION_SUFFIXES = ["/-/Blind_Submission", "/-/Submission"]
REVIEW_SUFFIXES = ["/-/Official_Review", "/-/Review"]


def fetch_meta_for_forum(forum_id: str, year: int = 2026, debug: bool = False) -> OpenReviewMeta:
    """
    Given a forum_id (e.g., 'kiVIVBmMTP'), fetch basic metadata and review ratings from OpenReview.

    Args:
        forum_id: OpenReview forum ID
        year: Used to select the correct API version
        debug: Whether to output debug logs
    """
    import logging
    logger = logging.getLogger(__name__)
    
    client = make_openreview_client_for_year(year)
    forum_notes = list(client.get_all_notes(forum=forum_id))

    if not forum_notes:
        return OpenReviewMeta(
            forum_id=forum_id,
            title=None,
            abstract=None,
            keywords=[],
            authors=[],
            primary_area=None,
            submission_number=None,
            ratings=[],
            rating_mean=None,
            url=f"https://openreview.net/forum?id={forum_id}",
            raw_submission_content={},
        )

    # 1) submission note
    submission = extract_first_matching_note(forum_notes, SUBMISSION_SUFFIXES) or forum_notes[0]
    content = getattr(submission, "content", {}) or {}
    
    # 提取所有字段的实际值（用于保存到 raw_submission_content）
    raw_submission_content = extract_all_content_values(content)

    title = get_val(content, TITLE_KEYS)
    abstract = get_val(content, ABSTRACT_KEYS)
    keywords = normalize_keywords(get_val(content, KEYWORDS_KEYS, []))

    # authors: v2 一般有 'authors' 字段；兼容一下
    authors_val = get_val(content, ["authors", "authorids"], [])
    authors = [str(a).strip() for a in (authors_val or []) if str(a).strip()]

    primary_area = get_val(content, PRIMARY_AREA_KEYS)
    submission_number = get_val(content, SUBMISSION_NUMBER_KEYS)

    # 2) reviews → ratings
    review_notes = extract_all_matching_notes(forum_notes, REVIEW_SUFFIXES)
    
    if debug:
        logger.info(f"DEBUG: forum_id={forum_id}, found {len(review_notes)} review notes")
        # 打印所有 notes 的 invitation，确认是否真的是 review
        for idx, note in enumerate(forum_notes):
            inv = getattr(note, "invitation", None) or (getattr(note, "invitations", None) or [None])[0]
            logger.info(f"DEBUG: note {idx} invitation={inv}")

    ratings: List[float] = []
    for idx, rn in enumerate(review_notes):
        rc = getattr(rn, "content", {}) or {}
        
        if debug:
            logger.info(f"DEBUG: review {idx} content keys: {list(rc.keys())}")
            # 打印完整的 content，方便查看结构
            import json
            logger.info(f"DEBUG: review {idx} full content: {json.dumps(rc, indent=2, default=str)}")
        
        rating_text = get_val(rc, REVIEW_RATING_KEYS)
        
        if debug:
            logger.info(f"DEBUG: review {idx} rating_text={rating_text!r}, type={type(rating_text)}")
        
        if isinstance(rating_text, str):
            m = re.match(r"\s*(\d+)", rating_text)
            if m:
                score = float(m.group(1))
                if debug:
                    logger.info(f"DEBUG: review {idx} parsed score={score}")
                ratings.append(score)
            else:
                if debug:
                    logger.info(f"DEBUG: review {idx} regex match failed for '{rating_text}'")
        elif isinstance(rating_text, (int, float)):
            if debug:
                logger.info(f"DEBUG: review {idx} numeric score={rating_text}")
            ratings.append(float(rating_text))
        else:
            if debug:
                logger.info(f"DEBUG: review {idx} rating_text is None or unsupported type")

    rating_mean = sum(ratings) / len(ratings) if ratings else None
    
    if debug:
        logger.info(f"DEBUG: final ratings={ratings}, mean={rating_mean}")

    return OpenReviewMeta(
        forum_id=forum_id,
        title=title,
        abstract=abstract,
        keywords=keywords,
        authors=authors,
        primary_area=primary_area,
        submission_number=submission_number,
        ratings=ratings,
        rating_mean=rating_mean,
        url=f"https://openreview.net/forum?id={forum_id}",
        raw_submission_content=raw_submission_content,
    )