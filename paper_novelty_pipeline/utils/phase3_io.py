from __future__ import annotations

"""
Utilities for Phase3 Short Survey I/O and normalization.

This module provides helpers to:
- read Phase2 `final/index.json`
- load Top50/Top10 files
- build canonical ids (doi > arxiv > normalized title)
- trim abstracts and prepare minimal metadata rows
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paper_novelty_pipeline.utils.paper_id import make_canonical_id
from paper_novelty_pipeline.utils.paths import PAPER_JSON


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------- Label formatting helpers (EN) ---------------------

def _surname_from_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return "Anon"
    # Handle "Surname, Given" format first
    if "," in n:
        return n.split(",", 1)[0].strip()
    # Else take last token
    parts = [p for p in n.split() if p]
    return parts[-1] if parts else "Anon"


def author_label_en(authors: Any) -> str:
    try:
        if isinstance(authors, list) and authors:
            if len(authors) == 1:
                return f"{_surname_from_name(authors[0])}"
            if len(authors) == 2:
                return f"{_surname_from_name(authors[0])} and {_surname_from_name(authors[1])}"
            return f"{_surname_from_name(authors[0])} et al."
        elif isinstance(authors, str) and authors.strip():
            # Try split a string list by comma
            arr = [a.strip() for a in authors.split(',') if a.strip()]
            return author_label_en(arr)
    except Exception:
        pass
    return "Anon et al."


def title_short_en(title: str, max_len: int = 60) -> str:
    t = (title or "").strip()
    if len(t) <= max_len:
        return t
    # Heuristic compression: keep capitalized/acronym/long tokens
    import re
    tokens = re.split(r"\s+", t)
    picked: List[str] = []
    for tok in tokens:
        if len(" ".join(picked)) + 1 + len(tok) > max_len:
            break
        if tok.isupper() or tok[:1].isupper() or len(tok) >= 5:
            picked.append(tok)
    if not picked:
        return (t[: max_len - 1] + "…")
    s = " ".join(picked)
    return s if len(s) <= max_len else (s[: max_len - 1] + "…")


def build_paper_label_en(item: Dict[str, Any], *, numbering: bool = True) -> str:
    rank = item.get("rank")
    title_short = title_short_en(item.get("title") or "")
    authors = item.get("authors") or []
    year = item.get("year") or "n.d."
    try:
        year = int(year)
    except Exception:
        year = "n.d."
    auth = author_label_en(authors)
    prefix = f"[#${{rank}}] " if numbering and rank else ""
    # Mermaid code block will replace ${rank} later, so we render concrete here
    if numbering and rank:
        prefix = f"[#{rank}] "
    return f"{prefix}{title_short} ({auth}, {year})"


def build_mermaid_mindmap(taxonomy: Dict[str, Any], id_to_label: Dict[str, str], *, root_name: Optional[str] = None, highlight_id: Optional[str] = None) -> str:
    lines: List[str] = ["mindmap"]
    root = taxonomy.get("name") or root_name or "Survey"
    lines.append(f"  {root}")

    def emit(node: Dict[str, Any], indent: int) -> None:
        pad = "  " * indent
        name = node.get("name") or "Unnamed"
        subs = node.get("subtopics")
        papers = node.get("papers")
        if isinstance(subs, list) and subs:
            lines.append(f"{pad}{name}")
            for ch in subs:
                if isinstance(ch, dict):
                    emit(ch, indent + 1)
            return
        if isinstance(papers, list) and papers:
            lines.append(f"{pad}{name}")
            for pid in papers:
                label = id_to_label.get(pid, pid)
                # Highlight handled by front-end via mapping; here仅输出文本
                lines.append(f"{pad}  {label}")

    for ch in taxonomy.get("subtopics", []) or []:
        if isinstance(ch, dict):
            emit(ch, 2)
    return "\n".join(lines)


# --------------------- Alternative: Flowchart-style taxonomy ---------------------
def _html_escape(text: str) -> str:
    """Minimal HTML escaping for label content (we allow HTML tags separately)."""
    t = (text or "").replace("\n", " ")
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = t.replace('"', '&quot;')
    return t


def _nowrap_html(text: str) -> str:
    """Wrap content with a span to prevent wrapping in Mermaid HTML labels."""
    return f"<span style=\"white-space:nowrap;\">{_html_escape(text)}</span>"


def _comma_join_nowrap(items: List[str]) -> str:
    """Join items with commas, prefer a single long line, but allow breaking at commas only.

    Implemented by inserting <wbr> after commas and applying white-space:nowrap.
    Browsers typically allow a break opportunity at <wbr> even inside nowrap.
    """
    clean = [ _html_escape(it.strip()) for it in items if (it or "").strip() ]
    if not clean:
        return ""
    s = ", ".join(clean)
    s = s.replace(",", ",<wbr>")
    return f"<span style=\"white-space:nowrap;\">{s}</span>"


def _wrap_join(items: List[str], *, max_line_chars: Optional[int] = 48) -> str:
    """Join items with comma and insert <br/> to wrap long lines.

    This helps produce compact boxes similar to survey figures.
    """
    # max_line_chars=None  -> no wrapping at all, commas only
    # max_line_chars==0    -> one item per line (join with <br/>)
    clean = [it.strip() for it in items if (it or "").strip()]
    if max_line_chars is None:
        return ", ".join(clean)
    if max_line_chars == 0:
        return "<br/>".join(clean)

    lines: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for it in items:
        it = (it or "").strip()
        if not it:
            continue
        add = (", ".join(cur) + (", " if cur else "") + it)
        if len(add) > max_line_chars and cur:
            lines.append(", ".join(cur))
            cur = [it]
            cur_len = len(it)
        else:
            cur.append(it)
            cur_len = len(add)
    if cur:
        lines.append(", ".join(cur))
    return "<br/>".join(lines)


def build_mermaid_flowchart(
    taxonomy: Dict[str, Any],
    id_to_label: Dict[str, str],
    *,
    root_name: Optional[str] = None,
    highlight_id: Optional[str] = None,
    papers_wrap: Optional[int] = 48,
) -> str:
    """Render taxonomy as a left-to-right flowchart with rounded category nodes
    and paper lists grouped in right-hand boxes.

    - Internal nodes (categories) use rounded shapes and colored classes.
    - Leaf nodes create a companion 'papers' box which lists all papers joined
      with commas and wrapped with <br/> line breaks.
    - If highlight_id is provided, prefix that paper with a star.
    """
    # Init directive: straight lines (curve=linear) + slightly tighter spacing
    lines: List[str] = [
        "%%{init: { 'securityLevel': 'loose', 'flowchart': { 'htmlLabels': true, 'curve': 'step', 'nodeSpacing': 12, 'rankSpacing': 28, 'padding': 4 } }}%%",
        "flowchart LR",
    ]
    # Define styles (kept simple for portability across Mermaid versions)
    lines += [
        "classDef root fill:#f5f7fa,stroke:#8a8f98,stroke-width:1,color:#2c3e50,font-weight:bold;",
        "classDef lvl1 fill:#e7f0ff,stroke:#5a8dee,stroke-width:1,color:#1f3a93;",
        "classDef lvl2 fill:#fdf0f5,stroke:#d86aa6,stroke-width:1,color:#7a2e52;",
        "classDef box fill:#ffffff,stroke:#b5bdc9,stroke-width:1,rx:6,ry:6,color:#2c3e50,font-size:11px;",
        "linkStyle default stroke:#b5bdc9,stroke-width:0.8;",
    ]

    nid = 0

    def new_id(prefix: str = "n") -> str:
        nonlocal nid
        nid += 1
        return f"{prefix}{nid}"

    def add_node(node_id: str, shape: str, label: str, cls: Optional[str] = None) -> None:
        # label is expected to be HTML-safe already
        lbl = label
        # shape: 'rect' | 'round' | 'stadium'
        if shape == "round":
            lines.append(f'{node_id}("{lbl}")')
        elif shape == "stadium":
            lines.append(f'{node_id}(["{lbl}"])')
        else:  # rect
            # Wrap with a compact div to tighten vertical padding
            lines.append(f'{node_id}["<div style=\'padding:2px 6px;line-height:1.15;\'>{lbl}</div>"]')
        if cls:
            lines.append(f"class {node_id} {cls};")

    def walk(node: Dict[str, Any], parent_id: Optional[str], depth: int) -> str:
        name = node.get("name") or "Unnamed"
        this_id = new_id("cat")
        # Rounded categories
        cls = "root" if depth == 0 else ("lvl1" if depth == 1 else "lvl2")
        add_node(this_id, "stadium", _nowrap_html(name), cls)
        # For stadium effect, prefer style via class; shapes are constrained in CLI
        if parent_id:
            lines.append(f"{parent_id} --> {this_id}")

        subs = node.get("subtopics")
        papers = node.get("papers")
        if isinstance(subs, list) and subs:
            for ch in subs:
                if isinstance(ch, dict):
                    walk(ch, this_id, depth + 1)
        elif isinstance(papers, list) and papers:
            # Build paper list box node
            labels: List[str] = []
            for pid in papers:
                lab = id_to_label.get(pid, str(pid))
                if highlight_id and pid == highlight_id:
                    lab = f"★ {lab}"
                labels.append(lab)
            if papers_wrap is None:
                text = _comma_join_nowrap(labels)
            elif papers_wrap == 0:
                # one paper per line
                text = "<br/>".join(_html_escape(x) for x in labels)
            else:
                text = _wrap_join(labels, max_line_chars=papers_wrap)
            box_id = new_id("box")
            add_node(box_id, "rect", text, "box")
            lines.append(f"{this_id} --> {box_id}")
        return this_id

    root = {"name": taxonomy.get("name") or (root_name or "Survey"), "subtopics": taxonomy.get("subtopics")}
    root_id = walk(root, None, 0)
    # Ensure root has style
    lines.append(f"class {root_id} root;")
    return "\n".join(lines)


def read_final_index(phase2_dir: Path) -> Dict[str, Any]:
    """Read Phase2 final/index.json and return a dict with file paths.

    Returns keys: core_task_file (str or None), contribution_files (List[str])
    
    Note: Paths in index.json may be relative (to phase2_dir) or absolute.
    This function resolves relative paths automatically against phase2_dir.
    """
    from paper_novelty_pipeline.config import PROJECT_ROOT
    
    index_path = Path(phase2_dir) / "final" / "index.json"
    if not index_path.exists():
        # Fallback to core/contrib files by convention
        core = Path(phase2_dir) / "final" / "core_task_perfect_top50.json"
        contrib_files = sorted((Path(phase2_dir) / "final").glob("contribution_*_perfect_top10.json"))
        return {
            "core_task_file": str(core) if core.exists() else None,
            "contribution_files": [str(p) for p in contrib_files],
        }
    
    data = read_json(index_path)
    
    # Helper: resolve relative paths
    def resolve_path(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        path_obj = Path(p)
        if path_obj.is_absolute():
            return str(path_obj)
        
        # Try resolving relative to phase2_dir first (new format)
        resolved = (Path(phase2_dir) / path_obj).resolve()
        if resolved.exists():
            return str(resolved)
        
        # Fallback: resolve against project root (legacy format)
        resolved_legacy = (PROJECT_ROOT / path_obj).resolve()
        if resolved_legacy.exists():
            return str(resolved_legacy)
        
        # If neither exists, prefer phase2_dir relative (new format)
        return str((Path(phase2_dir) / path_obj).resolve())
    
    core_file = resolve_path(data.get("core_task_file"))
    contrib_files = [resolve_path(f) for f in (data.get("contribution_files", []) or [])]
    
    return {
        "core_task_file": core_file,
        "contribution_files": contrib_files,
    }


_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_ARXIV_STD_RE = re.compile(r"^(\d{4})\.(\d{4,5})$", re.IGNORECASE)


def _normalize_title(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_arxiv_id_any(item: Dict[str, Any]) -> Optional[str]:
    def _clean(v: str) -> Optional[str]:
        v = (v or "").strip()
        if not v:
            return None
        v = _ARXIV_VERSION_RE.sub("", v)
        return v if _ARXIV_STD_RE.match(v) else None

    # direct field
    arx = _clean(str(item.get("arxiv_id") or ""))
    if arx:
        return arx
    # from urls/ids
    for k in ("url", "source_url", "id"):
        val = item.get(k)
        if isinstance(val, str) and "arxiv.org" in val:
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", val)
            if m:
                return m.group(1)
    # paper_id may be list or string
    pid = item.get("paper_id")
    if isinstance(pid, list) and pid:
        for v in pid:
            v = str(v)
            c = _clean(v)
            if c:
                return c
    elif isinstance(pid, str):
        c = _clean(pid)
        if c:
            return c
    # raw_metadata
    rm = item.get("raw_metadata") or {}
    if isinstance(rm, dict):
        val = rm.get("id") or rm.get("url")
        if isinstance(val, str) and "arxiv.org" in val:
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", val)
            if m:
                return m.group(1)
    return None


def canonical_id(item: Dict[str, Any]) -> str:
    """
    Build a canonical_id for Phase3 survey inputs using the global make_canonical_id helper.

    Priority is delegated to make_canonical_id:
    1) DOI          -> 'doi:{DOI}'
    2) arXiv ID     -> 'arxiv:{id}'
    3) OpenReview   -> 'openreview:{forum_id}'
    4) Title(+year) -> 'title:{normalized_title}|{year}|h:{short_hash}'
    5) Fallback     -> 'auto:{short_hash}'
    """
    raw_meta = item.get("raw_metadata") or {}
    try:
        return make_canonical_id(
            paper_id=item.get("paper_id") or raw_meta.get("paper_id") or raw_meta.get("id"),
            doi=item.get("doi") or raw_meta.get("doi"),
            arxiv_id=item.get("arxiv_id") or raw_meta.get("arxiv_id"),
            url=item.get("url") or item.get("source_url") or raw_meta.get("url"),
            pdf_url=item.get("pdf_url") or raw_meta.get("pdf_url"),
            title=item.get("title") or raw_meta.get("title"),
            year=item.get("year") or raw_meta.get("year"),
        )
    except Exception:
        # Fallback to simple title-based id if make_canonical_id fails for any reason
        title = item.get("title") or raw_meta.get("title") or ""
    return f"title:{_normalize_title(title)}"


def short_abstract(text: Optional[str], max_len: int = 600) -> str:
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text).strip()
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def prefer_url(item: Dict[str, Any]) -> Optional[str]:
    # prefer DOI URL if DOI present
    doi = (item.get("doi") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    for k in ("url", "source_url"):
        if item.get(k):
            return item.get(k)
    # arXiv
    arx = _extract_arxiv_id_any(item)
    if arx:
        return f"https://arxiv.org/abs/{arx}"
    return None


def load_topk_core(core_file: Path) -> List[Dict[str, Any]]:
    arr = read_json(Path(core_file))
    # Ensure list-like
    if isinstance(arr, dict):
        arr = arr.get("items") or arr.get("merged_dedup") or []
    items: List[Dict[str, Any]] = []
    for rank, it in enumerate(arr, start=1):
        row = {
            **it,
            "paper_id": canonical_id(it),
            "rank": rank,
            "url_pref": prefer_url(it),
            "abstract_short": short_abstract(it.get("abstract")),
        }
        items.append(row)
    return items


def load_contrib_file(path: Path) -> List[Dict[str, Any]]:
    arr = read_json(Path(path))
    if isinstance(arr, dict):
        arr = arr.get("items") or arr.get("merged_dedup") or []
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(arr, start=1):
        out.append({**it, "paper_id": canonical_id(it), "rank": idx, "url_pref": prefer_url(it)})
    return out


def read_paper_info(phase1_dir: Path) -> Dict[str, Any]:
    """
    Read original paper info from Phase1 (basic metadata) and Phase2 (canonical_id).
    
    Phase2 citation_index.json is the canonical source for canonical_id.
    Phase1 paper.json provides other metadata (title, authors, etc.).
    """
    # Ensure phase1_dir is a Path object
    phase1_dir = Path(phase1_dir)
    
    pj = phase1_dir / PAPER_JSON
    if not pj.exists():
        return {}
    d = read_json(pj)
    
    # Try to read canonical_id from Phase2 citation_index.json (index 0 = original paper)
    canonical_id = None
    try:
        phase2_citation_index = phase1_dir.parent / "phase2" / "final" / "citation_index.json"
        if phase2_citation_index.exists():
            citation_data = read_json(phase2_citation_index)
            items = citation_data.get("items", [])
            if items and len(items) > 0:
                original_item = items[0]
                roles = original_item.get("roles", [])
                if any(r.get("type") == "original_paper" for r in roles):
                    canonical_id = original_item.get("canonical_id")
    except Exception:
        pass
    
    # Fallback: generate canonical_id from title if Phase2 not available
    if not canonical_id:
        canonical_id = d.get("canonical_id")  # May be None if Phase1 doesn't have it
    
    return {
        "canonical_id": canonical_id,
        "paper_id": d.get("paper_id"),
        "title": d.get("title"),
        "authors": d.get("authors") or [],
        "venue": d.get("venue"),
        "year": d.get("year"),
        "doi": d.get("doi"),
        "url": d.get("paper_id") or d.get("url"),
    }


def canonical_id_from_paper_info(paper_info: Dict[str, Any]) -> str:
    """
    Derive a canonical_id for the original paper based on Phase1 paper_info.

    If paper_info already contains a canonical_id field, reuse it.
    Otherwise, delegate to make_canonical_id with the available metadata.
    """
    existing = (paper_info.get("canonical_id") or "").strip()
    if existing:
        return existing

    try:
        return make_canonical_id(
            paper_id=paper_info.get("paper_id") or paper_info.get("url"),
            doi=paper_info.get("doi"),
            arxiv_id=None,
            url=paper_info.get("url"),
            pdf_url=None,
            title=paper_info.get("title"),
            year=paper_info.get("year"),
        )
    except Exception:
        # Fallback to simple title-based id
        return f"title:{_normalize_title(paper_info.get('title') or '')}"
