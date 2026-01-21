# Phase 3 Data Flow and Dependencies

This document describes the execution order and data dependencies for Phase 3 components.

## Execution Order

Phase 3 components must be executed in the following order:

```
Part 1: Survey (3 steps: Short Survey, Taxonomy, Narrative)
Part 2: PDF Download & Text Extraction
Part 3: Textual Similarity Detection (optional, skip with SKIP_TEXTUAL_SIMILARITY=true)
Part 4: Core Task Comparisons
Part 5: Contribution Analysis
Part 6: Final Report Generation
```

### Part 1: Core Task Survey
**Purpose**: Generate taxonomy structure and initial citation_index

**Steps**:
1. **Initial Report** (`run_phase3_short_survey.py`)
   - Reads Phase2 final outputs
   - Builds initial `survey_report.json` with citation_index
   - Output: `phase3/core_task_survey/survey_report.json`

2. **Taxonomy Generation** (`run_phase3_taxonomy.py`)
   - Generates hierarchical taxonomy structure
   - Output: `phase3/core_task_survey/taxonomy.json`
   - Updates: `survey_report.json` with taxonomy mapping

3. **Narrative Generation** (`run_phase3_narrative.py`)
   - Generates narrative text describing the taxonomy
   - Output: `phase3/core_task_survey/narrative.json`
   - Updates: `survey_report.json` with narrative

**Outputs**:
- `phase3/core_task_survey/survey_report.json` (contains `citation_index` and `mapping`)
- `phase3/core_task_survey/taxonomy.json`
- `phase3/core_task_survey/narrative.json`

### Part 2: PDF Download & Text Extraction
**Purpose**: Download PDFs and extract full texts for all candidate papers, populating TextCache

**Script**: `scripts/run_phase3_pdf_download.py`

**Dependencies**:
- `phase2/final/core_task_perfect_top50.json` (Core Task candidates)
- `phase2/final/contribution_*_perfect_top10.json` (Contribution candidates)
- `phase1/fulltext_cleaned.txt` (original paper full text)

**Key Features**:
- Parallel PDF downloading (configurable max_workers)
- Text extraction and caching to TextCache
- Auto cleanup of temporary PDFs after extraction
- Deduplication by canonical_id

**Outputs**:
- `phase3/cached_paper_texts/*.json` (cached full texts)
- `phase3/pdf_extraction/metadata.json` (extraction statistics)

### Part 3: Textual Similarity Detection
**Purpose**: Detect textual similarities for all papers ONCE to avoid redundant LLM calls

**Script**: `scripts/run_phase3_textual_similarity.py`

**Prerequisites**:
- Part 2 (PDF Download) should run first to populate TextCache
- If TextCache is empty, papers will use abstracts as fallback

**Skippable**: Set `SKIP_TEXTUAL_SIMILARITY=true` to skip this part entirely

**Dependencies**:
- `phase3/cached_paper_texts/` (populated by Part 2)
- `phase3/core_task_survey/taxonomy.json` (optional, for sibling inference)
- `phase1/fulltext_cleaned.txt` (original paper full text)

**Detection Scope**:
- Core Task siblings (same taxonomy leaf as original paper)
- All Contribution candidates

**Key Features**:
- Deduplication: Each paper (by canonical_id) is analyzed ONCE
- Text loading: Reads from TextCache (no PDF downloads here)
- Independent execution: Results are reused by Part 4/5/6

**Outputs**:
- `phase3/textual_similarity_detection/results.json` (similarity segments for all papers)
- `phase3/textual_similarity_detection/metadata.json` (detection metadata)
- `phase3/textual_similarity_detection/per_paper/*.json` (per-paper detailed results)

**Data Flow**:
```
Part 2 populates TextCache
    ↓
Part 3 generates results.json (or skipped if SKIP_TEXTUAL_SIMILARITY=true)
    ↓
Part 4/5 perform comparisons (WITHOUT similarity detection)
    ↓
Part 6 loads results.json and fills similarity segments into final report
```

### Part 4: Core Task Comparisons
**Purpose**: Compare original paper against top50 core_task candidates using taxonomy-based hierarchical filtering

**Script**: `run_phase3_core_task_comparisons.py`

**Dependencies**:
- `phase3/core_task_survey/taxonomy.json` (required; hierarchical filtering + structural position)
- `phase3/core_task_survey/survey_report.json` (required; citation_index + canonical_id→taxonomy_path mapping)
- `phase3/textual_similarity_detection/results.json` (similarity segments, loaded by Part 6)

**Behavior**:
- **Hard prerequisite**: If `taxonomy.json` or `survey_report.json` is missing/unreadable, core-task comparisons are **skipped** and an empty `core_task_comparisons.json` + `summary.json` is written with a clear skip reason.
- **Eligibility**: Only candidates classified as **taxonomy siblings** (same leaf as the original paper) are compared at the paper level.
- **Paper-level comparison** (when sibling papers exist): For each sibling candidate, run a **brief distinction** LLM step (includes **duplicate/variant detection**). Comparison prefers **full text** when available; otherwise it falls back to **abstract-only**.
- **Subtopic-level fallback** (when *no* sibling papers exist but sibling subtopics exist under the same parent): no per-paper comparisons are produced; instead the system writes `subtopic_comparison.json` with an LLM summary comparing the original leaf against sibling subtopics.
- **Structural isolation** (when neither sibling papers nor sibling subtopics exist): write a structural note indicating the paper is isolated in the taxonomy.
- **NOTE**: This module does **not** run unified textual similarity detection (handled by Part 3). Similarity segments are filled into the final report during Part 6.

**Outputs**:
- `phase3/core_task_comparisons/*.json` (individual comparisons for sibling papers only)
- `phase3/core_task_comparisons/core_task_comparisons.json` (merged)
- `phase3/core_task_comparisons/summary.json`
- `phase3/core_task_comparisons/references.md` (unified references)
- `phase3/core_task_comparisons/citation_index.json` (extended citation_index)
- `phase3/core_task_comparisons/subtopic_comparison.json` (only when subtopic-level fallback is used)

### Part 5: Contribution Analysis
**Purpose**: Compare original paper's contributions against candidate papers

**Script**: `run_phase3_contribution_analysis.py`

**Dependencies**:
- `phase3/core_task_survey/survey_report.json` (for citation_index extension)
- `phase3/textual_similarity_detection/results.json` (similarity segments, loaded by Part 6)

**Behavior**:
- Loads existing citation_index from survey_report
- Extends citation_index with Contribution Analysis candidates
- Assigns new citation indices starting from max_index + 1
- **NOTE**: Does NOT perform textual similarity detection (handled by Part 3)
- Only performs Task 1: Contribution Comparison Analysis

**Outputs**:
- `phase3/contribution_analysis/contribution_*/paper_*.json` (per-contribution comparisons, WITHOUT similarity segments)
- `phase3/contribution_analysis/phase3_report.json` (main report)
- `phase3/contribution_analysis/references.md` (unified references)
- `phase3/contribution_analysis/citation_index.json` (extended citation_index)

### Part 6: Report Generation
**Purpose**: Generate unified final report with all Phase3 results

**Script**: `run_phase3_generate_complete_report.py`

**Dependencies**:
- All Phase3 component outputs (Part 1-5)
- Extended citation_index from both Core Task Analysis and Contribution Analysis
- `phase3/textual_similarity_detection/results.json` (similarity segments)

**Key Operations**:
1. Load Core Task Survey, Taxonomy, and Narrative
2. Load Contribution Analysis results
3. Load Core Task Comparisons results
4. **Load textual similarity results from Part 3**
5. **Fill similarity segments into comparisons** (via `_fill_similarity_to_comparisons()`)
6. Build unified plagiarism_detection index
7. Build references section

**Outputs**:
- `phase3/phase3_complete_report.json` (final merged report with ALL similarity segments filled)

## Data Dependencies (4.2)

### Core Task Analysis Dependencies

**Required**:
- `phase1/paper.json`
- `phase1/phase1_extracted.json`
- `phase2/final/core_task_perfect_top50.json`

**Optional (but recommended)**:
- `phase3/core_task_survey/taxonomy.json` - Enables hierarchical filtering
- `phase3/core_task_survey/survey_report.json` - Provides citation_index and taxonomy mapping

**Fallback Behavior**:
- If taxonomy files are missing, Core Task Analysis will:
  - Compare all candidates (no filtering)
  - Start citation_index from scratch (if survey_report missing)

### Contribution Analysis Dependencies

**Required**:
- `phase1/paper.json`
- `phase1/phase1_extracted.json`
- `phase2/final/contribution_*_topk.json` (per-contribution candidate files)

**Optional (but recommended)**:
- `phase3/core_task_survey/survey_report.json` - Provides initial citation_index

**Fallback Behavior**:
- If survey_report is missing, Contribution Analysis will:
  - Start citation_index from scratch (index 0)
  - All candidates will get new citation indices

### Report Generation Dependencies

**Required**:
- All Phase3 component outputs (from steps 1-3)

**Optional**:
- Extended citation_index files (for unified references)

## Citation Index Flow

1. **Initial State** (from Phase3 Survey):
   - `survey_report.json` contains `citation_index` with Top50 papers
   - Indices: 0 (original paper) to 49 (top candidates)

2. **Core Task Analysis Extension**:
   - Loads citation_index from survey_report
   - Adds any missing candidates (if not in Top50)
   - New indices start from max_index + 1

3. **Contribution Analysis Extension**:
   - Loads citation_index (from survey_report or Core Task Analysis)
   - Adds all Contribution Analysis candidates
   - New indices continue from current max_index + 1

4. **Final State**:
   - Unified citation_index containing:
     - Original paper (index 0)
     - Core Task Top50 (indices 1-50)
     - Additional Core Task candidates (if any)
     - Contribution Analysis candidates (indices 51+)

## Error Handling

All components handle missing dependencies gracefully:

- **Missing taxonomy**: Core Task Analysis falls back to comparing all candidates
- **Missing citation_index**: Components start with empty citation_index
- **Missing survey_report**: Components log warnings but continue execution

## Recommended Execution

Use `scripts/run_phase3_all.sh` which ensures correct execution order:

```bash
./scripts/run_phase3_all.sh output/claim_based/openreview_XXX_YYYYMMDD
```

This script:
1. Runs Phase3 Survey (3 steps)
2. Runs PDF Download & Text Extraction
3. Runs Textual Similarity Detection (optional via SKIP_TEXTUAL_SIMILARITY)
4. Runs Core Task Comparisons (with dependency checks)
5. Runs Contribution Analysis (with dependency checks)
6. Generates complete report

