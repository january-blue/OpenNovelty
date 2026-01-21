#!/bin/bash
# Run all Phase3 components in sequence with Pre-check and Post-audit
# Usage: ./scripts/run_phase3_all.sh [OUTPUT_DIR]
#
# Phase 3 Parts:
#   Part 1: Core Task Survey (3 steps: Short Survey, Taxonomy, Narrative)
#   Part 2: PDF Download & Text Extraction
#   Part 3: Textual Similarity Detection (optional, skip with SKIP_TEXTUAL_SIMILARITY=true)
#   Part 4: Core Task Comparisons
#   Part 5: Contribution Analysis
#   Part 6: Final Report Generation

# Use configurable PYTHONPATH (defaults to current working directory)
: "${PYTHONPATH:=$(pwd)}"
export PYTHONPATH

# Do not use set -e because we want to handle errors manually for balance detection
# set -e

OUTPUT_DIR="${1:-output/claim_based/openreview_ZgCCDwcGwn_20251203}"
PHASE1_DIR="$OUTPUT_DIR/phase1"
PHASE2_DIR="$OUTPUT_DIR/phase2"
OUT_DIR="$OUTPUT_DIR"

echo "=========================================="
echo "Running Phase3 for: $OUTPUT_DIR"
echo "=========================================="

# --- 1. Pre-check Logic ---

# A. Check Phase 2 completeness (must have citation_index and raw_responses >= 6)
if [ ! -f "$PHASE2_DIR/final/citation_index.json" ]; then
    echo "[SKIP] Missing phase2/final/citation_index.json"
    exit 0
fi

if [ -d "$PHASE2_DIR/raw_responses" ]; then
    RAW_COUNT=$(ls "$PHASE2_DIR/raw_responses" 2>/dev/null | wc -l)
    if [ "$RAW_COUNT" -lt 6 ]; then
        echo "[SKIP] Phase 2 incomplete (raw_responses has only $RAW_COUNT files, expected >= 6)"
        exit 0
    fi
else
    echo "[SKIP] Missing phase2/raw_responses directory"
    exit 0
fi

# B. Check if already completed (if complete_report exists, skip)
if [ -f "$OUT_DIR/phase3/phase3_complete_report.json" ]; then
    echo "[SKIP] Phase 3 report already exists"
    exit 0
fi

# --- 2. Step Execution Wrapper (with circuit breaker detection) ---

run_step() {
    local cmd="$1"
    local step_desc="$2"
    echo ""
    echo "=== $step_desc ==="

    local tmp_log=$(mktemp)

    # Execute and print logs in real-time while capturing to temp file
    eval "$cmd" 2>&1 | tee "$tmp_log"
    local ret=${PIPESTATUS[0]}

    if [ $ret -ne 0 ]; then
        # Circuit breaker: detect specific OpenRouter insufficient credits error
        if grep -q "Your account or API key has insufficient credits" "$tmp_log"; then
            echo "[FATAL] Detected API insufficient credits (402 Error)! Triggering global circuit breaker..."
            rm -f "$tmp_log"
            exit 42  # Special exit code for batch runner
        fi
        echo "$step_desc failed (exit code: $ret)"
        rm -f "$tmp_log"
        exit $ret
    fi
    rm -f "$tmp_log"
}

# --- 3. Run Phase 3 Components ---

# Part 1: Core Task Survey (3 steps)
run_step "python -m scripts.run_phase3_short_survey --phase2-dir '$PHASE2_DIR' --out-dir '$OUT_DIR' --language en --max-contrib 3 --log-level INFO" "Part 1 Step 1/3: Short Survey"

run_step "python -m scripts.run_phase3_taxonomy --phase2-dir '$PHASE2_DIR' --out-dir '$OUT_DIR' --language en --log-level INFO" "Part 1 Step 2/3: Taxonomy Generation"

run_step "python -m scripts.run_phase3_narrative --phase2-dir '$PHASE2_DIR' --out-dir '$OUT_DIR' --language en --log-level INFO" "Part 1 Step 3/3: Narrative Generation"

# Part 2: PDF Download & Text Extraction
run_step "python -m scripts.run_phase3_pdf_download --phase2-dir '$PHASE2_DIR/final' --phase3-dir '$OUT_DIR/phase3' --phase1-dir '$PHASE1_DIR' --max-workers 3" "Part 2: PDF Download & Text Extraction"

# Part 3: Textual Similarity Detection (optional, controlled by SKIP_TEXTUAL_SIMILARITY)
TAXONOMY_FILE="$OUT_DIR/phase3/core_task_survey/taxonomy.json"
if [ "${SKIP_TEXTUAL_SIMILARITY:-false}" = "true" ]; then
    echo ""
    echo "=== Part 3: Textual Similarity Detection - SKIPPED ==="
    echo "SKIP_TEXTUAL_SIMILARITY=true is set. Skipping Part 3."
else
    if [ -f "$TAXONOMY_FILE" ]; then
        run_step "python -m scripts.run_phase3_textual_similarity --phase2-dir '$PHASE2_DIR/final' --phase3-dir '$OUT_DIR/phase3' --phase1-dir '$PHASE1_DIR' --taxonomy '$TAXONOMY_FILE' --log-dir '$OUT_DIR/logs'" "Part 3: Textual Similarity Detection (with taxonomy)"
    else
        run_step "python -m scripts.run_phase3_textual_similarity --phase2-dir '$PHASE2_DIR/final' --phase3-dir '$OUT_DIR/phase3' --phase1-dir '$PHASE1_DIR' --log-dir '$OUT_DIR/logs'" "Part 3: Textual Similarity Detection (no taxonomy)"
    fi
fi

# Part 4: Core Task Comparisons
run_step "python -m scripts.run_phase3_core_task_comparisons --phase1-dir '$PHASE1_DIR' --phase2-dir '$PHASE2_DIR' --out-dir '$OUT_DIR' --max-candidates 50 --resume --log-level INFO" "Part 4: Core Task Comparisons"

# Part 5: Contribution Analysis
run_step "python -m scripts.run_phase3_contribution_analysis --phase1-dir '$PHASE1_DIR' --phase2-dir '$PHASE2_DIR' --out-dir '$OUT_DIR' --language en --max-candidates 50 --resume --log-level INFO" "Part 5: Contribution Analysis"

# Part 6: Final Report Generation
run_step "python -m scripts.run_phase3_generate_complete_report --out-dir '$OUT_DIR' --log-level INFO" "Part 6: Final Report Generation"

# --- 4. Post-audit Logic ---

echo ""
echo "=== Result Audit ==="
AUDIT_ERROR=0

MD_REPORT="$OUT_DIR/phase3/core_task_survey/reports/core_task_survey.md"
if [ ! -f "$MD_REPORT" ]; then
    echo "Audit FAILED: core_task_survey.md not generated"
    AUDIT_ERROR=1
fi

SUMMARY_JSON="$OUT_DIR/phase3/core_task_comparisons/summary.json"
if [ -f "$SUMMARY_JSON" ]; then
    # Fix: errored is inside 'statistics' object
    ERRORED=$(python3 -c "import json; data=json.load(open('$SUMMARY_JSON')); stats=data.get('statistics', {}); print(stats.get('errored', data.get('errored', 1)))")
    if [ "$ERRORED" -ne 0 ]; then
        echo "Audit FAILED: summary.json has errored count = $ERRORED"
        AUDIT_ERROR=1
    else
        echo "Audit PASSED: errored=0"
    fi
else
    echo "Audit FAILED: summary.json not found"
    AUDIT_ERROR=1
fi

if [ $AUDIT_ERROR -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Phase3 completed successfully and passed audit!"
    echo "=========================================="
    exit 0
else
    echo ""
    echo "=========================================="
    echo "Phase3 completed but failed quality audit"
    echo "=========================================="
    exit 1
fi
