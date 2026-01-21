#!/bin/bash
# Run Phase4 Report Generation
# Usage: ./scripts/run_phase4.sh [OUTPUT_DIR]
# Example: ./scripts/run_phase4.sh output/claim_based/openreview_ZgCCDwcGwn_20251208

export PYTHONPATH=/root/paper_novelty_pipeline

set -e

OUTPUT_DIR="${1:-output/claim_based/openreview_ZgCCDwcGwn_20251208}"
PHASE3_REPORT="$OUTPUT_DIR/phase3/phase3_complete_report.json"
PHASE4_DIR="$OUTPUT_DIR/phase4"

echo "=========================================="
echo "Running Phase4 Report Generation"
echo "=========================================="

# Check prerequisites
if [ ! -f "$PHASE3_REPORT" ]; then
    echo "❌ Error: Phase3 report not found: $PHASE3_REPORT"
    exit 1
fi

echo ""
echo "Input: $PHASE3_REPORT"
echo "Output: $PHASE4_DIR"
echo ""

# Generate Markdown + PDF report
python -m scripts.run_phase4_lightweight \
  --phase3-report "$PHASE3_REPORT" \
  --output-dir "$PHASE4_DIR" \
  --log-level INFO

echo ""
echo "=========================================="
echo "✅ Phase4 Report Generation Complete!"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh "$PHASE4_DIR"/*.md "$PHASE4_DIR"/*.pdf 2>/dev/null || echo "Check $PHASE4_DIR for output files"

