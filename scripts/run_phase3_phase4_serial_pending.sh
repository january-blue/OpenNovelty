#!/usr/bin/env bash
# Run Phase3+Phase4 in serial for all pending papers in a directory
# Usage: ./scripts/run_phase3_phase4_serial_pending.sh [BASE_DIR] [LOGDIR]
# Examples:
#   ./scripts/run_phase3_phase4_serial_pending.sh output/test_overlap_20251225
#   ./scripts/run_phase3_phase4_serial_pending.sh output/iclr2026_1-200
#   ./scripts/run_phase3_phase4_serial_pending.sh output/test
# 
# Environment variables:
#   SLEEP_BETWEEN_PHASES: seconds to wait between phase3 and phase4 (default: 20)
#   SLEEP_BETWEEN_PAPERS: seconds to wait between papers (default: 10)
set -euo pipefail

BASE_DIR="${1:-output/iclr2026_1-200}"
LOGDIR="${2:-logs/phase3_phase4_serial_$(date +%Y%m%d_%H%M%S)}"
SLEEP_BETWEEN_PHASES="${SLEEP_BETWEEN_PHASES:-20}"
SLEEP_BETWEEN_PAPERS="${SLEEP_BETWEEN_PAPERS:-10}"

# Ensure SSL verification works reliably in conda-based Python by pinning to the
# system CA bundle. This avoids long urllib3 retries on CERTIFICATE_VERIFY_FAILED
# for some hosts when conda's bundled CA is missing intermediates.
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"

mkdir -p "$LOGDIR"

# Best-effort: ensure clash is up if user relies on proxy
if [ -x "$HOME/clash/start_clash_tmux.sh" ]; then
  "$HOME/clash/start_clash_tmux.sh" >/dev/null 2>&1 || true
fi

# Snapshot targets at start: Phase2 complete AND Phase3 missing
python - <<PY > "$LOGDIR/targets.txt"
from pathlib import Path
base = Path("$BASE_DIR")
rows=[]
# Scan ALL subdirectories in base_dir
for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    # Check if this directory contains a phase1 subdirectory (the signature of a paper run)
    if not (d/'phase1').exists():
        continue
        
    p2_ok = (d/'phase2/final/stats.json').exists() and (d/'phase2/final/citation_index.json').exists()
    p3_ok = (d/'phase3/phase3_complete_report.json').exists()
    if p2_ok and (not p3_ok):
        rows.append(str(d))
print("\n".join(rows))
PY

total=$(grep -c . "$LOGDIR/targets.txt" || true)
{
  echo "[start] $(date -Is)"
  echo "base_dir=$BASE_DIR"
  echo "targets=$total"
  echo "sleep_between_phases=$SLEEP_BETWEEN_PHASES"
  echo "sleep_between_papers=$SLEEP_BETWEEN_PAPERS"
  echo "logdir=$LOGDIR"
} | tee -a "$LOGDIR/master.log"

if [ "$total" = "0" ]; then
  echo "[done] $(date -Is) nothing to do" | tee -a "$LOGDIR/master.log"
  exit 0
fi

i=0
while IFS= read -r out_dir; do
  [ -z "$out_dir" ] && continue
  i=$((i+1))
  name=$(basename "$out_dir")

  echo "[paper $i/$total] $(date -Is) START $out_dir" | tee -a "$LOGDIR/master.log"

  # Make sure phase3 dir exists (some scripts assume it)
  mkdir -p "$out_dir/phase3"

  if bash scripts/run_phase3_all.sh "$out_dir" > "$LOGDIR/${name}_phase3.log" 2>&1; then
    echo "[paper $i/$total] $(date -Is) Phase3 OK $out_dir" | tee -a "$LOGDIR/master.log"
  else
    echo "[paper $i/$total] $(date -Is) Phase3 FAIL $out_dir (see $LOGDIR/${name}_phase3.log)" | tee -a "$LOGDIR/master.log"
    echo "[paper $i/$total] $(date -Is) SLEEP $SLEEP_BETWEEN_PAPERS" | tee -a "$LOGDIR/master.log"
    sleep "$SLEEP_BETWEEN_PAPERS"
    continue
  fi

  echo "[paper $i/$total] $(date -Is) SLEEP between phases $SLEEP_BETWEEN_PHASES" | tee -a "$LOGDIR/master.log"
  sleep "$SLEEP_BETWEEN_PHASES"

  if bash scripts/run_phase4.sh "$out_dir" > "$LOGDIR/${name}_phase4.log" 2>&1; then
    echo "[paper $i/$total] $(date -Is) Phase4 OK $out_dir" | tee -a "$LOGDIR/master.log"
  else
    echo "[paper $i/$total] $(date -Is) Phase4 FAIL $out_dir (see $LOGDIR/${name}_phase4.log)" | tee -a "$LOGDIR/master.log"
  fi

  echo "[paper $i/$total] $(date -Is) DONE $out_dir" | tee -a "$LOGDIR/master.log"
  echo "[paper $i/$total] $(date -Is) SLEEP between papers $SLEEP_BETWEEN_PAPERS" | tee -a "$LOGDIR/master.log"
  sleep "$SLEEP_BETWEEN_PAPERS"

done < "$LOGDIR/targets.txt"

echo "[done] $(date -Is) all finished" | tee -a "$LOGDIR/master.log"
