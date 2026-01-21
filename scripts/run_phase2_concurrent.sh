#!/bin/bash
# ============================================================================
# Phase2 Concurrent Processing - Generic Script
# ============================================================================
# Usage:
#   1. Interactive input:     ./run_phase2_concurrent.sh [--base-dir DIR]
#   2. Read from file:        ./run_phase2_concurrent.sh papers.txt [--base-dir DIR]
#   3. CLI arguments:         ./run_phase2_concurrent.sh paper1 paper2 [--base-dir DIR]
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # Switch to project root directory

# Auto-load environment variables
if [ -f ".env" ]; then
  source .env
  echo "✓ Loaded environment variables from .env"
else
  echo "⚠️  Warning: .env file not found. Please ensure environment variables are set."
fi

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default configuration
PAPERS=()
BASE_DIR="output"  # Default search directory
MAX_WORKERS=10     # Default concurrency
AUTO_DISCOVER=false  # Whether to auto-discover papers

# ============================================================================
# Function: Show usage
# ============================================================================
show_usage() {
  cat << EOF
${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${CYAN}  Phase2 Concurrent Processing Tool${NC}
${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${YELLOW}Usage:${NC}

  ${GREEN}1. Interactive paper input:${NC}
     $ ./run_phase2_concurrent.sh [--base-dir DIR]
     
  ${GREEN}2. Read paper list from a file:${NC}
     $ ./run_phase2_concurrent.sh papers.txt [--base-dir DIR]
     
  ${GREEN}3. Specify papers directly in CLI:${NC}
     $ ./run_phase2_concurrent.sh paper1_id paper2_id [--base-dir DIR]
     
  ${GREEN}4. Auto-discover all papers:${NC}
     $ ./run_phase2_concurrent.sh --base-dir DIR --auto-discover

${YELLOW}Arguments:${NC}
  --base-dir DIR       Base directory to search for papers (default: output)
                       Examples: --base-dir output/test_batch
                                 --base-dir output/iclr2026_1-25
  
  --auto-discover      Auto-discover all papers that have phase1 under base-dir
                       No manual paper list required
  
  --max-workers N      Maximum concurrency (default: 10)
  
  -h, --help           Show this help message

${YELLOW}File format example (papers.txt):${NC}
  openreview_yRtgZ1K8hO_20251209
  openreview_zrFnwRHuQo_20251209
  # Lines starting with # will be ignored
  openreview_VKGTGGcwl6_20251209

${YELLOW}Examples:${NC}
  # Process 3 papers in a test directory
  $ ./run_phase2_concurrent.sh --base-dir output/test_batch paper1 paper2 paper3
  
  # Read from file with a specified directory
  $ ./run_phase2_concurrent.sh papers.txt --base-dir output/iclr2026_1-25
  
  # Auto-discover and process all papers under a directory (recommended!)
  $ ./run_phase2_concurrent.sh --base-dir output/test_batch/script_test --auto-discover
  
  # Limit concurrency to 5
  $ ./run_phase2_concurrent.sh --base-dir output/test --max-workers 5 papers.txt

${YELLOW}Notes:${NC}
  - Paper IDs should be full directory names (e.g., openreview_xxx_20251209)
  - Using --base-dir avoids mixing data across different batches
  - The script searches for each paper's phase1 directory under base-dir

${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
EOF
}

# ============================================================================
# Function: Read paper list from file
# ============================================================================
read_papers_from_file() {
  local file="$1"
  
  if [ ! -f "$file" ]; then
    echo -e "${RED}Error: file does not exist: $file${NC}"
    exit 1
  fi
  
  echo -e "${YELLOW}Reading paper list from file: $file${NC}"
  
  while IFS= read -r line || [ -n "$line" ]; do
    # Trim leading/trailing spaces
    line=$(echo "$line" | xargs)
    
    # Skip empty lines and comments
    if [ -z "$line" ] || [[ "$line" =~ ^# ]]; then
      continue
    fi
    
    PAPERS+=("$line")
  done < "$file"
  
  if [ ${#PAPERS[@]} -eq 0 ]; then
    echo -e "${RED}Error: no valid paper IDs found in the file${NC}"
    exit 1
  fi
}

# ============================================================================
# Function: Interactive paper input
# ============================================================================
interactive_input() {
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}  Interactive Paper Input${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
  echo -e "${YELLOW}Enter paper IDs (one per line). Submit an empty line to finish:${NC}"
  echo -e "${YELLOW}Example: openreview_yRtgZ1K8hO_20251209${NC}"
  echo ""
  
  local count=1
  while true; do
    read -p "$(echo -e ${GREEN}Paper$count:${NC} )" paper_id
    
    # End input on empty line
    if [ -z "$paper_id" ]; then
      if [ ${#PAPERS[@]} -eq 0 ]; then
        echo -e "${RED}Error: you must enter at least one paper ID${NC}"
        continue
      fi
      break
    fi
    
    # Trim leading/trailing spaces
    paper_id=$(echo "$paper_id" | xargs)
    
    # Skip empty lines and comments
    if [ -z "$paper_id" ] || [[ "$paper_id" =~ ^# ]]; then
      continue
    fi
    
    PAPERS+=("$paper_id")
    count=$((count + 1))
  done
  
  echo ""
}

# ============================================================================
# Function: Auto-discover papers
# ============================================================================
auto_discover_papers() {
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}  Auto-discover Papers${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
  echo -e "${YELLOW}Scanning directory: ${BASE_DIR}${NC}"
  echo ""
  
  # Check if base-dir exists
  if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}Error: base directory does not exist: $BASE_DIR${NC}"
    exit 1
  fi
  
  local count=0
  
  # Scan all subdirectories under base-dir
  for dir in "$BASE_DIR"/*/; do
    # Skip if not a directory (if no subdirectories, */ stays literal)
    [ -d "$dir" ] || continue
    
    # Directory name (strip path and trailing slash)
    local dirname=$(basename "$dir")
    
    # Check for phase1
    if [ -d "${dir}phase1" ]; then
      PAPERS+=("$dirname")
      echo -e "  ${GREEN}✓${NC} Found: $dirname"
      count=$((count + 1))
    fi
  done
  
  # If nested structure (e.g., output/test_batch/script_test/), scan one more level
  if [ $count -eq 0 ]; then
    echo -e "${YELLOW}No papers found in direct children of $BASE_DIR. Scanning second-level directories...${NC}"
    for subdir in "$BASE_DIR"/*/*/; do
      [ -d "$subdir" ] || continue
      local dirname=$(basename "$subdir")
      if [ -d "${subdir}phase1" ]; then
        PAPERS+=("$dirname")
        echo -e "  ${GREEN}✓${NC} Found: $dirname"
        count=$((count + 1))
      fi
    done
  fi
  
  echo ""
  
  if [ $count -eq 0 ]; then
    echo -e "${RED}Error: no paper directories containing phase1 were found under $BASE_DIR${NC}"
    echo -e "${YELLOW}Hint: run Phase1 first, or check your --base-dir argument${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}Discovered $count papers in total${NC}"
  echo ""
}

# ============================================================================
# Function: Validate paper directories exist (search within the specified base-dir)
# ============================================================================
validate_papers() {
  echo -e "${YELLOW}Validating paper directories (search scope: ${BASE_DIR})...${NC}"
  
  # Check if base-dir exists
  if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}Error: base directory does not exist: $BASE_DIR${NC}"
    exit 1
  fi
  
  declare -gA PAPER_PATHS  # Global associative array storing full paths for each paper
  local invalid_count=0
  
  for paper in "${PAPERS[@]}"; do
    local found=false
    local phase1_dir="${BASE_DIR}/${paper}/phase1"
    
    # Direct search under base-dir
    if [ -d "$phase1_dir" ]; then
      PAPER_PATHS["$paper"]="${BASE_DIR}/${paper}"
      echo -e "  ${GREEN}✓${NC} $paper"
      echo -e "      Phase1: $phase1_dir"
      found=true
    else
      # Try searching in subdirectories of base-dir (one level)
      for sub_dir in "${BASE_DIR}"/*/; do
        phase1_dir="${sub_dir}${paper}/phase1"
        if [ -d "$phase1_dir" ]; then
          PAPER_PATHS["$paper"]="${sub_dir}${paper}"
          echo -e "  ${GREEN}✓${NC} $paper"
          echo -e "      Phase1: $phase1_dir"
          found=true
          break
        fi
      done
    fi
    
    if [ "$found" = false ]; then
      echo -e "  ${RED}✗${NC} $paper - Phase1 directory does not exist"
      echo -e "      Searched path: ${BASE_DIR}/${paper}/phase1"
      invalid_count=$((invalid_count + 1))
    fi
  done
  
  if [ $invalid_count -gt 0 ]; then
    echo ""
    echo -e "${RED}Error: Phase1 directory is missing for ${invalid_count} paper(s)${NC}"
    echo -e "${YELLOW}Hints:${NC}"
    echo -e "  1. Check whether the paper IDs are correct"
    echo -e "  2. Verify the --base-dir argument (current: ${BASE_DIR})"
    echo -e "  3. Run Phase1: python scripts/run_phase1_only.py --openreview-id <id>"
    exit 1
  fi
  
  echo ""
}

# ============================================================================
# Function: Filter out papers that already completed Phase2 (resume support)
# ============================================================================
filter_completed_papers() {
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}  Checking Completed Papers (Resume Support)${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
  
  local completed_count=0
  declare -a PAPERS_TO_RUN
  
  for paper in "${PAPERS[@]}"; do
    local paper_full_path="${PAPER_PATHS[$paper]}"
    local phase2_output="${paper_full_path}/phase2/final/core_task_perfect_top50.json"
    
    # Check whether the key Phase2 output exists
    if [ -f "$phase2_output" ]; then
      echo -e "  ${GREEN}✓${NC} $paper - ${GREEN}already completed, skipping${NC}"
      completed_count=$((completed_count + 1))
    else
      echo -e "  ${YELLOW}○${NC} $paper - pending"
      PAPERS_TO_RUN+=("$paper")
    fi
  done
  
  echo ""
  
  if [ $completed_count -gt 0 ]; then
    echo -e "${GREEN}Found $completed_count completed paper(s); they will be skipped${NC}"
  fi
  
  if [ ${#PAPERS_TO_RUN[@]} -eq 0 ]; then
    echo -e "${GREEN}All papers have already completed Phase2!${NC}"
    echo ""
    exit 0
  fi
  
  echo -e "${YELLOW}Papers to process: ${#PAPERS_TO_RUN[@]} / ${#PAPERS[@]}${NC}"
  echo ""
  
  # Update PAPERS to only include those needing processing
  PAPERS=("${PAPERS_TO_RUN[@]}")
}

# ============================================================================
# Function: Run Phase2 for a single paper
# ============================================================================
run_paper_phase2() {
  local paper_id=$1
  local log_file="$LOG_DIR/${paper_id}.log"
  local paper_full_path="${PAPER_PATHS[$paper_id]}"
  
  echo "[$(date '+%H:%M:%S')] Starting: $paper_id" | tee -a "$LOG_DIR/master.log"
  echo "               Path: ${paper_full_path}" | tee -a "$LOG_DIR/master.log"
  
  # Set LOG_FILE=none to disable global file logging and prevent multi-process rotation deadlocks.
  # Added PYTHONPATH=. to ensure local code priority and avoid loading old code from other paths.
  # Individual logs are captured via shell redirection below.
  # Note: Stagnation timeout (socket timeout) is handled internally by the Python client.
  env PYTHONPATH=.:$PYTHONPATH LOG_FILE=none python scripts/run_phase2_only.py \
    --phase1-dir "${paper_full_path}/phase1" \
    --out-dir "${paper_full_path}" \
    --log-level INFO \
    > "$log_file" 2>&1
  
  local exit_code=$?
  
  if [ $exit_code -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✓ Done: $paper_id" | tee -a "$LOG_DIR/master.log"
  else
    echo "[$(date '+%H:%M:%S')] ✗ Failed: $paper_id (exit code: $exit_code)" | tee -a "$LOG_DIR/master.log"
  fi
  
  return $exit_code
}

# ============================================================================
# Main
# ============================================================================

# Header
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Phase2 Concurrent Processing Tool${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_usage
      exit 0
      ;;
    --base-dir)
      if [ -z "$2" ]; then
        echo -e "${RED}Error: --base-dir requires a directory${NC}"
        exit 1
      fi
      BASE_DIR="$2"
      shift 2
      ;;
    --max-workers)
      if [ -z "$2" ]; then
        echo -e "${RED}Error: --max-workers requires a number${NC}"
        exit 1
      fi
      MAX_WORKERS="$2"
      shift 2
      ;;
    --auto-discover)
      AUTO_DISCOVER=true
      shift
      ;;
    -*)
      echo -e "${RED}Error: unknown option $1${NC}"
      show_usage
      exit 1
      ;;
    *)
      # If it's a file, read from the file
      if [ -f "$1" ] && [ ${#PAPERS[@]} -eq 0 ]; then
        read_papers_from_file "$1"
        shift
      else
        # Otherwise, treat as a paper ID
        PAPERS+=("$1")
        shift
      fi
      ;;
  esac
done

# Auto-discover mode
if [ "$AUTO_DISCOVER" = true ]; then
  if [ ${#PAPERS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Warning: --auto-discover cannot be used together with manual paper IDs${NC}"
    echo -e "${YELLOW}Ignoring manually specified papers and using auto-discover mode${NC}"
    echo ""
    PAPERS=()
  fi
  auto_discover_papers
elif [ ${#PAPERS[@]} -eq 0 ]; then
  # If no papers and not auto-discover, fall back to interactive input
  interactive_input
fi

echo -e "${CYAN}Configuration:${NC}"
echo -e "  Base search directory: ${YELLOW}${BASE_DIR}${NC}"
echo -e "  Max workers: ${YELLOW}${MAX_WORKERS}${NC}"
echo -e "  Auto-discover: ${YELLOW}${AUTO_DISCOVER}${NC}"
echo -e "  Total papers: ${YELLOW}${#PAPERS[@]}${NC}"
echo ""

# Validate paper directories
validate_papers

# Filter completed papers (resume support)
filter_completed_papers

# Show papers to process
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Papers to Process (${#PAPERS[@]} paper(s))${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
for i in "${!PAPERS[@]}"; do
  echo -e "  $((i+1)). ${PAPERS[$i]}"
done
echo ""

# Confirm
read -p "$(echo -e ${YELLOW}Confirm running Phase2 concurrently? [y/N]:${NC} )" confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo -e "${YELLOW}Cancelled${NC}"
  exit 0
fi
echo ""

# Create log directory
LOG_DIR="logs/phase2_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo -e "${YELLOW}Log directory: $LOG_DIR${NC}"
echo ""

# ============================================================================
# Pre-refresh Wispaper Token (avoid lock contention in parallel workers)
# ============================================================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Pre-refresh Wispaper Token${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Note: Refresh the token before starting concurrent workers to avoid multiple workers contending on the token lock${NC}"
echo ""

python -c "
import sys
try:
    from paper_novelty_pipeline.services.wispaper_client import WispaperClient
    print('🔄 Refreshing token...')
    client = WispaperClient()
    print('✅ Token refreshed. Validity is sufficient.')
    print('   Concurrent workers can safely use the new token.')
    sys.exit(0)
except Exception as e:
    print(f'⚠️  Pre-refresh failed: {e}')
    print('   This is not fatal; workers will refresh the token at runtime if needed.')
    sys.exit(0)
" 2>&1

echo ""
echo -e "${GREEN}Starting Phase2 concurrent execution...${NC}"
echo ""

START_TIME=$(date +%s)

# Launch background jobs (bounded concurrency)
PIDS=()
declare -A PAPER_FOR_PID  # Record the paper ID for each PID
ACTIVE_COUNT=0
PAPER_INDEX=0
SUCCESS_COUNT=0
FAIL_COUNT=0

for paper in "${PAPERS[@]}"; do
  # If max concurrency reached, wait until any job finishes
  while [ $ACTIVE_COUNT -ge $MAX_WORKERS ]; do
    for pid in "${!PIDS[@]}"; do
      if ! kill -0 $pid 2>/dev/null; then
        # Process ended; collect status
        if wait $pid; then
          SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
          FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        unset PIDS[$pid]
        unset PAPER_FOR_PID[$pid]
        ACTIVE_COUNT=$((ACTIVE_COUNT - 1))
      fi
    done
    sleep 1
  done
  
  # Start new job
  run_paper_phase2 "$paper" &
  new_pid=$!
  PIDS[$new_pid]=1
  PAPER_FOR_PID[$new_pid]="$paper"
  ACTIVE_COUNT=$((ACTIVE_COUNT + 1))
  PAPER_INDEX=$((PAPER_INDEX + 1))
  echo -e "${YELLOW}▶${NC} Started worker [$PAPER_INDEX/${#PAPERS[@]}]: $paper (PID: $new_pid, Active: $ACTIVE_COUNT/$MAX_WORKERS)"
  sleep 2  # Stagger startup to reduce simultaneous access to shared resources (e.g., token file)
done

echo ""
echo -e "${YELLOW}Waiting for all workers to finish...${NC}"
echo ""

# Wait for all background jobs
for pid in "${!PIDS[@]}"; do
  paper="${PAPER_FOR_PID[$pid]}"
  
  if wait $pid; then
    echo -e "  ✓ $paper - ${GREEN}success${NC}"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    echo -e "  ✗ $paper - ${RED}failed${NC}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Phase2 Concurrent Processing Completed!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Total papers: ${#PAPERS[@]}"
echo -e "  Total time: ${MINUTES}m${SECONDS}s"
echo -e "  Success: ${GREEN}${SUCCESS_COUNT}${NC}/${#PAPERS[@]}"
echo -e "  Failed: ${RED}${FAIL_COUNT}${NC}/${#PAPERS[@]}"
echo ""
echo -e "${YELLOW}Detailed logs are in: $LOG_DIR/${NC}"
echo ""

# Check output files
echo -e "${YELLOW}Checking Phase2 output files...${NC}"
for paper in "${PAPERS[@]}"; do
  paper_full_path="${PAPER_PATHS[$paper]}"
  PHASE2_DIR="${paper_full_path}/phase2"
  if [ -f "$PHASE2_DIR/final/core_task_perfect_top50.json" ]; then
    FILE_COUNT=$(find "$PHASE2_DIR/final" -name "*.json" 2>/dev/null | wc -l)
    echo -e "  ✓ $paper - ${FILE_COUNT} output file(s)"
  else
    echo -e "  ✗ $paper - ${RED}missing output file(s)${NC}"
  fi
done

echo ""

if [ $FAIL_COUNT -eq 0 ]; then
  echo -e "${GREEN}✨ All papers processed successfully!${NC}"
  exit 0
else
  echo -e "${YELLOW}⚠️  ${FAIL_COUNT} paper(s) failed. Please check logs under $LOG_DIR/${NC}"
  exit 1
fi