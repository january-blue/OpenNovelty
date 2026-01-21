# OpenNovelty (English)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2601.01576-b31b1b.svg)](https://arxiv.org/abs/2601.01576)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2601.01576)
[![Website](https://img.shields.io/badge/Website-opennovelty.org-success)](https://www.opennovelty.org)

English | [中文](README_zh.md)

**OpenNovelty** is an LLM-powered agentic pipeline for **transparent, evidence-grounded, and verifiable** scholarly novelty assessment.

- Website (public reports): https://www.opennovelty.org  
- Technical report (arXiv): https://arxiv.org/abs/2601.01576  

> 🌟 **If you find OpenNovelty useful, please give us a star on [GitHub](https://github.com/Zhangbeibei1991/OpenNovelty) and an upvote on [Hugging Face](https://huggingface.co/papers/2601.01576) — your support helps us a lot!**

> **Repository status**: The codebase will be incrementally refactored over the next 2 months to improve quality; the current version is a demo.

---

## Why OpenNovelty

Novelty is a key criterion in peer review, but manual evaluation is often constrained by time, subjectivity, and retrieval coverage. By grounding analysis in retrieval and evidence alignment, OpenNovelty provides traceable novelty assessments and helps reduce subjective bias.

---

## System Architecture

![Pipeline Overview](docs/images/pipeline_overview.png)

*Figure: The full OpenNovelty workflow — a four-phase pipeline from paper input to report output.*

### Four-Phase Overview

| Phase | Functionality | Key Input | Key Output | Time | Dependencies |
|:-----:|--------------|----------|-----------|:----:|-------------|
| **I** | Information Extraction | Paper PDF URL | `phase1_extracted.json` | ~1 min | LLM API |
| **II** | Literature Retrieval | Phase 1 outputs | `citation_index.json` | ~10 min | Wispaper API |
| **III** | Deep Analysis | Phase 2 outputs | `phase3_complete_report.json` | ~10 min | LLM API |
| **IV** | Report Generation | Phase 3 outputs | `novelty_report.md/pdf` | ~30 sec | weasyprint |

### Details

**Phase I — Information Extraction**
- Download the PDF and extract full text + metadata
- Use an LLM to extract one core task and 1–3 contribution claims
- Generate 3 retrieval query variants for each task/contribution

**Phase II — Literature Retrieval**
- Semantic retrieval of related papers (via [WisPaper API](https://wispaper.ai) [[paper]](https://huggingface.co/papers/2512.06879))
- Quality filtering (perfect-match, time filtering) and deduplication (`canonical_id` + title normalization)
- Build a citation index (Core Task Top-50, Contribution Top-10)
- ⚠️ **Note**: Phase 2 depends on Wispaper API which is **not yet publicly available**. The API will be opened soon — please stay tuned for updates!

**Phase III — Deep Analysis**
- Build a related-work taxonomy and synthesize a survey
- Textual similarity detection (token-level fuzzy matching; ⚠️ experimental, recommended to skip)
- Full-text comparative verification and novelty judgments (`can_refute` / `cannot_refute` / `unclear`)
- 💡 **Recommended setting**: set `SKIP_TEXTUAL_SIMILARITY=true` in `.env` to skip similarity detection (this experimental module is being updated)

**Phase IV — Report Generation**
- Template-based rendering for Markdown/PDF reports (no LLM calls)
- Unified citation formatting, evidence snippets, and hierarchical structure

---

## Quick Start 🚀

### Requirements

| Type | Requirement |
|------|------------|
| **OS** | Linux (Ubuntu 20.04+) / macOS |
| **Python** | 3.8+ (recommended: 3.10+) |
| **Memory** | 8GB+ |
| **Network** | Access to OpenReview, Wispaper API, and an LLM API |

### 1️⃣ Install Dependencies

```bash
# Ubuntu/Debian system dependencies
sudo apt-get update && sudo apt-get install -y \
  git curl wget \
  libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 \
  libffi-dev libcairo2 libcairo2-dev libgirepository1.0-dev

# Python dependencies
cd /path/to/pnp_oss
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Main dependencies**: `requests`, `openai`, `openreview-py`, `pypdf`, `weasyprint`, `python-dotenv`, `tqdm`

### 2️⃣ Configuration

Create a `.env` file in the project root:

```bash
# ============ LLM API Configuration (Required) ============
export LLM_API_ENDPOINT="https://openrouter.ai/api/v1"           # Example
export LLM_API_KEY="sk-xxxxxxxx"
export LLM_MODEL_NAME="anthropic/claude-sonnet-4.5"           # Example

# ============ Wispaper API (Required for Phase 2) ============
# Token is saved to ~/.wispaper_tokens.json by default (see first-time setup below)

# ============ Phase 3 Configuration (Recommended) ============
export SKIP_TEXTUAL_SIMILARITY="true"           # Skip similarity detection (under development)

# ============ Optional Configuration ============
export HTTP_PROXY="http://127.0.0.1:7893"
export HTTPS_PROXY="http://127.0.0.1:7893"
```

#### 🔐 Wispaper Authentication (Before Running Phase 2)

> ⚠️ **Coming Soon**: Wispaper API is not yet publicly available. The following configuration will be enabled once the API is opened.

<!--
**One-time setup, long-term effective**:

```bash
python scripts/refresh_wispaper_token.py
# → 🌐 Automatically opens a browser for login (register first: https://wispaper.ai)
# → 💾 Token saved to ~/.wispaper_tokens.json
# → 🔄 Auto-refresh enabled, no need to repeat
```

**Custom token path** (optional):

```bash
export WISPAPER_TOKEN_FILE="/your/custom/path/wispaper_tokens.json"
```

**Verify configuration**:

```bash
python -c "from paper_novelty_pipeline.services.wispaper_client import WispaperClient; WispaperClient()"
# Seeing "Loaded token bundle" indicates success
```
-->

### 3️⃣ Example Run (Single Paper)

Using `https://openreview.net/pdf?id=ZgCCDwcGwn` as an example:

```bash
# Phase 1 - Extraction (~1 min)
python scripts/run_phase1_batch.py \
  --papers "https://openreview.net/pdf?id=ZgCCDwcGwn" \
  --out-root output/demo \
  --force-year 2026 \
  2>&1 | tee logs/phase1.log

# Phase 2 - Retrieval (~10 min) ⚠️ Requires Wispaper API (coming soon)
bash scripts/run_phase2_concurrent.sh \
  openreview_ZgCCDwcGwn_20260118 \
  --base-dir output/demo \
  2>&1 | tee logs/phase2.log

# Phase 3 - Deep analysis (~10 min)
bash scripts/run_phase3_all.sh \
  output/demo/openreview_ZgCCDwcGwn_20260118 \
  2>&1 | tee logs/phase3.log

# Phase 4 - Report generation (~30 sec)
bash scripts/run_phase4.sh \
  output/demo/openreview_ZgCCDwcGwn_20260118 \
  2>&1 | tee logs/phase4.log

# View the result
cat output/demo/openreview_ZgCCDwcGwn_20260118/phase4/novelty_report.md
```

> 💡 **Argument notes**: `--papers` paper URL | `--out-root` output directory | `--force-year` force year | `--base-dir` search directory | `| tee` save logs

### 4️⃣ Batch Processing

```bash
# Create a paper list
cat > papers.txt << EOF
https://openreview.net/pdf?id=PAPER_ID_1
https://openreview.net/pdf?id=PAPER_ID_2
https://openreview.net/pdf?id=PAPER_ID_3
EOF

# Phase 1: Batch extraction
python scripts/run_phase1_batch.py \
  --paper-file papers.txt \
  --out-root output/batch \
  --force-year 2026

# Phase 2: Batch retrieval (auto-discover all papers) ⚠️ Requires Wispaper API (coming soon)
bash scripts/run_phase2_concurrent.sh \
  --base-dir output/batch \
  --auto-discover \           # Auto-discover all papers with Phase 1 completed under base-dir
  --max-workers 10            # Concurrency (default 10; adjust based on machine capacity)

# Phase 3+4: Batch analysis and report generation
bash scripts/run_phase3_phase4_serial_pending.sh output/batch
```

> 💡 **Argument notes**:
>
> * `--paper-file`: a list file (one URL per line)
> * `--auto-discover`: auto-scan all papers that need processing
> * `--max-workers`: number of parallel workers (Phase 2 makes concurrent API calls)

### 5️⃣ Common Commands

| Command                                        | Purpose                               |
| ---------------------------------------------- | ------------------------------------- |
| `python scripts/refresh_wispaper_token.py`     | Refresh Wispaper token ⚠️ (coming soon) |
| `python scripts/run_phase1_batch.py --help`    | Show Phase 1 help                     |
| `bash scripts/run_phase2_concurrent.sh --help` | Show Phase 2 help ⚠️ (coming soon)     |
| `cat logs/phase2.log \| grep ERROR`            | Locate error logs                     |

---

## Technical Reference

### Script Entrypoints (`scripts/`)

| Script                                | Function                    | Use Case                               |       Time      |
| ------------------------------------- | --------------------------- | -------------------------------------- | :-------------: |
| `run_phase1_batch.py`                 | Information extraction      | Single / batch                         |  ~1 min / paper |
| `run_phase2_concurrent.sh`            | Literature retrieval ⚠️      | Single / batch (coming soon)           | ~10 min / paper |
| `run_phase2_only.py`                  | Retrieval (single paper) ⚠️  | Coming soon                            | ~10 min / paper |
| `run_phase3_all.sh`                   | Deep analysis (7 sub-steps) | Single / batch                         | ~10 min / paper |
| `run_phase4.sh`                       | Report generation           | Single paper                           | ~30 sec / paper |
| `run_phase3_phase4_serial_pending.sh` | Batch completion            | Auto-discover papers with Phase 2 done |        -        |
| `refresh_wispaper_token.py`           | Token refresh ⚠️             | Coming soon                            |     ~10 sec     |

### Directory Layout

```bash
output/<run>/<paper_id>/
├── phase1/                         # Phase 1 outputs
│   ├── phase1_extracted.json       # ⭐ Core task and contributions (with query variants)
│   ├── paper.json                  # Paper metadata
│   ├── pub_date.json               # Publication date
│   ├── fulltext_raw.txt            # Raw full text from PDF
│   ├── fulltext_cleaned.txt        # Cleaned full text
│   ├── body_for_core_task.txt      # Text relevant to the core task
│   ├── body_for_claims.txt         # Text relevant to contribution claims
│   └── raw_llm_responses/          # Raw LLM responses (reproducibility)
│
├── phase2/final/                   # Phase 2 outputs
│   ├── citation_index.json         # ⭐ Citation index (required for Phase 3)
│   ├── core_task_perfect_top50.json # Core task candidates Top-50
│   ├── contribution_*_perfect_top10.json # Contribution candidates Top-10
│   ├── stats.json                  # Search statistics
│   └── raw_responses/              # Raw API responses
│
├── phase3/                         # Phase 3 outputs
│   ├── phase3_complete_report.json # ⭐ Full analysis report (required for Phase 4)
│   ├── core_task_survey/           # Taxonomy and survey synthesis
│   ├── core_task_comparisons/      # Core-task comparisons
│   ├── contribution_analysis/      # Contribution-level analysis
│   ├── textual_similarity_detection/ # Textual similarity detection
│   ├── cached_paper_texts/         # Cached full texts
│   └── raw_llm_responses/          # Raw LLM responses
│
└── phase4/                         # Phase 4 outputs
    ├── novelty_report.md           # ⭐ Markdown report
    └── novelty_report.pdf          # PDF report (requires weasyprint)
```

> ⭐ Files marked are **required inputs** for the next phase.

---

## Troubleshooting 🔧

### Common Issues

| Issue                            | Symptom                              | Solution                                                                     |
| -------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------- |
| **Wispaper token expired**       | `401 Unauthorized` / `Token expired` | ⚠️ Wispaper API coming soon                                                   |
| **PDF download failure**         | `ConnectionError` / `Timeout`        | Check network, URL validity, and proxy settings                              |
| **LLM API call failure**         | `API Error` / `Invalid key`          | Check `LLM_API_KEY`, `LLM_MODEL_NAME`, and quota                             |
| **PDF generation failure**       | `weasyprint error`                   | `sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0` |
| **Phase 3 appears stuck**        | No output for > 5 minutes            | Usually normal; a single sub-step can take 2–5 minutes                       |
| **Phase 1 year detection fails** | `Year extraction failed`             | Use `--force-year YYYY`                                                      |

### Debug Tips

**View detailed logs**:

```bash
# Locate errors
cat logs/phase2.log | grep -i error

# Inspect LLM calls
cat logs/phase3.log | grep "LLM call"

# Check generated files
ls -lh output/demo/openreview_XXX/phase2/final/
```

**Re-run failed sub-steps** (Phase 3):

```bash
# Re-run taxonomy generation only
python -m scripts.run_phase3_taxonomy \
  --phase2-dir output/demo/openreview_XXX/phase2 \
  --out-dir output/demo/openreview_XXX \
  --log-level INFO

# Re-run comparison analysis only
python -m scripts.run_phase3_core_task_comparisons \
  --phase1-dir output/demo/openreview_XXX/phase1 \
  --phase2-dir output/demo/openreview_XXX/phase2 \
  --out-dir output/demo/openreview_XXX \
  --resume \
  --log-level INFO
```

**Verify configuration**:

```bash
# Check environment variables
echo "LLM_API_KEY: ${LLM_API_KEY:0:10}..."
echo "LLM_MODEL_NAME: $LLM_MODEL_NAME"

# Validate Wispaper token (⚠️ coming soon)
# python -c "from paper_novelty_pipeline.services.wispaper_client import WispaperClient; client = WispaperClient(); print('Token valid!')"

# Check Phase 1 output
cat output/demo/openreview_XXX/phase1/phase1_extracted.json | jq '.core_task'
```

---

## Citation

```bibtex
@article{zhang2026opennovelty,
  title={OpenNovelty: An LLM-powered Agentic System for Verifiable Scholarly Novelty Assessment},
  author={Zhang, Ming and Tan, Kexin and Huang, Yueyuan and Shen, Yujiong and Ma, Chunchun and Ju, Li and Zhang, Xinran and Wang, Yuhui and Jing, Wenqing and Deng, Jingyi and others},
  journal={arXiv preprint arXiv:2601.01576},
  year={2026}
}
```

---

## License

This project is released under the **Apache License 2.0**. See [`LICENSE`](LICENSE).

---

## Disclaimer

All reports are generated based on retrieved literature and LLM outputs. Coverage is constrained by retrieval recall, and conclusions are provided for reference only and should not be treated as final novelty judgments. OpenNovelty aims to provide transparent, evidence-grounded assistance rather than replacing human review.

---

## Contact

* Ming Zhang: [mingzhang23@m.fudan.edu.cn](mailto:mingzhang23@m.fudan.edu.cn)
* Kexin Tan: [kxtan18@fudan.edu.cn](mailto:kxtan18@fudan.edu.cn) or [bluellatan@gmail.com](mailto:bluellatan@gmail.com)