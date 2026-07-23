# 🤖 LoRA Code Review Bot

Fine-tune **Qwen2.5-Coder** using **QLoRA** on real GitHub pull request diffs and developer review comments to build an inline code critique engine.

Unlike standard code generation models, this bot is trained to **review and critique code** — identifying security flaws, performance bottlenecks, resource leaks, and anti-patterns in git diffs.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.46-orange?logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

| Component | Description |
|-----------|-------------|
| **Task** | Code critique and inline review (not code generation) |
| **Base Model** | `Qwen/Qwen2.5-Coder-3B-Instruct` (or 0.5B / 1.5B) |
| **Fine-Tuning** | QLoRA — 4-bit NF4 quantization with PEFT LoRA adapters |
| **Dataset** | 98 real PR review pairs scraped from FastAPI, Flask, Requests, Starlette, SQLModel |
| **Demo** | Gradio web app with side-by-side A/B comparison (Base vs. Fine-Tuned) |
| **GPU Requirement** | Google Colab T4 (free tier) — ~15 min training time |

---

## Repository Structure

```
lora-code-review-bot/
├── config.py                   # Model IDs, LoRA hyperparams, system prompt
├── requirements.txt            # Pinned dependency matrix
├── app.py                      # Gradio web interface with A/B comparison
├── LICENSE                     # MIT License
├── .gitignore
├── data/
│   ├── raw_reviews.json        # Scraped GitHub PR review pairs (98 samples)
│   ├── code_review_train.jsonl # ChatML-formatted training set (84 samples)
│   └── code_review_val.jsonl   # ChatML-formatted validation set (14 samples)
├── src/
│   ├── __init__.py
│   ├── scraper.py              # GitHub API scraper + offline seed dataset
│   ├── dataset.py              # Converts diffs → ChatML instruction format
│   ├── train.py                # QLoRA fine-tuning with SFTTrainer
│   └── inference.py            # Inference engine with Base vs. LoRA comparison
└── notebooks/
    └── train_code_review_lora.ipynb  # 1-click Google Colab notebook
```

---

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/<your-username>/lora-code-review-bot.git
cd lora-code-review-bot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Dataset

Scrapes real PR review comments from open-source repos (works without GitHub token):

```bash
python src/scraper.py
python src/dataset.py
```

### 3. Train with QLoRA

**On Google Colab (recommended):** Upload `notebooks/train_code_review_lora.ipynb` and set runtime to T4 GPU.

**Locally (requires CUDA GPU):**

```bash
python src/train.py --epochs 3 --batch_size 2
```

**Dry run (verify setup without training):**

```bash
python src/train.py --dry_run
```

### 4. Launch Web Demo

```bash
python app.py
```

Open `http://localhost:7860` — paste any git diff and get inline code review feedback.

---

## How It Works

### Data Pipeline

```
GitHub API (PR comments + diff hunks)
        │
        ▼
  src/scraper.py ──► data/raw_reviews.json (98 samples)
        │
        ▼
  src/dataset.py ──► data/code_review_train.jsonl (ChatML format)
                     data/code_review_val.jsonl
```

Each sample is formatted as:

```json
{
  "messages": [
    {"role": "system", "content": "You are a senior code reviewer..."},
    {"role": "user", "content": "File: `auth.py`\n```diff\n+if user.password == raw_password:...\n```"},
    {"role": "assistant", "content": "⚠️ Plaintext password comparison detected..."}
  ]
}
```

### QLoRA Training

- **Quantization**: 4-bit NF4 with double quantization (`bitsandbytes`)
- **LoRA Config**: `r=16`, `alpha=32`, `dropout=0.05`
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Trainer**: `trl.SFTTrainer` with gradient accumulation

### Inference & A/B Testing

The `CodeReviewer` class supports:
- **Single review**: Generate inline critique for any diff
- **A/B comparison**: Disable LoRA adapter → run base model → re-enable adapter → compare outputs side-by-side

---

## Example

**Input diff:**
```diff
@@ -10,4 +10,6 @@ def get_user_by_email(db, email):
+    sql = f"SELECT * FROM users WHERE email = '{email}'"
+    return db.execute(sql).fetchone()
```

**Model output:**
> ⛔ **SQL Injection Vulnerability**: F-string interpolation into raw SQL allows attacker-controlled `email` values to execute arbitrary queries. Use parameterized queries: `db.execute("SELECT * FROM users WHERE email = ?", (email,))`.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `transformers` | 4.46.3 | Model loading and tokenization |
| `peft` | 0.11.1 | LoRA adapter management |
| `trl` | 0.11.3 | SFT training loop |
| `bitsandbytes` | ≥0.43 | 4-bit NF4 quantization (GPU only) |
| `torch` | ≥2.2 | PyTorch backend |
| `gradio` | 4.44.0 | Interactive web demo |
| `datasets` | ≥2.20 | Dataset loading |

---

## Why This Project Stands Out

1. **Unique Task**: Trains models to *critique* code, not *generate* it — an underserved and distinct skill
2. **Real Data**: Uses actual developer review comments from merged PRs, not synthetic data
3. **Explainable**: Side-by-side A/B comparison makes fine-tuning improvements immediately visible
4. **Resource Efficient**: QLoRA enables training on a free Colab T4 GPU in ~15 minutes
5. **End-to-End**: Complete pipeline from data scraping → training → inference → interactive demo

---

## License

[MIT](LICENSE)
# LoRA-code-review-bot
