"""
Central configuration for LoRA Code Review Bot.
"""
import os
import sys
from pathlib import Path

# Ensure PyTorch framework detection for HuggingFace Transformers & PEFT
os.environ["USE_TORCH"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

try:
    import torch
except ImportError:
    pass

# Base Directory
BASE_DIR = Path(__file__).parent.resolve()

# Target LLM:
# Default to "Qwen/Qwen2.5-Coder-0.5B-Instruct" for fast, lightweight local CPU execution (loads in 5s)
# Can be set to "Qwen/Qwen2.5-Coder-3B-Instruct" or "Qwen/Qwen2.5-Coder-7B-Instruct" for Colab/GPU environments.
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-0.5B-Instruct")

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

# Training Settings (Tuned for Colab T4 GPU)
TRAIN_SETTINGS = {
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "max_seq_length": 1024,
    "warmup_ratio": 0.05,
    "logging_steps": 5,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "fp16": True,
    "bf16": False,
}

# System Prompt for Code Reviewer
SYSTEM_PROMPT = (
    "You are a senior principal software engineer and expert code reviewer. "
    "Analyze the provided git diff hunk carefully and provide constructive, actionable, "
    "human-style inline code review comments. Focus on potential bugs, security flaws, "
    "performance issues, edge cases, missing error handling, code readability, and modern best practices."
)

# File Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw_reviews.json"
TRAIN_DATA_PATH = DATA_DIR / "code_review_train.jsonl"
VAL_DATA_PATH = DATA_DIR / "code_review_val.jsonl"
OUTPUT_DIR = BASE_DIR / "models" / "code-review-lora-adapter"

# Create directories if needed
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
