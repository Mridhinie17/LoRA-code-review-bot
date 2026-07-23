"""
Dataset Preprocessor and Formatter for Qwen2.5-Coder LoRA Fine-Tuning.
Converts raw PR diffs and review comments into ChatML format JSONL files.
"""
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    SYSTEM_PROMPT,
    DATA_DIR
)
from src.scraper import generate_dataset

def format_sample_to_chatml(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats a single diff-review pair into ChatML messages format.
    """
    filename = sample.get("filename", "code_patch.py")
    diff_hunk = sample.get("diff_hunk", "")
    review_comment = sample.get("review_comment", "")

    user_content = (
        f"File: `{filename}`\n"
        f"```diff\n"
        f"{diff_hunk}\n"
        f"```\n\n"
        f"Please review this code diff and provide detailed inline feedback."
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": review_comment}
        ]
    }

def prepare_and_split_dataset(val_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Reads raw json, formats samples, performs reproducible train/val split.
    """
    if not RAW_DATA_PATH.exists():
        print(f"⚠️ {RAW_DATA_PATH} not found. Running dataset generator/scraper...")
        generate_dataset()

    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    print(f"📊 Formatting {len(raw_samples)} raw samples into ChatML instruction format...")
    formatted_samples = [format_sample_to_chatml(s) for s in raw_samples]

    random.seed(seed)
    random.shuffle(formatted_samples)

    val_size = int(len(formatted_samples) * val_ratio)
    val_size = max(1, val_size) if len(formatted_samples) > 1 else 0

    train_samples = formatted_samples[val_size:] if val_size > 0 else formatted_samples
    val_samples = formatted_samples[:val_size] if val_size > 0 else []

    # Write TRAIN JSONL
    with open(TRAIN_DATA_PATH, "w", encoding="utf-8") as f:
        for item in train_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write VAL JSONL
    with open(VAL_DATA_PATH, "w", encoding="utf-8") as f:
        for item in val_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Prepared dataset:")
    print(f"   - Train set: {len(train_samples)} samples -> {TRAIN_DATA_PATH}")
    print(f"   - Val set:   {len(val_samples)} samples -> {VAL_DATA_PATH}")

    return train_samples, val_samples

def load_code_review_dataset():
    """
    Loads Hugging Face Dataset object directly for training.
    """
    from datasets import load_dataset
    if not TRAIN_DATA_PATH.exists() or not VAL_DATA_PATH.exists():
        prepare_and_split_dataset()

    return load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_DATA_PATH),
            "validation": str(VAL_DATA_PATH)
        }
    )

if __name__ == "__main__":
    prepare_and_split_dataset()
