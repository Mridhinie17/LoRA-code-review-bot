"""
QLoRA Training Script for Qwen2.5-Coder Code Review Bot.
Fine-tunes base LLM using bitsandbytes 4-bit NF4 quantization and PEFT LoRA adapters.
"""
import os
import sys
import torch
import argparse
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DEFAULT_MODEL_NAME,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    TRAIN_SETTINGS,
    OUTPUT_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH
)
from src.dataset import prepare_and_split_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tune Code Review Bot")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="HuggingFace model repo")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output adapter directory")
    parser.add_argument("--epochs", type=int, default=TRAIN_SETTINGS["num_train_epochs"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=TRAIN_SETTINGS["per_device_train_batch_size"], help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=TRAIN_SETTINGS["learning_rate"], help="Learning rate")
    parser.add_argument("--dry_run", action="store_true", help="Quick setup check without full training")
    return parser.parse_args()

def train():
    args = parse_args()
    print("=" * 60)
    print("🔥 Starting QLoRA Fine-Tuning for Code Review Bot")
    print(f"   Base Model: {args.model_name}")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   LoRA Config: r={LORA_R}, alpha={LORA_ALPHA}, target_modules={LORA_TARGET_MODULES}")
    print("=" * 60)

    # 1. Ensure dataset exists
    if not TRAIN_DATA_PATH.exists() or not VAL_DATA_PATH.exists():
        prepare_and_split_dataset()

    from datasets import load_dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_DATA_PATH),
            "validation": str(VAL_DATA_PATH)
        }
    )

    # 2. Imports for PyTorch/Transformers
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # 3. Configure 4-bit Quantization (QLoRA)
    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    if use_cuda:
        print("⚡ CUDA GPU detected. Loading model in 4-bit NF4 precision with BitsAndBytes...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        device_map = "auto"
    else:
        print("💻 Running without CUDA (CPU/MPS mode). Using float16/float32 precision for model initialization...")
        bnb_config = None
        device_map = None

    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. Load Base Model
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.float16 if use_mps else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if use_cuda and bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # 6. Configure PEFT / LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if args.dry_run:
        print("\n✅ DRY RUN COMPLETE: Model, Tokenizer, and LoRA adapters configured successfully!")
        return

    # 7. Configure SFT Training Arguments
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=TRAIN_SETTINGS["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAIN_SETTINGS["gradient_accumulation_steps"],
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_seq_length=TRAIN_SETTINGS["max_seq_length"],
        warmup_ratio=TRAIN_SETTINGS["warmup_ratio"],
        logging_steps=TRAIN_SETTINGS["logging_steps"],
        save_strategy=TRAIN_SETTINGS["save_strategy"],
        eval_strategy=TRAIN_SETTINGS["eval_strategy"],
        fp16=use_cuda,
        dataset_text_field="messages",
        packing=False
    )

    # 8. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=sft_config,
    )

    # 9. Train Model
    print("🚀 Training starting...")
    trainer.train()

    # 10. Save LoRA Adapter & Tokenizer
    print(f"\n💾 Saving fine-tuned LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("🎉 Training successfully completed!")

if __name__ == "__main__":
    train()
