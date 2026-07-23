"""
Inference and Evaluation Engine for Code Review Bot.
Supports single diff reviews and side-by-side comparison (Base Model vs. Fine-Tuned LoRA).
"""
import os
import sys
from pathlib import Path

# Ensure PyTorch framework detection for Hugging Face & PEFT
os.environ["USE_TORCH"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import argparse
from typing import Tuple, Dict, Any

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_MODEL_NAME, OUTPUT_DIR, SYSTEM_PROMPT

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class CodeReviewer:
    """
    Inference wrapper for base LLM and fine-tuned LoRA adapter.
    """
    def __init__(self, base_model_name: str = DEFAULT_MODEL_NAME, adapter_path: str = str(OUTPUT_DIR)):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.tokenizer = None
        self.model = None
        self.lora_loaded = False
        
    def load_models(self, load_lora: bool = True):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Hugging Face transformers and peft are required. Install via pip install -r requirements.txt")

        print(f"📦 Loading tokenizer for {self.base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_cuda = torch.cuda.is_available()
        use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        device_map = "auto" if use_cuda else None
        
        if use_cuda:
            print("⚡ CUDA GPU detected. Loading model with BitsAndBytes 4-bit...")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"⚠️ BitsAndBytes 4-bit failed: {e}. Falling back to standard float16 precision...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    device_map=device_map,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
        else:
            print("💻 Running on CPU/MPS...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map=device_map,
                torch_dtype=torch.float16 if use_mps else torch.float32,
                trust_remote_code=True
            )

        adapter_dir = Path(self.adapter_path)
        if load_lora and adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            print(f"⚡ Attaching fine-tuned LoRA adapter from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.lora_loaded = True
        else:
            print("💡 No trained LoRA adapter found or load_lora=False. Operating in Base Model mode.")
            self.model = base_model
            self.lora_loaded = False

        self.model.eval()

    def review_diff(self, diff_hunk: str, filename: str = "script.py", max_new_tokens: int = 512, use_lora: bool = True) -> str:
        """
        Generate review comment for a given git diff hunk.
        """
        if self.model is None:
            self.load_models(load_lora=use_lora)

        user_prompt = (
            f"File: `{filename}`\n"
            f"```diff\n"
            f"{diff_hunk.strip()}\n"
            f"```\n\n"
            f"Please review this code diff and provide detailed inline feedback."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Slice off prompt tokens to return only the generated response
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        review_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return review_text.strip()

    def compare_base_vs_lora(self, diff_hunk: str, filename: str = "script.py") -> Tuple[str, str]:
        """
        Runs inference first without LoRA adapter (Base Model) and then with LoRA adapter attached.
        Returns: (base_model_review, lora_model_review)
        """
        # 1. Base Model Review
        if self.lora_loaded and isinstance(self.model, PeftModel):
            print("🔹 Disabling LoRA adapter for Base Model baseline...")
            with self.model.disable_adapter():
                base_review = self.review_diff(diff_hunk, filename=filename)
        else:
            base_review = self.review_diff(diff_hunk, filename=filename)

        # 2. LoRA Fine-Tuned Review
        if self.lora_loaded:
            print("🔹 Enabling LoRA adapter for Fine-Tuned review...")
            lora_review = self.review_diff(diff_hunk, filename=filename)
        else:
            lora_review = "(LoRA Adapter not trained yet. Run train.py first to enable fine-tuned model comparison.)"

        return base_review, lora_review


SAMPLE_DIFF = """@@ -15,6 +15,9 @@ def authenticate_user(db: Session, username: str, password: str):
     user = db.query(User).filter(User.username == username).first()
     if not user:
         return False
+    # Plaintext password check
+    if user.password == password:
+        return user
     if not verify_password(password, user.hashed_password):
         return False
     return user"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Code Review Bot Inference")
    parser.add_argument("--filename", type=str, default="auth.py", help="Filename for diff context")
    parser.add_argument("--diff", type=str, default=SAMPLE_DIFF, help="Git diff hunk to review")
    args = parser.parse_args()

    print("🚀 Initializing Code Reviewer Engine...")
    reviewer = CodeReviewer()
    reviewer.load_models(load_lora=True)
    
    print("\n🔍 Input Diff Hunk:")
    print(args.diff)
    print("-" * 60)

    print("\n📝 Generating Code Review...")
    review = reviewer.review_diff(args.diff, filename=args.filename)
    print("\n🤖 Model Review Output:")
    print(review)
