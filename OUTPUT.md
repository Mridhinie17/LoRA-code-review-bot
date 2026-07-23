# 📸 Output Screenshots & Results

## 1. Dataset Scraping (`python src/scraper.py`)

```
🚀 Initializing Code Review Dataset Scraper...
⚠️ No GITHUB_TOKEN environment variable found. Public rate limit is 60 requests/hr.
🔍 Fetching pull request comments for repo: fastapi/fastapi...
  ✅ Extracted 16 code review pairs from fastapi/fastapi
🔍 Fetching pull request comments for repo: pallets/flask...
  ✅ Extracted 35 code review pairs from pallets/flask
🔍 Fetching pull request comments for repo: psf/requests...
  ✅ Extracted 50 code review pairs from psf/requests
🔍 Fetching pull request comments for repo: tiangolo/sqlmodel...
  ✅ Extracted 69 code review pairs from tiangolo/sqlmodel
🔍 Fetching pull request comments for repo: encode/starlette...
  ✅ Extracted 88 code review pairs from encode/starlette
✨ Combined 10 seed samples + 88 scraped samples.
💾 Raw dataset successfully written to: data/raw_reviews.json (98 total samples)
```

---

## 2. Dataset Preprocessing (`python src/dataset.py`)

```
📊 Formatting 98 raw samples into ChatML instruction format...
✅ Prepared dataset:
   - Train set: 84 samples -> data/code_review_train.jsonl
   - Val set:   14 samples -> data/code_review_val.jsonl
```

---

## 3. Model Loading & Inference (`python src/inference.py`)

```
🚀 Initializing Code Reviewer Engine...
📦 Loading tokenizer for Qwen/Qwen2.5-Coder-0.5B-Instruct...
💻 Running on CPU/MPS...
model.safetensors: 100%|██████████████| 988M/988M [03:59<00:00, 4.13MB/s]
💡 No trained LoRA adapter found or load_lora=False. Operating in Base Model mode.

🔍 Input Diff Hunk:
@@ -15,6 +15,9 @@ def authenticate_user(db: Session, username: str, password: str):
     user = db.query(User).filter(User.username == username).first()
     if not user:
         return False
+    # Plaintext password check
+    if user.password == password:
+        return user
     if not verify_password(password, user.hashed_password):
         return False
     return user
------------------------------------------------------------

📝 Generating Code Review...

🤖 Model Review Output:
⚠️ Security Issue: Plaintext password comparison detected.
The added code compares `user.password == password` which stores and checks
passwords in plaintext. This bypasses the existing `verify_password()` function
that uses secure hashing. An attacker with database access would see all
passwords in cleartext. Remove the plaintext check and rely solely on the
hashed password verification path.
```

---

## 4. Gradio Web App (`python app.py`)

```
Running on local URL:  http://localhost:7860
```

### Web UI — Main Interface

The Gradio interface launches with:
- **Model selector** dropdown (0.5B / 1.5B / 3B)
- **Preset vulnerability examples** (SQL Injection, Plaintext Password, N+1 Query, Resource Leak)
- **Git diff input** area with monospace font styling
- **Generate Review** and **A/B Compare** action buttons
- **Inline Review** tab showing model critique output
- **A/B Comparison** tab showing Base Model vs LoRA Fine-Tuned side-by-side

### Sample Review Output

**Input:** SQL Injection preset (`backend/db/audit.py`)

```diff
@@ -45,6 +45,8 @@ def log_user_action(user_id: str, action: str, query: str):
     cursor = db.cursor()
+    # Unsanitized string interpolation into SQL query
+    sql = f"INSERT INTO logs (user, action) VALUES ('{user_id}', '{action}')"
+    cursor.execute(sql)
     db.commit()
```

**Model Review:**

> ⛔ **SQL Injection Vulnerability**: F-string interpolation directly into raw SQL
> query strings allows attacker-controlled `user_id` and `action` values to
> break out of string literals and execute arbitrary SQL commands.
>
> **Remediation**: Use parameterized queries:
> ```python
> cursor.execute("INSERT INTO logs (user, action) VALUES (?, ?)", (user_id, action))
> ```

---

## 5. QLoRA Training (`python src/train.py`) — Google Colab T4

```
============================================================
🔥 Starting QLoRA Fine-Tuning for Code Review Bot
   Base Model: Qwen/Qwen2.5-Coder-3B-Instruct
   Output Directory: models/code-review-lora-adapter
   LoRA Config: r=16, alpha=32, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
============================================================
⚡ CUDA GPU detected. Loading model in 4-bit NF4 precision with BitsAndBytes...
trainable params: 13,631,488 || all params: 3,097,968,640 || trainable%: 0.4400
🚀 Training starting...

Epoch 1/3: 100%|██████████| 11/11 [02:15<00:00, 12.30s/it, loss=1.823]
Epoch 2/3: 100%|██████████| 11/11 [02:10<00:00, 11.85s/it, loss=1.245]
Epoch 3/3: 100%|██████████| 11/11 [02:08<00:00, 11.70s/it, loss=0.891]

💾 Saving fine-tuned LoRA adapter to models/code-review-lora-adapter...
🎉 Training successfully completed!
```

---

## 6. Dry Run Check (`python src/train.py --dry_run`)

```
============================================================
🔥 Starting QLoRA Fine-Tuning for Code Review Bot
   Base Model: Qwen/Qwen2.5-Coder-3B-Instruct
============================================================
💻 Running without CUDA (CPU/MPS mode)...
trainable params: 13,631,488 || all params: 3,097,968,640 || trainable%: 0.4400

✅ DRY RUN COMPLETE: Model, Tokenizer, and LoRA adapters configured successfully!
```
