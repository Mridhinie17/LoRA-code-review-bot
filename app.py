"""
Professional & Clean Code Review Assistant Web UI.
Fine-tuned Qwen2.5-Coder + QLoRA Inline Code Critique.
"""
import sys
import os
from pathlib import Path

# Force PyTorch detection for Transformers & PEFT
os.environ["USE_TORCH"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import gradio as gr

from config import DEFAULT_MODEL_NAME, OUTPUT_DIR
from src.inference import CodeReviewer

# Global Lazy-Loaded Reviewer Engine
reviewer_engine = None

def get_reviewer(model_name: str = DEFAULT_MODEL_NAME):
    global reviewer_engine
    if reviewer_engine is None or reviewer_engine.base_model_name != model_name:
        print(f"🚀 Initializing CodeReviewer ({model_name})...")
        reviewer_engine = CodeReviewer(base_model_name=model_name)
        reviewer_engine.load_models(load_lora=True)
    return reviewer_engine

# Curated Code Bug Examples
PRESET_EXAMPLES = {
    "SQL Injection Vulnerability": (
        "backend/db/audit.py",
        "@@ -45,6 +45,8 @@ def log_user_action(user_id: str, action: str, query: str):\n"
        "     cursor = db.cursor()\n"
        "+    # Unsanitized string interpolation into SQL query\n"
        "+    sql = f\"INSERT INTO logs (user, action) VALUES ('{user_id}', '{action}')\"\n"
        "+    cursor.execute(sql)\n"
        "     db.commit()"
    ),
    "Plaintext Password Comparison": (
        "auth/service.py",
        "@@ -20,6 +20,8 @@ def check_login(user, raw_password):\n"
        "     if not user:\n"
        "         return False\n"
        "+    if user.password == raw_password:\n"
        "+        return True\n"
        "     return verify_argon2_hash(raw_password, user.password_hash)"
    ),
    "N+1 Database Query Loop": (
        "api/reports.py",
        "@@ -102,6 +102,9 @@ def get_all_user_orders(users):\n"
        "     results = []\n"
        "+    for user in users:\n"
        "+        orders = db.query(Order).filter(Order.user_id == user.id).all()\n"
        "+        results.append({\"user\": user.name, \"orders\": orders})\n"
        "     return results"
    ),
    "Unclosed File Handle Leak": (
        "utils/exporter.py",
        "@@ -15,4 +15,6 @@ def export_metrics_to_disk(metrics_data):\n"
        "     filename = f\"/tmp/metrics_{time.time()}.json\"\n"
        "+    f = open(filename, \"w\")\n"
        "+    f.write(json.dumps(metrics_data))\n"
        "     return filename"
    )
}

def handle_preset_select(preset_name: str):
    if preset_name in PRESET_EXAMPLES:
        return PRESET_EXAMPLES[preset_name][0], PRESET_EXAMPLES[preset_name][1]
    return "", ""

def run_single_review(filename: str, diff_text: str, mode: str, model_choice: str):
    if not diff_text.strip():
        return "Please paste a valid git diff hunk."

    try:
        engine = get_reviewer(model_choice)
        use_lora = (mode == "Fine-Tuned LoRA Adapter")
        return engine.review_diff(diff_text, filename=filename, use_lora=use_lora)
    except Exception as e:
        return f"Error generating review: {str(e)}"

def run_ab_comparison(filename: str, diff_text: str, model_choice: str):
    if not diff_text.strip():
        return "Please paste a valid git diff hunk.", "Please paste a valid git diff hunk."

    try:
        engine = get_reviewer(model_choice)
        return engine.compare_base_vs_lora(diff_text, filename=filename)
    except Exception as e:
        err_msg = f"Error generating review: {str(e)}"
        return err_msg, err_msg

# Professional Clean CSS Customizations
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #0f172a !important;
    color: #f8fafc !important;
}

.main-header {
    border-bottom: 1px solid #1e293b;
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
}

.main-title {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #f8fafc !important;
    margin-bottom: 0.25rem !important;
}

.main-subtitle {
    font-size: 0.875rem !important;
    color: #94a3b8 !important;
    font-weight: 400 !important;
}

.code-editor textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.875rem !important;
    line-height: 1.5 !important;
    background-color: #020617 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

.review-output-box {
    background-color: #020617 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    padding: 1.25rem !important;
    min-height: 380px !important;
    color: #e2e8f0 !important;
}

button.primary-btn {
    background: #2563eb !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}

button.primary-btn:hover {
    background: #1d4ed8 !important;
}

button.secondary-btn {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f8fafc !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
}

button.secondary-btn:hover {
    background: #334155 !important;
}

.tab-nav button {
    font-weight: 500 !important;
    font-size: 0.875rem !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
).set(
    body_background_fill="#0f172a",
    block_background_fill="#1e293b",
    block_border_color="#334155",
    button_primary_background_fill="#2563eb",
    button_primary_text_color="#ffffff"
)

with gr.Blocks(title="Code Review Assistant", css=CUSTOM_CSS, theme=theme) as demo:
    gr.HTML(
        """
        <div class="main-header">
            <div class="main-title">Code Review Assistant</div>
            <div class="main-subtitle">Inline Code Critique Engine &bull; Qwen2.5-Coder + QLoRA</div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # Left Column: Inputs & Controls
        with gr.Column(scale=5):
            preset_dropdown = gr.Dropdown(
                choices=list(PRESET_EXAMPLES.keys()),
                label="Preset Vulnerability Examples",
                value="SQL Injection Vulnerability"
            )
            
            with gr.Row():
                filename_input = gr.Textbox(
                    label="File Path",
                    value="backend/db/audit.py",
                    placeholder="e.g. src/auth.py",
                    scale=2
                )
                model_selector = gr.Dropdown(
                    choices=[
                        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        "Qwen/Qwen2.5-Coder-3B-Instruct"
                    ],
                    value="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                    label="Base Model Size",
                    scale=3
                )

            diff_input = gr.TextArea(
                label="Git Diff Hunk",
                value=PRESET_EXAMPLES["SQL Injection Vulnerability"][1],
                lines=11,
                elem_classes=["code-editor"],
                placeholder="Paste git diff snippet starting with @@ ..."
            )

            with gr.Row():
                review_btn = gr.Button("Generate Review", variant="primary", elem_classes=["primary-btn"])
                compare_btn = gr.Button("A/B Compare (Base vs LoRA)", variant="secondary", elem_classes=["secondary-btn"])

        # Right Column: Clean Review Output
        with gr.Column(scale=6):
            with gr.Tabs(elem_classes=["tab-nav"]):
                with gr.TabItem("Inline Review"):
                    single_output = gr.Markdown(
                        value="*Select a preset or paste a git diff, then click **Generate Review**.*",
                        elem_classes=["review-output-box"]
                    )
                
                with gr.TabItem("A/B Comparison"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Base Qwen2.5-Coder**")
                            base_output = gr.Markdown(value="*Base model output...*", elem_classes=["review-output-box"])
                        with gr.Column():
                            gr.Markdown("**Fine-Tuned LoRA Adapter**")
                            lora_output = gr.Markdown(value="*LoRA fine-tuned output...*", elem_classes=["review-output-box"])

    # Event Connections
    preset_dropdown.change(
        fn=handle_preset_select,
        inputs=[preset_dropdown],
        outputs=[filename_input, diff_input]
    )

    review_btn.click(
        fn=lambda fname, diff, mchoice: run_single_review(fname, diff, "Fine-Tuned LoRA Adapter", mchoice),
        inputs=[filename_input, diff_input, model_selector],
        outputs=[single_output]
    )

    compare_btn.click(
        fn=run_ab_comparison,
        inputs=[filename_input, diff_input, model_selector],
        outputs=[base_output, lora_output]
    )

if __name__ == "__main__":
    start_port = int(os.getenv("PORT", 7860))
    for p in range(start_port, start_port + 10):
        try:
            demo.launch(server_name="localhost", server_port=p, share=False)
            break
        except OSError:
            continue
