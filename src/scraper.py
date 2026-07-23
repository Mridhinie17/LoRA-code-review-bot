"""
GitHub PR Review Scraper & Dataset Generator.
Scrapes PR inline review comments and diff hunks from open-source repositories.
Includes a built-in high-quality seed dataset for offline or rate-limited training.
"""
import os
import json
import time
import requests
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DATA_PATH, DATA_DIR

DEFAULT_TARGET_REPOS = [
    "fastapi/fastapi",
    "pallets/flask",
    "psf/requests",
    "tiangolo/sqlmodel",
    "encode/starlette"
]

# Rich Seed Dataset for Offline / Instant Execution
SEED_REVIEWS = [
    {
        "repo": "fastapi/fastapi",
        "pr_id": 4102,
        "filename": "fastapi/routing.py",
        "diff_hunk": "@@ -120,6 +120,8 @@ def get_request_handler(\n     is_coroutine = asyncio.iscoroutinefunction(dependant.call)\n     async def app(request: Request) -> Response:\n         try:\n+            body = await request.body()\n+            data = json.loads(body)\n             body_bytes = await request.body()\n         except Exception:\n             pass",
        "review_comment": "🚨 **Multiple issues here**:\n1. `request.body()` is being called twice sequentially, which is redundant.\n2. Raw `json.loads(body)` can throw `json.JSONDecodeError`, but catching broad `Exception` with a silent `pass` swallows error details and hides malformed payload bugs.\n3. Recommend using Starlette's built-in `await request.json()` inside a specific `JSONDecodeError` try-except block with explicit HTTP 400 response."
    },
    {
        "repo": "pallets/flask",
        "pr_id": 3891,
        "filename": "src/flask/app.py",
        "diff_hunk": "@@ -45,7 +45,7 @@ class Flask(_PackageBoundObject):\n     def __init__(\n         self,\n         import_name: str,\n-        static_url_path: str | None = None,\n+        static_url_path: str | None = None, config_dict: dict = {},\n     ) -> None:",
        "review_comment": "⚠️ **Anti-pattern detected**: Using a mutable default argument `config_dict: dict = {}` in Python means all instances of `Flask` that default this parameter will share the exact same dictionary object in memory!\n\n**Fix**:\n```python\nconfig_dict: dict | None = None\n```\nAnd inside `__init__`:\n```python\nself.config_dict = config_dict if config_dict is not None else {}\n```"
    },
    {
        "repo": "psf/requests",
        "pr_id": 5210,
        "filename": "requests/sessions.py",
        "diff_hunk": "@@ -512,5 +512,8 @@ def send(self, request, **kwargs):\n         history = []\n         url = request.url\n+        # Execute SQL debug log\n+        cursor.execute(f\"INSERT INTO audit_logs VALUES ('{url}', '{time.time()}')\")\n         while url:\n             resp = self.send_single_request(request)",
        "review_comment": "⛔ **CRITICAL SECURITY RISK: SQL Injection Vulnerability**\n\nF-string interpolation directly into raw SQL query strings (`f\"INSERT INTO audit_logs VALUES ('{url}')\"`) allows attacker-controlled URLs to break out of string literals and execute arbitrary SQL commands.\n\n**Remediation**: Always use parameterized queries / bind variables:\n```python\ncursor.execute(\"INSERT INTO audit_logs VALUES (?, ?)\", (url, time.time()))\n```"
    },
    {
        "repo": "encode/starlette",
        "pr_id": 1820,
        "filename": "starlette/middleware/cors.py",
        "diff_hunk": "@@ -80,4 +80,6 @@ def process_response(self, headers):\n         if self.allow_origins:\n             headers[\"Access-Control-Allow-Origin\"] = \"*\"\n+        if self.allow_credentials:\n+            headers[\"Access-Control-Allow-Origin\"] = \"*\"",
        "review_comment": "🔒 **CORS Security Specification Violation**:\nAccording to the W3C CORS specification, `Access-Control-Allow-Origin: *` cannot be combined with `Access-Control-Allow-Credentials: true`. Web browsers will reject the response or block credentials.\n\n**Recommendation**: When `allow_credentials` is `True`, dynamic reflection of the request's `Origin` header must be used instead of wildcard `*`."
    },
    {
        "repo": "fastapi/fastapi",
        "pr_id": 4890,
        "filename": "fastapi/encoders.py",
        "diff_hunk": "@@ -102,6 +102,9 @@ def jsonable_encoder(obj: Any) -> Any:\n     if isinstance(obj, dict):\n         encoded_dict = {}\n         for key, value in obj.items():\n+            encoded_dict[str(key)] = jsonable_encoder(value)\n+        return encoded_dict\n+    return str(obj)",
        "review_comment": "⚡ **Performance & Recursion Depth Issue**:\nDirect unconditional recursive calling of `jsonable_encoder(value)` without depth limit or circular reference check can trigger `RecursionError` on deeply nested dictionary structures or self-referencing objects.\n\nAdditionally, returning `str(obj)` as fallback coerces unhandled objects (like custom classes) into strings silently instead of raising a typed `ValueError` or `TypeError`."
    },
    {
        "repo": "pallets/flask",
        "pr_id": 4100,
        "filename": "src/flask/blueprints.py",
        "diff_hunk": "@@ -150,6 +150,7 @@ def register_blueprint(self, blueprint, **options):\n         if blueprint.name in self.blueprints:\n             assert blueprint is self.blueprints[blueprint.name]\n+            return\n         self.blueprints[blueprint.name] = blueprint",
        "review_comment": "💡 **Consider explicit logging/warning**:\nSilently returning when re-registering an existing blueprint (`if blueprint.name in self.blueprints: return`) might hide accidental duplicate registration bugs in user code.\n\nRecommend emitting a `UserWarning` via `warnings.warn(f\"Blueprint '{blueprint.name}' is already registered.\")` to aid developer debugging."
    },
    {
        "repo": "psf/requests",
        "pr_id": 5890,
        "filename": "requests/adapters.py",
        "diff_hunk": "@@ -210,6 +210,8 @@ def get_connection(self, url, proxies=None):\n         conn = self.poolmanager.connection_from_url(url)\n+        f = open(\"/tmp/conn_log.txt\", \"a\")\n+        f.write(f\"Opened connection to {url}\\n\")\n         return conn",
        "review_comment": "🧹 **Resource Leak & Unsafe File I/O**:\n1. The file handle `f` is opened but never explicitly closed (`f.close()`), leading to dangling file descriptors.\n2. Writing to hardcoded `/tmp` paths is vulnerable to symlink attacks in multi-user POSIX environments.\n\n**Fix**: Use Python's standard `logging` module instead of direct file I/O operations."
    },
    {
        "repo": "django/django",
        "pr_id": 14200,
        "filename": "django/db/models/query.py",
        "diff_hunk": "@@ -310,6 +310,7 @@ def filter(self, *args, **kwargs):\n+        res = [item for item in self.all() if item.is_active]\n+        return res",
        "review_comment": "🐢 **N+1 Database Query Performance Warning**:\nUsing Python list comprehension `[item for item in self.all() if item.is_active]` pulls ALL records into RAM from the database and filters in Python, bypassing SQL indexing and causing massive memory bloat.\n\n**Fix**: Perform filtering at the database layer using QuerySet methods:\n```python\nreturn self.filter(is_active=True)\n```"
    },
    {
        "repo": "scikit-learn/scikit-learn",
        "pr_id": 21090,
        "filename": "sklearn/ensemble/_forest.py",
        "diff_hunk": "@@ -405,6 +405,7 @@ def predict(self, X):\n         X = self._validate_X_predict(X)\n+        if np.isnan(X).any():\n+            X[np.isnan(X)] = 0.0\n         return self.predict_proba(X).argmax(axis=1)",
        "review_comment": "⚠️ **Silent Data Corruption Risk**:\nSilently replacing NaN values with `0.0` inside `predict()` mutates feature distributions without informing the user, potentially producing invalid model predictions.\n\n**Recommendation**: Raise a `ValueError(\"Input contains NaN. Please use SimpleImputer first.\")` or require explicit user-side imputation."
    },
    {
        "repo": "tiangolo/sqlmodel",
        "pr_id": 310,
        "filename": "sqlmodel/main.py",
        "diff_hunk": "@@ -88,4 +88,7 @@ def create_db_and_tables(engine):\n     SQLModel.metadata.create_all(engine)\n+    session = Session(engine)\n+    session.execute(\"CREATE INDEX IF NOT EXISTS idx_user ON user(email)\")\n+    session.commit()",
        "review_comment": "🛠️ **Database Portability Issue**:\nRaw DDL SQL string execution (`CREATE INDEX IF NOT EXISTS...`) can break compatibility across different SQL dialects (PostgreSQL vs SQLite vs MySQL/MariaDB syntax differences).\n\n**Fix**: Define indexes cleanly using SQLAlchemy/SQLModel column metadata (`Field(index=True)`) or Alembic migration scripts."
    }
]

def fetch_github_pr_reviews(repos: List[str] = DEFAULT_TARGET_REPOS, limit_per_repo: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch PR review comments from GitHub API if token is provided.
    Returns list of dicts: {repo, pr_id, filename, diff_hunk, review_comment}
    """
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        print("🔑 Using provided GITHUB_TOKEN for API scraping.")
    else:
        print("⚠️ No GITHUB_TOKEN environment variable found. Public rate limit is 60 requests/hr.")

    scraped_reviews = []
    
    for repo in repos:
        print(f"🔍 Fetching pull request comments for repo: {repo}...")
        url = f"https://api.github.com/repos/{repo}/pulls/comments?per_page={limit_per_repo}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                comments = resp.json()
                for c in comments:
                    diff_hunk = c.get("diff_hunk", "").strip()
                    body = c.get("body", "").strip()
                    filename = c.get("path", "")
                    pr_url = c.get("html_url", "")
                    
                    # Filter out short or non-substantive comments (e.g. "+1", "thanks")
                    if diff_hunk and len(body) > 25 and not body.lower().startswith(("thanks", "+1", "lgtm")):
                        scraped_reviews.append({
                            "repo": repo,
                            "pr_id": c.get("pull_request_review_id", 0),
                            "filename": filename,
                            "diff_hunk": diff_hunk,
                            "review_comment": body,
                            "source": "github_api"
                        })
                print(f"  ✅ Extracted {len(scraped_reviews)} code review pairs from {repo}")
            else:
                print(f"  ⚠️ GitHub API returned status {resp.status_code}: {resp.text[:100]}")
                break
        except Exception as e:
            print(f"  ❌ Error fetching from {repo}: {e}")
            break

    return scraped_reviews

def generate_dataset():
    """
    Main entrypoint: Fetches live data or combines live data with high-quality seed dataset.
    Saves output to RAW_DATA_PATH.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("🚀 Initializing Code Review Dataset Scraper...")
    
    live_data = fetch_github_pr_reviews()
    
    combined_data = SEED_REVIEWS.copy()
    for item in combined_data:
        item["source"] = "seed"
        
    if live_data:
        combined_data.extend(live_data)
        print(f"✨ Combined {len(SEED_REVIEWS)} seed samples + {len(live_data)} scraped samples.")
    else:
        print(f"💡 Operating in seed dataset mode ({len(SEED_REVIEWS)} high-quality PR review pairs ready).")

    with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
    print(f"💾 Raw dataset successfully written to: {RAW_DATA_PATH} ({len(combined_data)} total samples)")
    return combined_data

if __name__ == "__main__":
    generate_dataset()
