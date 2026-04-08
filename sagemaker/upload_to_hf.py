"""
upload_to_hf.py  —  Upload model.tar.gz to Hugging Face
=========================================================
Handles large files (200MB+) using the huggingface_hub library,
which automatically uses Git LFS for files over 10MB.

Usage:
    python sagemaker/upload_to_hf.py

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login        ← run once, saves token to ~/.cache/huggingface
"""

import os
from huggingface_hub import HfApi, login

# ── Configuration — edit these 3 lines ───────────────────────────────────────
HF_REPO_ID  = "YOUR-USERNAME/YOUR-REPO-NAME"   # e.g. "zerogravity/email-gatekeeper"
HF_TOKEN    = os.getenv("HF_TOKEN")            # set env var  OR  paste token below
# HF_TOKEN  = "hf_xxxxxxxxxxxxxxxxxxxx"        # ← dev only, never commit this

LOCAL_FILE  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.tar.gz"
)
REPO_FILE   = "model.tar.gz"                   # path inside the HF repo
REPO_TYPE   = "model"                          # "model" | "dataset" | "space"
# ─────────────────────────────────────────────────────────────────────────────


def upload():
    # ── Validate local file exists ────────────────────────────────────────────
    if not os.path.exists(LOCAL_FILE):
        raise FileNotFoundError(
            f"model.tar.gz not found at:\n  {LOCAL_FILE}\n"
            "Run  python sagemaker/package.py  first to build it."
        )

    size_mb = os.path.getsize(LOCAL_FILE) / (1024 * 1024)
    print(f"\n  File     : {LOCAL_FILE}")
    print(f"  Size     : {size_mb:.1f} MB")
    print(f"  Repo     : {HF_REPO_ID}")
    print(f"  Dest     : {REPO_FILE}\n")

    # ── Authenticate ──────────────────────────────────────────────────────────
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)
    else:
        # Falls back to cached token from  huggingface-cli login
        print("  No HF_TOKEN env var found — using cached login credentials.")

    # ── Upload ────────────────────────────────────────────────────────────────
    api = HfApi()

    print("  Uploading... (large files use Git LFS automatically)\n")

    url = api.upload_file(
        path_or_fileobj=LOCAL_FILE,
        path_in_repo=REPO_FILE,
        repo_id=HF_REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Upload model.tar.gz — Email Gatekeeper RL Agent",
    )

    print(f"\n  ✅ Upload complete!")
    print(f"  View : https://huggingface.co/{HF_REPO_ID}")
    print(f"  File : {url}\n")

    # ── Print the download URL for use in deploy.py ───────────────────────────
    download_url = (
        f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{REPO_FILE}"
    )
    print(f"  Direct download URL (use in deploy.py):")
    print(f"  {download_url}\n")


if __name__ == "__main__":
    upload()
