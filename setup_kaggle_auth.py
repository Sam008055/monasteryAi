"""
MonasteryAI - Kaggle Authentication Setup Script
Configures local ~/.kaggle/kaggle.json and ~/.kaggle/access_token securely
using environment variables (KAGGLE_USERNAME, KAGGLE_KEY, KAGGLE_API_TOKEN)
or from a local .env file.
"""

import os
import json
from pathlib import Path

# Load credentials from environment variables (never hardcoded in source control)
kaggle_username = os.environ.get("KAGGLE_USERNAME", "")
kaggle_key = os.environ.get("KAGGLE_KEY", "")
kaggle_token = os.environ.get("KAGGLE_API_TOKEN", "")

if not kaggle_username or not kaggle_key:
    print("[INFO] KAGGLE_USERNAME or KAGGLE_KEY environment variables not set.")
    print("[INFO] Please configure your .env file or set environment variables before running.")
    print("[INFO] Example:")
    print("       export KAGGLE_USERNAME='your_username'")
    print("       export KAGGLE_KEY='your_api_key'")

kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(parents=True, exist_ok=True)

# 1. Write ~/.kaggle/access_token if token provided
if kaggle_token:
    token_file = kaggle_dir / "access_token"
    token_file.write_text(kaggle_token.strip() + "\n", encoding="utf-8")
    print(f"[SUCCESS] Configured: {token_file}")

# 2. Write ~/.kaggle/kaggle.json
if kaggle_username and kaggle_key:
    json_file = kaggle_dir / "kaggle.json"
    json_content = {"username": kaggle_username, "key": kaggle_key}
    json_file.write_text(json.dumps(json_content, indent=2) + "\n", encoding="utf-8")
    print(f"[SUCCESS] Configured: {json_file}")
