"""
One-Click Hugging Face Space Deployment for BawaGo Universal AI Agent
Target: jaay2/bawago-ai-agent
"""

import os
import sys
from huggingface_hub import HfApi, create_repo

def find_token():
    # 1. CLI argument
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    # 2. Environment variables
    for k in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
        val = os.environ.get(k)
        if val and val.strip():
            return val.strip()
    # 3. .env files
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "..", ".env"),
        os.path.join(base, "..", "prototype", ".env"),
        os.path.join(base, ".env"),
        os.path.expanduser("~/.cache/huggingface/token"),
        os.path.expanduser("~/.huggingface/token")
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip().upper() in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_KEY"]:
                                return v.strip().strip("'\"")
                        elif line.startswith("hf_"):
                            return line
            except Exception:
                pass
    return None

def deploy(token: str = None):
    token = token or find_token()

    if not token:
        print("=" * 60)
        print("  Hugging Face Deployment: Token Required")
        print("=" * 60)
        print("Please provide your Hugging Face write token:")
        print("1. Get a Write token at: https://huggingface.co/settings/tokens")
        print("2. Run: python deploy_to_hf.py <YOUR_HF_TOKEN>")
        print("   OR: set HF_TOKEN=<YOUR_TOKEN> && python deploy_to_hf.py")
        print("=" * 60)
        return False

    repo_id = "jaay2/bawago-ai-agent"
    print(f">> Initializing Hugging Face Space: {repo_id}...")
    api = HfApi(token=token)

    try:
        url = create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="static",
            exist_ok=True,
            token=token
        )
        print(f">> Space verified/created: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f">> Notice during space creation (proceeding with upload): {e}")

    print(f">> Uploading universal agent files to Hugging Face Space...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        api.upload_folder(
            folder_path=base_dir,
            repo_id=repo_id,
            repo_type="space",
            token=token,
            ignore_patterns=[
                "__pycache__/*",
                "*.pyc",
                ".git/*",
                "*.log",
                ".env"
            ],
            commit_message="Deploy BawaGo Universal AI Agent (130 intents, 7 verticals, jargon-free)"
        )
        print("=" * 60)
        print("  DEPLOYMENT SUCCESSFUL!")
        print(f"  Live Space URL: https://huggingface.co/spaces/{repo_id}")
        print("=" * 60)
        return True
    except Exception as e:
        print(f">> Upload error: {e}")
        return False

if __name__ == "__main__":
    deploy()

