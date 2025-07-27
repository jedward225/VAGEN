#!/usr/bin/env python3
"""
Download Qwen2.5-VL-7B-Instruct model to local directory
"""
import os
from huggingface_hub import snapshot_download

def download_qwen25vl_7b():
    """Download Qwen2.5-VL-7B-Instruct model"""
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    local_dir = "/root/models/Qwen2.5-VL-7B-Instruct"
    
    print(f"Downloading {model_name} to {local_dir}...")
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            resume_download=True,
            local_files_only=False
        )
        print(f"✅ Successfully downloaded {model_name} to {local_dir}")
        return local_dir
        
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return None

if __name__ == "__main__":
    download_qwen25vl_7b()