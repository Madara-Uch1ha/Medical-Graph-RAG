"""
Download mimic_ex dataset from HuggingFace for Medical-Graph-RAG testing.
"""

import os
from huggingface_hub import snapshot_download

def download_mimic_ex(target_dir: str = "./dataset/mimic_ex"):
    """
    Downloads the mimic_ex dataset from HuggingFace.
    
    Args:
        target_dir: Directory to save the dataset (default: ./dataset/mimic_ex)
    """
    print(f"Downloading mimic_ex dataset to {target_dir}...")
    
    # Create parent directory if needed
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    
    # Download from HuggingFace
    snapshot_download(
        repo_id="Morson/mimic_ex",
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    
    print(f"✅ Dataset downloaded to {target_dir}")
    
    # List downloaded files
    files = os.listdir(target_dir)
    print(f"📁 Files: {files}")
    
    return target_dir

if __name__ == "__main__":
    download_mimic_ex()
