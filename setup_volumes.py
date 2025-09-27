#!/usr/bin/env python3
"""
Setup script for XTTS Modal deployment.
This script helps download your XTTS model files from Hugging Face and upload them to the Modal volume.
"""

import os
import sys
from pathlib import Path

import modal
from huggingface_hub import hf_hub_download

def download_model_from_hf(model_repo_id: str):
    """Download XTTS model files from Hugging Face Hub to the Modal volume"""
    print(f"Downloading model files from {model_repo_id} to Modal volume...")

    # Get or create the model volume
    model_volume = modal.Volume.from_name("xtts-model", create_if_missing=True)

    # List of files to download
    files_to_download = ["config.json", "model.pth", "vocab.json"]

    try:
        # Download and upload files in a batch
        with model_volume.batch_upload() as batch:
            for filename in files_to_download:
                print(f"  Downloading {filename}...")
                # Download file from Hugging Face Hub
                downloaded_path = hf_hub_download(repo_id=model_repo_id, filename=filename)
                # Upload to Modal Volume
                batch.put_file(downloaded_path, f"/{filename}")
        
        print("Model files downloaded and uploaded successfully!")
        return True
    except Exception as e:
        print(f"Error during model download/upload: {e}")
        return False

def create_api_secret(api_key: str):
    """Create a Modal secret for the API key"""
    try:
        # Try to create the secret
        secret = modal.Secret.from_dict({"API_KEY": api_key})
        secret.save("xtts-api-key")
        print("API key secret created successfully!")
        return True
    except Exception as e:
        print(f"Warning: Could not create API key secret: {e}")
        print("You can create it manually in the Modal dashboard or skip API key authentication")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python setup_volumes.py <hugging_face_model_id> [api_key]")
        print("Example: python setup_volumes.py Genarabia-ai/Kuwaiti_XTTS_Latest my_secret_key")
        sys.exit(1)
    
    model_repo_id = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("XTTS Modal Setup")
    print("=================")
    
    # Download model files from Hugging Face
    if not download_model_from_hf(model_repo_id):
        sys.exit(1)
    
    # Create API secret if provided
    if api_key:
        create_api_secret(api_key)
    else:
        print("No API key provided - authentication will be disabled")
    
    print("\nSetup complete!")
    print("Next steps:")
    print("1. Deploy the server: modal deploy modal_xtts_server.py")
    print("2. Test the deployment: curl <your-modal-url>/healthz")

if __name__ == "__main__":
    main()
