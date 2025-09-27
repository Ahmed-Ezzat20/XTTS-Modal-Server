#!/usr/bin/env python3
"""
Setup script for XTTS Modal deployment.
This script helps upload your XTTS model files to the Modal volume.
"""

import os
import sys
from pathlib import Path

import modal

def upload_model_files(model_dir: str):
    """Upload XTTS model files to the Modal volume"""
    model_path = Path(model_dir)
    
    if not model_path.exists():
        print(f"Error: Model directory {model_dir} does not exist")
        return False
    
    # Check for required files
    required_files = ["config.json", "model.pth"]
    optional_files = ["vocab.json"]
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        return False
    
    print(f"Uploading model files from {model_dir} to Modal volume...")
    
    # Get or create the model volume
    model_volume = modal.Volume.from_name("xtts-model", create_if_missing=True)
    
    # Upload files
    with model_volume.batch_upload() as batch:
        for file in required_files + optional_files:
            file_path = model_path / file
            if file_path.exists():
                print(f"  Uploading {file}...")
                batch.put_file(str(file_path), f"/{file}")
            else:
                print(f"  Skipping {file} (not found)")
    
    print("Model files uploaded successfully!")
    return True

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
        print("Usage: python setup_volumes.py <model_directory> [api_key]")
        print("Example: python setup_volumes.py ./my_xtts_model my_secret_key")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("XTTS Modal Setup")
    print("================")
    
    # Upload model files
    if not upload_model_files(model_dir):
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
