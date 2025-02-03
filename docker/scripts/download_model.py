#!/usr/bin/env python3
import os
import sys
import argparse
import requests
from pathlib import Path
from typing import List

def download_file(url: str, output_dir: Path, model_type: str, overwrite:str) -> bool:
    """Download a file from URL to the specified directory.
    
    Returns:
        bool: True if download succeeded, False otherwise
    """
    filename = os.path.basename(url)
    if not filename.endswith(f'.{model_type}'):
        print(f"Warning: {filename} is not a .{model_type} file", file=sys.stderr)
        return False
        
    output_path = output_dir / filename
    
    if os.path.exists(output_path):
        print(f"{filename} exists. Canceling download")
        return True
    
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}", file=sys.stderr)
        return False

def find_project_root() -> Path:
    """Find project root by looking for api directory."""
    max_steps = 5
    current = Path(__file__).resolve()
    for _ in range(max_steps):
        if (current / 'api').is_dir():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (no api directory found)")

def main() -> int:
    """Download models to the project.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description='Download model files')
    parser.add_argument('--type', choices=['pth', 'onnx'], required=True,
                      help='Model type to download (pth or onnx)')
    parser.add_argument('--overwrite', action='store_true', help='Overwite existing files')
    parser.add_argument('urls', nargs='*', help='Optional model URLs to download')
    args = parser.parse_args()

    try:
        # Find project root and ensure models directory exists
        project_root = find_project_root()
        models_dir = project_root / 'api' / 'src' / 'models'
        print(f"Downloading models to {models_dir}")
        models_dir.mkdir(exist_ok=True)
        
        # Default models if no arguments provided
        default_models = {
            'pth': [
                "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19.pth",
                "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19-half.pth"
            ],
            'onnx': [
                "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19.onnx",
                "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19_fp16.onnx"
            ]
        }
        
        # Use provided models or default
        models_to_download = args.urls if args.urls else default_models[args.type]
        
        # Download all models
        success = True
        for model_url in models_to_download:
            if not download_file(model_url, models_dir, args.type,args.overwrite):
                success = False
        
        if success:
            print(f"{args.type.upper()} model download complete!")
            return 0
        else:
            print("Some downloads failed", file=sys.stderr)
            return 1
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())