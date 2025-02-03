#!/bin/bash

# Find project root by looking for api directory
find_project_root() {
    local current_dir="$PWD"
    local max_steps=5
    local steps=0
    
    while [ $steps -lt $max_steps ]; do
        if [ -d "$current_dir/api" ]; then
            echo "$current_dir"
            return 0
        fi
        current_dir="$(dirname "$current_dir")"
        ((steps++))
    done
    
    echo "Error: Could not find project root (no api directory found)" >&2
    exit 1
}

# Function to download a file
download_file() {
    local url="$1"
    local output_dir="$2"
    local model_type="$3"
    local filename=$(basename "$url")
    
    # Validate file extension
    if [[ ! "$filename" =~ \.$model_type$ ]]; then
        echo "Warning: $filename is not a .$model_type file" >&2
        return 1
    }
    
    echo "Downloading $filename..."
    if curl -L "$url" -o "$output_dir/$filename"; then
        echo "Successfully downloaded $filename"
        return 0
    else
        echo "Error downloading $filename" >&2
        return 1
    fi
}

# Parse arguments
MODEL_TYPE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        *)
            # If no flag specified, treat remaining args as model URLs
            break
            ;;
    esac
done

# Validate model type
if [ "$MODEL_TYPE" != "pth" ] && [ "$MODEL_TYPE" != "onnx" ]; then
    echo "Error: Must specify model type with --type (pth or onnx)" >&2
    exit 1
fi

# Find project root and ensure models directory exists
PROJECT_ROOT=$(find_project_root)
if [ $? -ne 0 ]; then
    exit 1
fi

MODELS_DIR="$PROJECT_ROOT/api/src/models"
echo "Downloading models to $MODELS_DIR"
mkdir -p "$MODELS_DIR"

# Default models if no arguments provided
if [ "$MODEL_TYPE" = "pth" ]; then
    DEFAULT_MODELS=(
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19.pth"
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19-half.pth"
    )
else
    DEFAULT_MODELS=(
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19.onnx"
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19_fp16.onnx"
    )
fi

# Use provided models or default
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# Download all models
success=true
for model in "${MODELS[@]}"; do
    if ! download_file "$model" "$MODELS_DIR" "$MODEL_TYPE"; then
        success=false
    fi
done

if [ "$success" = true ]; then
    echo "${MODEL_TYPE^^} model download complete!"
    exit 0
else
    echo "Some downloads failed" >&2
    exit 1
fi