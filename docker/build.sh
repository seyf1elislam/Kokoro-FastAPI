#!/bin/bash
set -e

# Get version from argument or use default
VERSION=${1:-"latest"}

# Build both CPU and GPU images using docker buildx bake
echo "Building CPU and GPU images..."
VERSION=$VERSION docker buildx bake --push

echo "Build complete!"
echo "Created images with version: $VERSION"
