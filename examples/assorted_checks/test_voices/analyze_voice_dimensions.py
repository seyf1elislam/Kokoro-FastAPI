import os
import torch
from loguru import logger

def analyze_voice_file(file_path):
    """Analyze dimensions and statistics of a voice tensor."""
    try:
        tensor = torch.load(file_path, map_location="cpu")
        logger.info(f"\nAnalyzing {os.path.basename(file_path)}:")
        logger.info(f"Shape: {tensor.shape}")
        logger.info(f"Mean: {tensor.mean().item():.4f}")
        logger.info(f"Std: {tensor.std().item():.4f}")
        logger.info(f"Min: {tensor.min().item():.4f}")
        logger.info(f"Max: {tensor.max().item():.4f}")
        return tensor.shape
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
        return None

def main():
    """Analyze voice files in the voices directory."""
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    voices_dir = os.path.join(project_root, "api", "src", "voices", "v1_0")
    
    logger.info(f"Scanning voices in: {voices_dir}")
    
    # Track shapes for comparison
    shapes = {}
    
    # Analyze each .pt file
    for file in os.listdir(voices_dir):
        if file.endswith('.pt'):
            file_path = os.path.join(voices_dir, file)
            shape = analyze_voice_file(file_path)
            if shape:
                shapes[file] = shape
    
    # Report findings
    logger.info("\nShape Analysis:")
    shape_groups = {}
    for file, shape in shapes.items():
        if shape not in shape_groups:
            shape_groups[shape] = []
        shape_groups[shape].append(file)
    
    for shape, files in shape_groups.items():
        logger.info(f"\nShape {shape}:")
        for file in files:
            logger.info(f"  - {file}")

if __name__ == "__main__":
    main()