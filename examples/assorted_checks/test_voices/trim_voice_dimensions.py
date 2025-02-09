import os
import torch
from loguru import logger

def analyze_voice_content(tensor):
    """Analyze the content distribution in the voice tensor."""
    # Look at the variance along the first dimension to see where the information is concentrated
    variance = torch.var(tensor, dim=(1,2))  # Variance across features
    logger.info(f"Variance distribution:")
    logger.info(f"First 5 rows variance: {variance[:5].mean().item():.6f}")
    logger.info(f"Last 5 rows variance: {variance[-5:].mean().item():.6f}")
    return variance

def trim_voice_tensor(tensor):
    """Trim a 511x1x256 tensor to 510x1x256 by removing the row with least impact."""
    if tensor.shape[0] != 511:
        raise ValueError(f"Expected tensor with first dimension 511, got {tensor.shape[0]}")
    
    # Analyze variance contribution of each row
    variance = analyze_voice_content(tensor)
    
    # Determine which end has lower variance (less information)
    start_var = variance[:5].mean().item()
    end_var = variance[-5:].mean().item()
    
    # Remove from the end with lower variance
    if end_var < start_var:
        logger.info("Trimming last row (lower variance at end)")
        return tensor[:-1]
    else:
        logger.info("Trimming first row (lower variance at start)")
        return tensor[1:]

def process_voice_file(file_path):
    """Process a single voice file."""
    try:
        tensor = torch.load(file_path, map_location="cpu")
        if tensor.shape[0] != 511:
            logger.info(f"Skipping {os.path.basename(file_path)} - already correct shape {tensor.shape}")
            return False
            
        logger.info(f"\nProcessing {os.path.basename(file_path)}:")
        logger.info(f"Original shape: {tensor.shape}")
        
        # Create backup
        backup_path = file_path + ".backup"
        if not os.path.exists(backup_path):
            torch.save(tensor, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        # Trim tensor
        trimmed = trim_voice_tensor(tensor)
        logger.info(f"New shape: {trimmed.shape}")
        
        # Save trimmed tensor
        torch.save(trimmed, file_path)
        logger.info(f"Saved trimmed tensor to {file_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def main():
    """Process voice files in the voices directory."""
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    voices_dir = os.path.join(project_root, "api", "src", "voices", "v1_0")
    
    logger.info(f"Processing voices in: {voices_dir}")
    
    processed = 0
    for file in os.listdir(voices_dir):
        if file.endswith('.pt') and not file.endswith('.backup'):
            file_path = os.path.join(voices_dir, file)
            if process_voice_file(file_path):
                processed += 1
    
    logger.info(f"\nProcessed {processed} voice files")
    logger.info("Backups created with .backup extension")
    logger.info("To restore backups if needed, remove .backup extension to replace trimmed files")

if __name__ == "__main__":
    main()