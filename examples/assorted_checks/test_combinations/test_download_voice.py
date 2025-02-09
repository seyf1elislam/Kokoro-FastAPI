#!/usr/bin/env python3
import os
from pathlib import Path
import requests

# Create output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

def download_combined_voice(voice1: str, voice2: str, weights: tuple[float, float] = None) -> str:
    """Download a combined voice file.
    
    Args:
        voice1: First voice name
        voice2: Second voice name
        weights: Optional tuple of weights (w1, w2). If not provided, uses equal weights.
    
    Returns:
        Path to downloaded .pt file
    """
    print(f"\nDownloading combined voice: {voice1} + {voice2}")
    
    # Construct voice string with optional weights
    if weights:
        voice_str = f"{voice1}({weights[0]})+{voice2}({weights[1]})"
    else:
        voice_str = f"{voice1}+{voice2}"
    
    # Make the request to combine voices
    response = requests.post(
        "http://localhost:8880/v1/audio/voices/combine",
        json=voice_str
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to combine voices: {response.text}")
    
    # Save the .pt file
    output_path = output_dir / f"{voice_str}.pt"
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    print(f"Saved combined voice to {output_path}")
    return str(output_path)

def main():
    # Test downloading various voice combinations
    combinations = [
        # Equal weights (default)
        ("af_bella", "af_kore"),
        
        # Different weight combinations
        ("af_bella", "af_kore", (0.2, 0.8)),
        ("af_bella", "af_kore", (0.8, 0.2)),
        ("af_bella", "af_kore", (0.5, 0.5)),
        
        # Test with different voices
        ("af_bella", "af_jadzia"),
        ("af_bella", "af_jadzia", (0.3, 0.7))
    ]
    
    for combo in combinations:
        try:
            if len(combo) == 3:
                voice1, voice2, weights = combo
                download_combined_voice(voice1, voice2, weights)
            else:
                voice1, voice2 = combo
                download_combined_voice(voice1, voice2)
        except Exception as e:
            print(f"Error downloading combination {combo}: {e}")

if __name__ == "__main__":
    main()