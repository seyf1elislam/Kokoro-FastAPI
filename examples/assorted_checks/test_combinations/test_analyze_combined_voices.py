#!/usr/bin/env python3
import os
import time
import wave
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

# Create output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Initialize OpenAI client
client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

# Test text that showcases voice characteristics
text = """The quick brown fox jumps over the lazy dog.
         How vexingly quick daft zebras jump!
         The five boxing wizards jump quickly."""

def generate_and_save_audio(voice: str, output_path: str):
    """Generate audio using specified voice and save to WAV file."""
    print(f"\nGenerating audio for voice: {voice}")
    start_time = time.time()
    
    # Generate audio using streaming response
    with client.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice=voice,
        response_format="wav",
        input=text,
    ) as response:
        # Save the audio stream to file
        with open(output_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    
    duration = time.time() - start_time
    print(f"Generated in {duration:.2f}s")
    print(f"Saved to {output_path}")
    return output_path

def analyze_audio(filepath: str):
    """Analyze audio file and return key characteristics."""
    print(f"\nAnalyzing {filepath}")
    try:
        print(f"\nTrying to read {filepath}")
        with wave.open(filepath, 'rb') as wf:
            sample_rate = wf.getframerate()
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        print(f"Successfully read file:")
        print(f"Sample rate: {sample_rate}")
        print(f"Samples shape: {samples.shape}")
        print(f"Samples dtype: {samples.dtype}")
        print(f"First few samples: {samples[:10]}")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        raise
    
    # Convert to float64 for calculations
    samples = samples.astype(np.float64) / 32768.0  # Normalize 16-bit audio
    
    # Convert to mono if stereo
    if len(samples.shape) > 1:
        samples = np.mean(samples, axis=1)
    
    # Calculate basic stats
    duration = len(samples) / sample_rate
    max_amp = np.max(np.abs(samples))
    rms = np.sqrt(np.mean(samples**2))
    
    # Calculate frequency characteristics
    # Compute FFT
    N = len(samples)
    yf = np.fft.fft(samples)
    xf = np.fft.fftfreq(N, 1 / sample_rate)[:N//2]
    magnitude = 2.0/N * np.abs(yf[0:N//2])
    # Calculate spectral centroid
    spectral_centroid = np.sum(xf * magnitude) / np.sum(magnitude)
    # Determine dominant frequencies
    dominant_freqs = xf[magnitude.argsort()[-5:]][::-1].tolist()
    
    return {
        'samples': samples,
        'sample_rate': sample_rate,
        'duration': duration,
        'max_amplitude': max_amp,
        'rms': rms,
        'spectral_centroid': spectral_centroid,
        'dominant_frequencies': dominant_freqs
    }

def plot_comparison(analyses, output_path):
    """Create comparison plot of the audio analyses."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Plot waveforms
    for i, (name, data) in enumerate(analyses.items()):
        ax = plt.subplot(3, 1, i+1)
        samples = data['samples']
        time = np.arange(len(samples)) / data['sample_rate']
        plt.plot(time, samples / data['max_amplitude'], linewidth=0.5, color='#ff2a6d')
        plt.title(f"Waveform: {name}", color='white', pad=20)
        plt.xlabel("Time (seconds)", color='white')
        plt.ylabel("Normalized Amplitude", color='white')
        plt.grid(True, alpha=0.3)
        ax.set_facecolor('#1a1a2e')
        plt.ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_path}")

def main():
    # Generate audio for each voice
    voices = {
        'af_bella': output_dir / 'af_bella.wav',
        'af_irulan': output_dir / 'af_irulan.wav',
        'af_bella+af_irulan': output_dir / 'af_bella+af_irulan.wav'
    }
    
    for voice, path in voices.items():
        generate_and_save_audio(voice, str(path))
    
    # Analyze each audio file
    analyses = {}
    for name, path in voices.items():
        analyses[name] = analyze_audio(str(path))
    
    # Create comparison plot
    plot_comparison(analyses, output_dir / 'voice_comparison.png')

if __name__ == "__main__":
    main()
