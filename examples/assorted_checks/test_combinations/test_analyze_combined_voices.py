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

def plot_comparison(analyses, output_dir):
    """Create detailed comparison plots of the audio analyses."""
    plt.style.use('dark_background')
    
    # Plot waveforms
    fig_wave = plt.figure(figsize=(15, 10))
    fig_wave.patch.set_facecolor('#1a1a2e')
    
    for i, (name, data) in enumerate(analyses.items()):
        ax = plt.subplot(len(analyses), 1, i+1)
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
    plt.savefig(output_dir / 'waveforms.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot spectral characteristics
    fig_spec = plt.figure(figsize=(15, 10))
    fig_spec.patch.set_facecolor('#1a1a2e')
    
    for i, (name, data) in enumerate(analyses.items()):
        # Calculate spectrogram
        samples = data['samples']
        sample_rate = data['sample_rate']
        nperseg = 2048
        f, t, Sxx = plt.mlab.specgram(samples, NFFT=2048, Fs=sample_rate, 
                                     noverlap=nperseg//2, scale='dB')
        
        ax = plt.subplot(len(analyses), 1, i+1)
        plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma')
        plt.title(f"Spectrogram: {name}", color='white', pad=20)
        plt.ylabel('Frequency [Hz]', color='white')
        plt.xlabel('Time [sec]', color='white')
        plt.colorbar(label='Intensity [dB]')
        ax.set_facecolor('#1a1a2e')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spectrograms.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot voice characteristics comparison
    fig_chars = plt.figure(figsize=(15, 8))
    fig_chars.patch.set_facecolor('#1a1a2e')
    
    # Extract characteristics
    names = list(analyses.keys())
    rms_values = [data['rms'] for data in analyses.values()]
    centroids = [data['spectral_centroid'] for data in analyses.values()]
    max_amps = [data['max_amplitude'] for data in analyses.values()]
    
    # Plot characteristics
    x = np.arange(len(names))
    width = 0.25
    
    ax = plt.subplot(111)
    ax.bar(x - width, rms_values, width, label='RMS (Texture)', color='#ff2a6d')
    ax.bar(x, [c/1000 for c in centroids], width, label='Spectral Centroid/1000 (Brightness)', color='#05d9e8')
    ax.bar(x + width, max_amps, width, label='Max Amplitude', color='#ff65bd')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_title('Voice Characteristics Comparison', color='white', pad=20)
    ax.set_facecolor('#1a1a2e')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved comparison plots to {output_dir}")

def main():
    # Test different voice combinations with weights
    voices = {
        'af_bella': output_dir / 'af_bella.wav',
        'af_kore': output_dir / 'af_kore.wav',
        'af_bella(0.2)+af_kore(0.8)': output_dir / 'af_bella_20_af_kore_80.wav',
        'af_bella(0.8)+af_kore(0.2)': output_dir / 'af_bella_80_af_kore_20.wav',
        'af_bella(0.5)+af_kore(0.5)': output_dir / 'af_bella_50_af_kore_50.wav'
    }
    
    # Generate audio for each voice/combination
    for voice, path in voices.items():
        try:
            generate_and_save_audio(voice, str(path))
        except Exception as e:
            print(f"Error generating audio for {voice}: {e}")
            continue
    
    # Analyze each audio file
    analyses = {}
    for name, path in voices.items():
        try:
            analyses[name] = analyze_audio(str(path))
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            continue
    
    # Create comparison plots
    if analyses:
        plot_comparison(analyses, output_dir)
    else:
        print("No analyses to plot")

if __name__ == "__main__":
    main()
