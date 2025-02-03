#!/usr/bin/env python3
"""Benchmark script for unified streaming implementation"""

import asyncio
import time
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt

# Initialize OpenAI client
client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

TEST_TEXTS = {
    "short": "The quick brown fox jumps over the lazy dog.",
    "medium": """In a bustling city, life moves at a rapid pace. 
                People hurry along the sidewalks, while cars navigate 
                through the busy streets. The air is filled with the 
                sounds of urban activity.""",
    "long": """The technological revolution has transformed how we live and work. 
              From artificial intelligence to renewable energy, innovations continue 
              to shape our future. As we face global challenges, scientific advances 
              offer new solutions. The intersection of technology and human creativity 
              drives progress forward, opening new possibilities for tomorrow."""
}

async def benchmark_streaming(text_name: str, text: str) -> Tuple[float, float, int]:
    """Benchmark streaming performance
    
    Returns:
        Tuple of (time to first byte, total time, total bytes)
    """
    start_time = time.time()
    total_bytes = 0
    first_byte_time = None
    
    with client.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice="af_bella",
        response_format="pcm",
        input=text,
    ) as response:
        for chunk in response.iter_bytes(chunk_size=1024):
            if first_byte_time is None:
                first_byte_time = time.time() - start_time
            total_bytes += len(chunk)
    
    total_time = time.time() - start_time
    return first_byte_time, total_time, total_bytes

async def benchmark_non_streaming(text_name: str, text: str) -> Tuple[float, int]:
    """Benchmark non-streaming performance
    
    Returns:
        Tuple of (total time, total bytes)
    """
    start_time = time.time()
    speech_file = Path(__file__).parent / f"non_stream_{text_name}.mp3"
    
    with client.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice="af_bella",
        input=text,
    ) as response:
        response.stream_to_file(speech_file)
    
    total_time = time.time() - start_time
    total_bytes = speech_file.stat().st_size
    return total_time, total_bytes

def plot_results(results: dict):
    """Plot benchmark results"""
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    text_lengths = [len(text) for text in TEST_TEXTS.values()]
    streaming_times = [r["streaming"]["total_time"] for r in results.values()]
    non_streaming_times = [r["non_streaming"]["total_time"] for r in results.values()]
    first_byte_times = [r["streaming"]["first_byte_time"] for r in results.values()]
    
    # Plot times
    x = np.arange(len(TEST_TEXTS))
    width = 0.25
    
    plt.bar(x - width, streaming_times, width, label='Streaming Total Time')
    plt.bar(x, non_streaming_times, width, label='Non-Streaming Total Time')
    plt.bar(x + width, first_byte_times, width, label='Time to First Byte')
    
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Time (seconds)')
    plt.title('Unified Streaming Performance Comparison')
    plt.xticks(x, text_lengths)
    plt.legend()
    
    # Save plot
    plt.savefig(Path(__file__).parent / 'benchmark_results.png')
    plt.close()

async def main():
    """Run benchmarks"""
    print("Starting unified streaming benchmarks...")
    
    results = {}
    
    for name, text in TEST_TEXTS.items():
        print(f"\nTesting {name} text ({len(text)} chars)...")
        
        # Test streaming
        print("Running streaming test...")
        first_byte_time, stream_total_time, stream_bytes = await benchmark_streaming(name, text)
        
        # Test non-streaming
        print("Running non-streaming test...")
        non_stream_total_time, non_stream_bytes = await benchmark_non_streaming(name, text)
        
        results[name] = {
            "text_length": len(text),
            "streaming": {
                "first_byte_time": first_byte_time,
                "total_time": stream_total_time,
                "total_bytes": stream_bytes,
                "throughput": stream_bytes / stream_total_time / 1024  # KB/s
            },
            "non_streaming": {
                "total_time": non_stream_total_time,
                "total_bytes": non_stream_bytes,
                "throughput": non_stream_bytes / non_stream_total_time / 1024  # KB/s
            }
        }
        
        # Print results for this test
        print(f"\nResults for {name} text:")
        print(f"Streaming:")
        print(f"  Time to first byte: {first_byte_time:.3f}s")
        print(f"  Total time: {stream_total_time:.3f}s")
        print(f"  Throughput: {stream_bytes/stream_total_time/1024:.1f} KB/s")
        print(f"Non-streaming:")
        print(f"  Total time: {non_stream_total_time:.3f}s")
        print(f"  Throughput: {non_stream_bytes/non_stream_total_time/1024:.1f} KB/s")
    
    # Plot results
    plot_results(results)
    print("\nBenchmark results have been plotted to benchmark_results.png")

if __name__ == "__main__":
    asyncio.run(main())