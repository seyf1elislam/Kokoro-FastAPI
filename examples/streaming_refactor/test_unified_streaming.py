#!/usr/bin/env python3
"""Test script for unified streaming implementation"""

import asyncio
import time
from pathlib import Path

from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

async def test_streaming_to_file():
    """Test streaming to file"""
    print("\nTesting streaming to file...")
    speech_file = Path(__file__).parent / "stream_output.mp3"
    
    start_time = time.time()
    with client.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice="af_bella",
        input="Testing unified streaming implementation with a short phrase.",
    ) as response:
        response.stream_to_file(speech_file)
    
    print(f"Streaming to file completed in {(time.time() - start_time):.2f}s")
    print(f"Output saved to: {speech_file}")

async def test_streaming_chunks():
    """Test streaming chunks for real-time playback"""
    print("\nTesting chunk streaming...")
    
    start_time = time.time()
    chunk_count = 0
    total_bytes = 0
    
    with client.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice="af_bella",
        response_format="pcm",
        input="""This is a longer text to test chunk streaming.
                We want to verify that the unified streaming implementation
                works efficiently for both small and large inputs.""",
    ) as response:
        print(f"Time to first byte: {(time.time() - start_time):.3f}s")
        
        for chunk in response.iter_bytes(chunk_size=1024):
            chunk_count += 1
            total_bytes += len(chunk)
            # In real usage, this would go to audio playback
            # For testing, we just count chunks and bytes
    
    total_time = time.time() - start_time
    print(f"Received {chunk_count} chunks, {total_bytes} bytes")
    print(f"Total streaming time: {total_time:.2f}s")
    print(f"Average throughput: {total_bytes/total_time/1024:.1f} KB/s")

async def main():
    """Run all tests"""
    print("Starting unified streaming tests...")
    
    # Test both streaming modes
    await test_streaming_to_file()
    await test_streaming_chunks()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    asyncio.run(main())