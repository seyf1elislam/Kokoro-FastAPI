#!/usr/bin/env rye run python
import asyncio
import time
from pathlib import Path
import pyaudio
from openai import AsyncOpenAI

# Initialize async client
openai = AsyncOpenAI(base_url="http://localhost:8880/v1", api_key="not-needed-for-local")

# Create a shared PyAudio instance
p = pyaudio.PyAudio()

async def stream_to_speakers(text: str, stream_id: int) -> None:
    """Stream TTS audio to speakers asynchronously"""
    player_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True
    )

    start_time = time.time()
    print(f"Starting stream {stream_id}")

    try:
        async with openai.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice="af_bella",
            response_format="pcm",
            input=text
        ) as response:
            print(f"Stream {stream_id} - Time to first byte: {int((time.time() - start_time) * 1000)}ms")
            
            async for chunk in response.iter_bytes(chunk_size=1024):
                player_stream.write(chunk)
                # Small sleep to allow other coroutines to run
                await asyncio.sleep(0.001)

        print(f"Stream {stream_id} completed in {int((time.time() - start_time) * 1000)}ms")
    
    finally:
        player_stream.stop_stream()
        player_stream.close()

async def save_to_file(text: str, file_id: int) -> None:
    """Save TTS output to file asynchronously"""
    speech_file_path = Path(__file__).parent / f"speech_{file_id}.mp3"
    
    async with openai.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice="af_bella",
        input=text
    ) as response:
        # Open file in binary write mode
        with open(speech_file_path, 'wb') as f:
            async for chunk in response.iter_bytes():
                f.write(chunk)
        print(f"File {file_id} saved to {speech_file_path}")

async def main() -> None:
    # Different text samples for variety
    texts = [
        "The quick brown fox jumped over the lazy dogs. I see skies of blue and clouds of white",
        "I see skies of blue and clouds of white. I see skies of blue and clouds of white",
    ]
    
    # Create tasks for streaming to speakers
    speaker_tasks = [
        stream_to_speakers(text, i) 
        for i, text in enumerate(texts)
    ]
    
    # Create tasks for saving to files
    file_tasks = [
        save_to_file(text, i) 
        for i, text in enumerate(texts)
    ]
    
    # Combine all tasks
    all_tasks = speaker_tasks + file_tasks
    
    # Run all tasks concurrently
    try:
        await asyncio.gather(*all_tasks)
    finally:
        # Clean up PyAudio
        p.terminate()

if __name__ == "__main__":
    asyncio.run(main())