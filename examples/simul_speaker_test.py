#!/usr/bin/env rye run python
import asyncio
import time
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
    
    # Run speaker tasks concurrently
    try:
        await asyncio.gather(*speaker_tasks)
    finally:
        # Clean up PyAudio
        p.terminate()

if __name__ == "__main__":
    asyncio.run(main())