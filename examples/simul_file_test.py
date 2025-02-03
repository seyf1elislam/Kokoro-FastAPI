#!/usr/bin/env rye run python
import asyncio
import time
from pathlib import Path
from openai import AsyncOpenAI

# Initialize async client
openai = AsyncOpenAI(base_url="http://localhost:8880/v1", api_key="not-needed-for-local")

async def save_to_file(text: str, file_id: int) -> None:
    """Save TTS output to file asynchronously"""
    speech_file_path = Path(__file__).parent / f"speech_{file_id}.mp3"
    
    start_time = time.time()
    print(f"Starting file {file_id}")
    
    try:
        # Use streaming endpoint with mp3 format
        async with openai.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice="af_bella",
            input=text,
            response_format="mp3"
        ) as response:
            print(f"File {file_id} - Time to first byte: {int((time.time() - start_time) * 1000)}ms")
            
            # Open file in binary write mode
            with open(speech_file_path, 'wb') as f:
                async for chunk in response.iter_bytes():
                    f.write(chunk)
            
            print(f"File {file_id} completed in {int((time.time() - start_time) * 1000)}ms")
    except Exception as e:
        print(f"Error processing file {file_id}: {e}")

async def main() -> None:
    # Different text samples for variety
    texts = [
        "The quick brown fox jumped over the lazy dogs. I see skies of blue and clouds of white",
        "I see skies of blue and clouds of white. I see skies of blue and clouds of white",
    ]
    
    # Create tasks for saving to files
    file_tasks = [
        save_to_file(text, i) 
        for i, text in enumerate(texts)
    ]
    
    # Run file tasks concurrently
    await asyncio.gather(*file_tasks)

if __name__ == "__main__":
    asyncio.run(main())