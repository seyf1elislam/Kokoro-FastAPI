from typing import List

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from ..services.audio import AudioService, AudioNormalizer
from ..services.streaming_audio_writer import StreamingAudioWriter
from ..services.text_processing import phonemize, smart_split
from ..services.text_processing.vocabulary import tokenize
from ..services.tts_service import TTSService
from ..structures.text_schemas import (
    GenerateFromPhonemesRequest,
    PhonemeRequest,
    PhonemeResponse,
)

router = APIRouter(tags=["text processing"])


async def get_tts_service() -> TTSService:
    """Dependency to get TTSService instance"""
    return await TTSService.create()  # Create service with properly initialized managers

@router.post("/dev/phonemize", response_model=PhonemeResponse)
async def phonemize_text(request: PhonemeRequest) -> PhonemeResponse:
    """Convert text to phonemes and tokens

    Args:
        request: Request containing text and language
        tts_service: Injected TTSService instance

    Returns:
        Phonemes and token IDs
    """
    try:
        if not request.text:
            raise ValueError("Text cannot be empty")

        # Get phonemes
        phonemes = phonemize(request.text, request.language)
        if not phonemes:
            raise ValueError("Failed to generate phonemes")

        # Get tokens (without adding start/end tokens to match process_text behavior)
        tokens = tokenize(phonemes)
        return PhonemeResponse(phonemes=phonemes, tokens=tokens)
    except ValueError as e:
        logger.error(f"Error in phoneme generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error in phoneme generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )
@router.post("/dev/generate_from_phonemes")
async def generate_from_phonemes(
    request: GenerateFromPhonemesRequest,
    client_request: Request,
    tts_service: TTSService = Depends(get_tts_service),
) -> StreamingResponse:
    """Generate audio directly from phonemes with proper streaming"""
    try:
        # Basic validation
        if not isinstance(request.phonemes, str):
            raise ValueError("Phonemes must be a string")
        if not request.phonemes:
            raise ValueError("Phonemes cannot be empty")

        # Create streaming audio writer and normalizer
        writer = StreamingAudioWriter(format="wav", sample_rate=24000, channels=1)
        normalizer = AudioNormalizer()

        async def generate_chunks():
            try:
                has_data = False
                # Process phonemes in chunks
                async for chunk_text, _ in smart_split(request.phonemes):
                    # Check if client is still connected
                    is_disconnected = client_request.is_disconnected
                    if callable(is_disconnected):
                        is_disconnected = await is_disconnected()
                    if is_disconnected:
                        logger.info("Client disconnected, stopping audio generation")
                        break

                    chunk_audio, _ = await tts_service.generate_from_phonemes(
                        phonemes=chunk_text,
                        voice=request.voice,
                        speed=1.0
                    )
                    if chunk_audio is not None:
                        has_data = True
                        # Normalize audio before writing
                        normalized_audio = await normalizer.normalize(chunk_audio)
                        # Write chunk and yield bytes
                        chunk_bytes = writer.write_chunk(normalized_audio)
                        if chunk_bytes:
                            yield chunk_bytes

                if not has_data:
                    raise ValueError("Failed to generate any audio data")

                # Finalize and yield remaining bytes if we still have a connection
                if not (callable(is_disconnected) and await is_disconnected()):
                    final_bytes = writer.write_chunk(finalize=True)
                    if final_bytes:
                        yield final_bytes
            except Exception as e:
                logger.error(f"Error in audio chunk generation: {str(e)}")
                # Clean up writer on error
                writer.write_chunk(finalize=True)
                # Re-raise the original exception
                raise

        return StreamingResponse(
            generate_chunks(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked"
            }
        )


    except ValueError as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error"
            }
        )
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error"
            }
        )
