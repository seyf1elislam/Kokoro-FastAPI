from typing import List

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from ..services.audio import AudioService, AudioNormalizer
from ..services.streaming_audio_writer import StreamingAudioWriter
from ..services.text_processing import smart_split
from kokoro import KPipeline
from ..services.tts_service import TTSService
from ..structures.text_schemas import (
    GenerateFromPhonemesRequest,
    PhonemeRequest,
    PhonemeResponse,
)
from ..structures import (
    CaptionedSpeechRequest,
    CaptionedSpeechResponse,
    WordTimestamp
)

router = APIRouter(tags=["text processing"])


async def get_tts_service() -> TTSService:
    """Dependency to get TTSService instance"""
    return await TTSService.create()  # Create service with properly initialized managers

@router.post("/dev/phonemize", response_model=PhonemeResponse)
async def phonemize_text(request: PhonemeRequest) -> PhonemeResponse:
    """Convert text to phonemes using Kokoro's quiet mode.

    Args:
        request: Request containing text and language

    Returns:
        Phonemes and token IDs
    """
    try:
        if not request.text:
            raise ValueError("Text cannot be empty")

        # Initialize Kokoro pipeline in quiet mode (no model)
        pipeline = KPipeline(lang_code=request.language, model=False)
        
        # Get first result from pipeline (we only need one since we're not chunking)
        for result in pipeline(request.text):
            # result.graphemes = original text
            # result.phonemes = phonemized text
            # result.tokens = token objects (if available)
            return PhonemeResponse(phonemes=result.phonemes, tokens=[])

        raise ValueError("Failed to generate phonemes")
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
    """Generate audio directly from phonemes using Kokoro's phoneme format"""
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
                # Generate audio from phonemes
                chunk_audio, _ = await tts_service.generate_from_phonemes(
                    phonemes=request.phonemes,  # Pass complete phoneme string
                    voice=request.voice,
                    speed=1.0
                )
                
                if chunk_audio is not None:
                    # Normalize audio before writing
                    normalized_audio = await normalizer.normalize(chunk_audio)
                    # Write chunk and yield bytes
                    chunk_bytes = writer.write_chunk(normalized_audio)
                    if chunk_bytes:
                        yield chunk_bytes
                    
                    # Finalize and yield remaining bytes
                    final_bytes = writer.write_chunk(finalize=True)
                    if final_bytes:
                        yield final_bytes
                else:
                    raise ValueError("Failed to generate audio data")
                    
            except Exception as e:
                logger.error(f"Error in audio generation: {str(e)}")
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

@router.post("/dev/captioned_speech")
async def create_captioned_speech(
    request: CaptionedSpeechRequest,
    tts_service: TTSService = Depends(get_tts_service),
) -> StreamingResponse:
    """Generate audio with word-level timestamps using Kokoro's output"""
    try:
        # Get voice path
        voice_name, voice_path = await tts_service._get_voice_path(request.voice)

        # Generate audio with timestamps
        audio, _, word_timestamps = await tts_service.generate_audio(
            text=request.input,
            voice=voice_name,
            speed=request.speed,
            return_timestamps=True
        )

        # Create streaming audio writer
        writer = StreamingAudioWriter(format=request.response_format, sample_rate=24000, channels=1)
        normalizer = AudioNormalizer()

        async def generate_chunks():
            try:
                if audio is not None:
                    # Normalize audio before writing
                    normalized_audio = await normalizer.normalize(audio)
                    # Write chunk and yield bytes
                    chunk_bytes = writer.write_chunk(normalized_audio)
                    if chunk_bytes:
                        yield chunk_bytes
                    
                    # Finalize and yield remaining bytes
                    final_bytes = writer.write_chunk(finalize=True)
                    if final_bytes:
                        yield final_bytes
                else:
                    raise ValueError("Failed to generate audio data")
                    
            except Exception as e:
                logger.error(f"Error in audio generation: {str(e)}")
                # Clean up writer on error
                writer.write_chunk(finalize=True)
                # Re-raise the original exception
                raise

        # Convert timestamps to JSON and add as header
        import json
        logger.debug(f"Processing {len(word_timestamps)} word timestamps")
        timestamps_json = json.dumps([{
            'word': str(ts['word']),  # Ensure string for text
            'start_time': float(ts['start_time']),  # Ensure float for timestamps
            'end_time': float(ts['end_time'])
        } for ts in word_timestamps])
        logger.debug(f"Generated timestamps JSON: {timestamps_json}")

        return StreamingResponse(
            generate_chunks(),
            media_type=f"audio/{request.response_format}",
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
                "X-Word-Timestamps": timestamps_json
            }
        )

    except ValueError as e:
        logger.error(f"Error in captioned speech generation: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error"
            }
        )
    except Exception as e:
        logger.error(f"Error in captioned speech generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error"
            }
        )
