"""TTS service using model and voice managers."""

import time
from typing import List, Tuple, Optional, AsyncGenerator, Union
import asyncio

import numpy as np
import torch
from loguru import logger

from ..core.config import settings
from ..inference.model_manager import get_manager as get_model_manager
from ..inference.voice_manager import get_manager as get_voice_manager
from .audio import AudioNormalizer, AudioService
from .text_processing.text_processor import process_text_chunk, smart_split
from .text_processing import tokenize

class TTSService:
    """Text-to-speech service."""

    # Limit concurrent chunk processing
    _chunk_semaphore = asyncio.Semaphore(4)

    def __init__(self, output_dir: str = None):
        """Initialize service."""
        self.output_dir = output_dir
        self.model_manager = None
        self._voice_manager = None

    @classmethod
    async def create(cls, output_dir: str = None) -> 'TTSService':
        """Create and initialize TTSService instance."""
        service = cls(output_dir)
        service.model_manager = await get_model_manager()
        service._voice_manager = await get_voice_manager()
        return service

    async def _process_chunk(
        self,
        tokens: List[int],
        voice_tensor: torch.Tensor,
        speed: float,
        output_format: Optional[str] = None,
        is_first: bool = False,
        is_last: bool = False,
        normalizer: Optional[AudioNormalizer] = None,
    ) -> Optional[Union[np.ndarray, bytes]]:
        """Process tokens into audio."""
        async with self._chunk_semaphore:
            try:
                # Handle stream finalization
                if is_last:
                    # Skip format conversion for raw audio mode
                    if not output_format:
                        return np.array([], dtype=np.float32)
                    
                    return await AudioService.convert_audio(
                        np.array([0], dtype=np.float32),  # Dummy data for type checking
                        24000,
                        output_format,
                        is_first_chunk=False,
                        normalizer=normalizer,
                        is_last_chunk=True
                    )
                
                # Skip empty chunks
                if not tokens:
                    return None

                # Generate audio using pre-warmed model
                chunk_audio = await self.model_manager.generate(
                    tokens,
                    voice_tensor,
                    speed=speed
                )
                
                if chunk_audio is None:
                    logger.error("Model generated None for audio chunk")
                    return None
                
                if len(chunk_audio) == 0:
                    logger.error("Model generated empty audio chunk")
                    return None
                    
                # For streaming, convert to bytes
                if output_format:
                    try:
                        return await AudioService.convert_audio(
                            chunk_audio,
                            24000,
                            output_format,
                            is_first_chunk=is_first,
                            normalizer=normalizer,
                            is_last_chunk=is_last
                        )
                    except Exception as e:
                        logger.error(f"Failed to convert audio: {str(e)}")
                        return None
                    
                return chunk_audio
            except Exception as e:
                logger.error(f"Failed to process tokens: {str(e)}")
                return None

    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "wav",
    ) -> AsyncGenerator[bytes, None]:
        """Generate and stream audio chunks."""
        stream_normalizer = AudioNormalizer()
        voice_tensor = None
        chunk_index = 0
        
        try:
            # Get backend and load voice (should be fast if cached)
            backend = self.model_manager.get_backend()
            voice_tensor = await self._voice_manager.load_voice(voice, device=backend.device)

            # Process text in chunks with smart splitting
            async for chunk_text, tokens in smart_split(text):
                try:
                    # Process audio for chunk
                    result = await self._process_chunk(
                        tokens,  # Now always a flat List[int]
                        voice_tensor,
                        speed,
                        output_format,
                        is_first=(chunk_index == 0),
                        is_last=False,  # We'll update the last chunk later
                        normalizer=stream_normalizer
                    )
                    
                    if result is not None:
                        yield result
                        chunk_index += 1
                    else:
                        logger.warning(f"No audio generated for chunk: '{chunk_text[:100]}...'")
                        
                except Exception as e:
                    logger.error(f"Failed to process audio for chunk: '{chunk_text[:100]}...'. Error: {str(e)}")
                    continue

            # Only finalize if we successfully processed at least one chunk
            if chunk_index > 0:
                try:
                    # Empty tokens list to finalize audio
                    final_result = await self._process_chunk(
                        [],  # Empty tokens list
                        voice_tensor,
                        speed,
                        output_format,
                        is_first=False,
                        is_last=True,
                        normalizer=stream_normalizer
                    )
                    if final_result is not None:
                        logger.debug("Yielding final chunk to finalize audio")
                        yield final_result
                    else:
                        logger.warning("Final chunk processing returned None")
                except Exception as e:
                    logger.error(f"Failed to process final chunk: {str(e)}")
            else:
                logger.warning("No audio chunks were successfully processed")

        except Exception as e:
            logger.error(f"Error in audio generation stream: {str(e)}")
            raise
        finally:
            if voice_tensor is not None:
                del voice_tensor
                torch.cuda.empty_cache()

    async def generate_from_phonemes(
        self, phonemes: str, voice: str, speed: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Generate audio from phonemes.
        
        Args:
            phonemes: Phoneme string to synthesize
            voice: Voice ID to use
            speed: Speed multiplier
            
        Returns:
            Tuple of (audio array, processing time)
        """
        start_time = time.time()
        voice_tensor = None

        try:
            # Get backend and load voice
            backend = self.model_manager.get_backend()
            voice_tensor = await self._voice_manager.load_voice(voice, device=backend.device)

            # Convert phonemes to tokens
            tokens = tokenize(phonemes)
            if len(tokens) > 500:  # Model context limit
                raise ValueError(f"Phoneme sequence too long ({len(tokens)} tokens, max 500)")
                
            tokens = [0] + tokens + [0]  # Add start/end tokens
            
            # Generate audio
            audio = await self.model_manager.generate(
                tokens,
                voice_tensor,
                speed=speed
            )
            
            if audio is None:
                raise ValueError("Failed to generate audio")

            processing_time = time.time() - start_time
            return audio, processing_time

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
            raise
        finally:
            if voice_tensor is not None:
                del voice_tensor
                torch.cuda.empty_cache()

    async def generate_audio(
        self, text: str, voice: str, speed: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Generate complete audio for text using streaming internally."""
        start_time = time.time()
        chunks = []
        
        try:
            # Use streaming generator but collect all valid chunks
            async for chunk in self.generate_audio_stream(
                text, voice, speed,  # Default to WAV for raw audio
            ):
                if chunk is not None:
                    chunks.append(chunk)

            if not chunks:
                raise ValueError("No audio chunks were generated successfully")

            # Combine chunks, ensuring we have valid arrays
            if len(chunks) == 1:
                audio = chunks[0]
            else:
                # Filter out any zero-dimensional arrays
                valid_chunks = [c for c in chunks if c.ndim > 0]
                if not valid_chunks:
                    raise ValueError("No valid audio chunks to concatenate")
                audio = np.concatenate(valid_chunks)
            processing_time = time.time() - start_time
            return audio, processing_time

        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise

    async def combine_voices(self, voices: List[str]) -> str:
        """Combine multiple voices."""
        return await self._voice_manager.combine_voices(voices)

    async def list_voices(self) -> List[str]:
        """List available voices."""
        return await self._voice_manager.list_voices()