"""Audio conversion service"""

from io import BytesIO
import struct

import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
from loguru import logger
from pydub import AudioSegment

from ..core.config import settings
from .streaming_audio_writer import StreamingAudioWriter

class AudioNormalizer:
    """Handles audio normalization state for a single stream"""

    def __init__(self):
        self.chunk_trim_ms = settings.gap_trim_ms
        self.sample_rate = 24000  # Sample rate of the audio
        self.samples_to_trim = int(self.chunk_trim_ms * self.sample_rate / 1000)

    async def normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert audio data to int16 range and trim silence from start and end
        
        Args:
            audio_data: Input audio data as numpy array
            
        Returns:
            Normalized and trimmed audio data
        """
        if len(audio_data) == 0:
            raise ValueError("Empty audio data")
            
        # Trim start and end if enough samples
        if len(audio_data) > (2 * self.samples_to_trim):
            audio_data = audio_data[self.samples_to_trim:-self.samples_to_trim]
        
        # Scale directly to int16 range with clipping
        return np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)


class AudioService:
    """Service for audio format conversions with streaming support"""

    # Supported formats
    SUPPORTED_FORMATS = {"wav", "mp3", "opus", "flac", "aac", "pcm", "ogg"}

    # Default audio format settings balanced for speed and compression
    DEFAULT_SETTINGS = {
        "mp3": {
            "bitrate_mode": "CONSTANT",  # Faster than variable bitrate
            "compression_level": 0.0,  # Balanced compression
        },
        "opus": {
            "compression_level": 0.0,  # Good balance for speech
        },
        "flac": {
            "compression_level": 0.0,  # Light compression, still fast
        },
        "aac": {
            "bitrate": "192k",  # Default AAC bitrate
        },
    }

    _writers = {}

    @staticmethod
    async def convert_audio(
        audio_data: np.ndarray,
        sample_rate: int,
        output_format: str,
        is_first_chunk: bool = True,
        is_last_chunk: bool = False,
        normalizer: AudioNormalizer = None,
    ) -> bytes:
        """Convert audio data to specified format with streaming support

        Args:
            audio_data: Numpy array of audio samples
            sample_rate: Sample rate of the audio
            output_format: Target format (wav, mp3, ogg, pcm)
            is_first_chunk: Whether this is the first chunk
            is_last_chunk: Whether this is the last chunk
            normalizer: Optional AudioNormalizer instance for consistent normalization

        Returns:
            Bytes of the converted audio chunk
        """
        try:
            # Validate format
            if output_format not in AudioService.SUPPORTED_FORMATS:
                raise ValueError(f"Format {output_format} not supported")

            # Always normalize audio to ensure proper amplitude scaling
            if normalizer is None:
                normalizer = AudioNormalizer()
            normalized_audio = await normalizer.normalize(audio_data)

            # Get or create format-specific writer
            writer_key = f"{output_format}_{sample_rate}"
            if is_first_chunk or writer_key not in AudioService._writers:
                AudioService._writers[writer_key] = StreamingAudioWriter(
                    output_format, sample_rate
                )
            writer = AudioService._writers[writer_key]

            # Write audio data first
            if len(normalized_audio) > 0:
                chunk_data = writer.write_chunk(normalized_audio)

            # Then finalize if this is the last chunk
            if is_last_chunk:
                final_data = writer.write_chunk(finalize=True)
                del AudioService._writers[writer_key]
                return final_data if final_data else b''
            
            return chunk_data if chunk_data else b''

        except Exception as e:
            logger.error(f"Error converting audio stream to {output_format}: {str(e)}")
            raise ValueError(f"Failed to convert audio stream to {output_format}: {str(e)}")
