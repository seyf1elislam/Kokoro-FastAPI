"""Audio conversion service with proper streaming support"""

from io import BytesIO
import struct
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger
from pydub import AudioSegment

class StreamingAudioWriter:
    """Handles streaming audio format conversions"""

    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.buffer = BytesIO()

        # Format-specific setup
        if self.format == "wav":
            self._write_wav_header_initial()
        elif self.format in ["ogg", "opus"]:
            # For OGG/Opus, write to memory buffer
            self.writer = sf.SoundFile(
                file=self.buffer,
                mode='w',
                samplerate=sample_rate,
                channels=channels,
                format='OGG',
                subtype='VORBIS' if self.format == "ogg" else "OPUS"
            )
        elif self.format == "flac":
            # For FLAC, write to memory buffer
            self.writer = sf.SoundFile(
                file=self.buffer,
                mode='w',
                samplerate=sample_rate,
                channels=channels,
                format='FLAC'
            )
        elif self.format in ["mp3", "aac"]:
            # For MP3/AAC, we'll use pydub's incremental writer
            self.segments = []  # Store segments until we have enough data
            self.total_duration = 0  # Track total duration in milliseconds
            # Initialize an empty AudioSegment as our encoder
            self.encoder = AudioSegment.silent(duration=0, frame_rate=self.sample_rate)
        elif self.format == "pcm":
            # PCM doesn't need initialization, we'll write raw bytes
            pass
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _write_wav_header_initial(self) -> None:
        """Write initial WAV header with placeholders"""
        self.buffer.write(b'RIFF')
        self.buffer.write(struct.pack('<L', 0))  # Placeholder for file size
        self.buffer.write(b'WAVE')
        self.buffer.write(b'fmt ')
        self.buffer.write(struct.pack('<L', 16))  # fmt chunk size
        self.buffer.write(struct.pack('<H', 1))   # PCM format
        self.buffer.write(struct.pack('<H', self.channels))
        self.buffer.write(struct.pack('<L', self.sample_rate))
        self.buffer.write(struct.pack('<L', self.sample_rate * self.channels * 2))  # Byte rate
        self.buffer.write(struct.pack('<H', self.channels * 2))  # Block align
        self.buffer.write(struct.pack('<H', 16))  # Bits per sample
        self.buffer.write(b'data')
        self.buffer.write(struct.pack('<L', 0))  # Placeholder for data size

    def write_chunk(self, audio_data: Optional[np.ndarray] = None, finalize: bool = False) -> bytes:
        """Write a chunk of audio data and return bytes in the target format.
        
        Args:
            audio_data: Audio data to write, or None if finalizing
            finalize: Whether this is the final write to close the stream
        """
        output_buffer = BytesIO()

        if finalize:
            if self.format == "wav":
                # Calculate actual file and data sizes
                file_size = self.bytes_written + 36  # RIFF header bytes
                data_size = self.bytes_written

                # Seek to the beginning to overwrite the placeholders
                self.buffer.seek(4)
                self.buffer.write(struct.pack('<L', file_size))
                self.buffer.seek(40)
                self.buffer.write(struct.pack('<L', data_size))

                self.buffer.seek(0)
                return self.buffer.read()
            elif self.format in ["ogg", "opus", "flac"]:
                self.writer.close()
                return self.buffer.getvalue()
            elif self.format in ["mp3", "aac"]:
                if hasattr(self, 'encoder') and len(self.encoder) > 0:
                    format_args = {
                        "mp3": {"format": "mp3", "codec": "libmp3lame"},
                        "aac": {"format": "adts", "codec": "aac"}
                    }[self.format]
                    
                    parameters = []
                    if self.format == "mp3":
                        parameters.extend([
                            "-q:a", "2",
                            "-write_xing", "1",  # XING header for MP3
                            "-id3v1", "1",
                            "-id3v2", "1",
                            "-write_vbr", "1",
                            "-vbr_quality", "2"
                        ])
                    elif self.format == "aac":
                        parameters.extend([
                            "-q:a", "2",
                            "-write_xing", "0",
                            "-write_id3v1", "0",
                            "-write_id3v2", "0"
                        ])
                    
                    self.encoder.export(
                        output_buffer,
                        **format_args,
                        bitrate="192k",
                        parameters=parameters
                    )
                    self.encoder = None

                return output_buffer.getvalue()
        
        if audio_data is None or len(audio_data) == 0:
            return b''

        if self.format == "wav":
            # Write raw PCM data
            self.buffer.write(audio_data.tobytes())
            self.bytes_written += len(audio_data.tobytes())
            return b''
        
        elif self.format in ["ogg", "opus", "flac"]:
            # Write to soundfile buffer
            self.writer.write(audio_data)
            self.writer.flush()
            return self.buffer.getvalue()
        
        elif self.format in ["mp3", "aac"]:
            # Convert chunk to AudioSegment and encode
            segment = AudioSegment(
                audio_data.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=audio_data.dtype.itemsize,
                channels=self.channels
            )
            
            # Track total duration
            self.total_duration += len(segment)
            
            # Add segment to encoder
            self.encoder += segment
            
            # Export current state to buffer without final metadata
            format_args = {
                "mp3": {"format": "mp3", "codec": "libmp3lame"},
                "aac": {"format": "adts", "codec": "aac"}
            }[self.format]
            
            # For chunks, export without duration metadata or XING headers
            self.encoder.export(output_buffer, **format_args, bitrate="192k", parameters=[
                "-q:a", "2",
                "-write_xing", "0"  # No XING headers for chunks
            ])
            
            # Get the encoded data
            encoded_data = output_buffer.getvalue()
            
            # Reset encoder to prevent memory growth
            self.encoder = AudioSegment.silent(duration=0, frame_rate=self.sample_rate)
            
            return encoded_data
        
        elif self.format == "pcm":
            # Write raw bytes
            return audio_data.tobytes()

        return b''

    def close(self) -> Optional[bytes]:
        """Finish the audio file and return any remaining data"""
        if self.format == "wav":
            # Re-finalize WAV file by updating headers
            self.buffer.seek(0)
            file_content = self.write_chunk(finalize=True)
            return file_content

        elif self.format in ["ogg", "opus", "flac"]:
            # Finalize other formats
            self.writer.close()
            return self.buffer.getvalue()

        elif self.format in ["mp3", "aac"]:
            # Finalize MP3/AAC
            final_data = self.write_chunk(finalize=True)
            return final_data

        return None