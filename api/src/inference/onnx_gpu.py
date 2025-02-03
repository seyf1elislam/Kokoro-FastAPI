"""GPU-based ONNX inference backend."""

from typing import Optional

import numpy as np
import torch
from loguru import logger
from onnxruntime import InferenceSession

from ..core import paths
from ..core.model_config import model_config
from .base import BaseModelBackend
from .session_pool import create_session_options, create_provider_options


class ONNXGPUBackend(BaseModelBackend):
    """ONNX-based GPU inference backend."""

    def __init__(self):
        """Initialize GPU backend."""
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        self._device = "cuda"
        self._session: Optional[InferenceSession] = None
        
        # Configure GPU
        torch.cuda.set_device(model_config.onnx_gpu.device_id)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._session is not None

    async def load_model(self, path: str) -> None:
        """Load ONNX model.
        
        Args:
            path: Path to model file
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get verified model path
            model_path = await paths.get_model_path(path)
            
            logger.info(f"Loading ONNX model on GPU: {model_path}")
            
            # Configure session
            options = create_session_options(is_gpu=True)
            provider_options = create_provider_options(is_gpu=True)
            
            # Create session with CUDA provider
            self._session = InferenceSession(
                model_path,
                sess_options=options,
                providers=["CUDAExecutionProvider"],
                provider_options=[provider_options]
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def generate(
        self,
        tokens: list[int],
        voice: torch.Tensor,
        speed: float = 1.0
    ) -> np.ndarray:
        """Generate audio using ONNX model.
        
        Args:
            tokens: Input token IDs
            voice: Voice embedding tensor
            speed: Speed multiplier
            
        Returns:
            Generated audio samples
            
        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Prepare inputs
            tokens_input = np.array([[0, *tokens, 0]], dtype=np.int64)  # Add start/end tokens
            # Use modulo to ensure index stays within voice tensor bounds
            style_idx = (len(tokens) + 2) % voice.size(0)  # Add 2 for start/end tokens
            style_input = voice[style_idx].cpu().numpy()  # Move to CPU for ONNX
            speed_input = np.full(1, speed, dtype=np.float32)

            # Run inference
            result = self._session.run(
                None,
                {
                    "tokens": tokens_input,
                    "style": style_input,
                    "speed": speed_input
                }
            )
            
            return result[0]
            
        except Exception as e:
            if "out of memory" in str(e).lower():
                # Clear CUDA cache and retry
                torch.cuda.empty_cache()
                return self.generate(tokens, voice, speed)
            raise RuntimeError(f"Generation failed: {e}")

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._session is not None:
            del self._session
            self._session = None
            torch.cuda.empty_cache()