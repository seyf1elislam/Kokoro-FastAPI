"""CPU-based ONNX inference backend."""

from typing import Optional

import numpy as np
import torch
from loguru import logger
from onnxruntime import InferenceSession

from ..core import paths
from ..core.model_config import model_config
from .base import BaseModelBackend
from .session_pool import create_session_options, create_provider_options


class ONNXCPUBackend(BaseModelBackend):
    """ONNX-based CPU inference backend."""

    def __init__(self):
        """Initialize CPU backend."""
        super().__init__()
        self._device = "cpu"
        self._session: Optional[InferenceSession] = None

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
            
            logger.info(f"Loading ONNX model: {model_path}")
            
            # Configure session
            options = create_session_options(is_gpu=False)
            provider_options = create_provider_options(is_gpu=False)
            
            # Create session
            self._session = InferenceSession(
                model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
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
            # Prepare inputs with start/end tokens
            tokens_input = np.array([[0, *tokens, 0]], dtype=np.int64)  # Add start/end tokens
            style_input = voice[len(tokens) + 2].numpy()  # Adjust index for start/end tokens
            speed_input = np.full(1, speed, dtype=np.float32)

            # Build base inputs
            inputs = {
                "style": style_input,
                "speed": speed_input
            }
            
            # Try both possible token input names #TODO: 
            for token_name in ["tokens", "input_ids"]:
                try:
                    inputs[token_name] = tokens_input
                    result = self._session.run(None, inputs)
                    return result[0]
                except Exception:
                    del inputs[token_name]
                    continue
                    
            raise RuntimeError("Model does not accept either 'tokens' or 'input_ids' as input name")
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._session is not None:
            del self._session
            self._session = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()