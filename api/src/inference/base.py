"""Base interfaces for model inference."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch


class ModelBackend(ABC):
    """Abstract base class for model inference backends."""

    @abstractmethod
    async def load_model(self, path: str) -> None:
        """Load model from path.
        
        Args:
            path: Path to model file
            
        Raises:
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    def generate(
        self,
        tokens: List[int],
        voice: torch.Tensor,
        speed: float = 1.0
    ) -> np.ndarray:
        """Generate audio from tokens.
        
        Args:
            tokens: Input token IDs
            voice: Voice embedding tensor
            speed: Speed multiplier
            
        Returns:
            Generated audio samples
            
        Raises:
            RuntimeError: If generation fails
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Get device model is running on.
        
        Returns:
            Device string ('cpu' or 'cuda')
        """
        pass


class BaseModelBackend(ModelBackend):
    """Base implementation of model backend."""

    def __init__(self):
        """Initialize base backend."""
        self._model: Optional[torch.nn.Module] = None
        self._device: str = "cpu"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._device

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()