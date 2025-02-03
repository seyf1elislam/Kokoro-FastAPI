"""Model management and caching."""

import asyncio
from typing import Dict, Optional, Tuple

import torch
from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend
from .onnx_cpu import ONNXCPUBackend
from .onnx_gpu import ONNXGPUBackend
from .pytorch_backend import PyTorchBackend
from .session_pool import CPUSessionPool, StreamingSessionPool


# Global singleton instance and lock for thread-safe initialization
_manager_instance = None
_manager_lock = asyncio.Lock()

class ModelManager:
    """Manages model loading and inference across backends."""
    # Class-level state for shared resources
    _loaded_models = {}
    _backends = {}
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize model manager.
        Note:
            This should not be called directly. Use get_manager() instead.
        """
        self._config = config or model_config
        
        # Initialize session pools
        self._session_pools = {
            'onnx_cpu': CPUSessionPool(),
            'onnx_gpu': StreamingSessionPool()
        }
        
        # Initialize locks
        self._backend_locks: Dict[str, asyncio.Lock] = {}

    def _determine_device(self) -> str:
        """Determine device based on settings."""
        if settings.use_gpu and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    async def initialize(self) -> None:
        """Initialize backends."""
        if self._backends:
            logger.debug("Using existing backend instances")
            return
            
        device = self._determine_device()
        
        try:
            if device == "cuda":
                if settings.use_onnx:
                    self._backends['onnx_gpu'] = ONNXGPUBackend()
                    self._current_backend = 'onnx_gpu'
                    logger.info("Initialized new ONNX GPU backend")
                else:
                    self._backends['pytorch'] = PyTorchBackend()
                    self._current_backend = 'pytorch'
                    logger.info("Initialized new PyTorch backend on GPU")
            else:
                if settings.use_onnx:
                    self._backends['onnx_cpu'] = ONNXCPUBackend()
                    self._current_backend = 'onnx_cpu'
                    logger.info("Initialized new ONNX CPU backend")
                else:
                    self._backends['pytorch'] = PyTorchBackend()
                    self._current_backend = 'pytorch'
                    logger.info("Initialized new PyTorch backend on CPU")
                    
            # Initialize locks for each backend
            for backend in self._backends:
                self._backend_locks[backend] = asyncio.Lock()
                
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            raise RuntimeError("Failed to initialize backend")

    async def initialize_with_warmup(self, voice_manager) -> tuple[str, str, int]:
        """Initialize model with warmup and pre-cache voices.
        Args:
            voice_manager: Voice manager instance for loading voices
        Returns:
            Tuple of (device type, model type, number of loaded voices)
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Determine backend type based on settings
            if settings.use_onnx:
                backend_type = 'onnx_gpu' if settings.use_gpu and torch.cuda.is_available() else 'onnx_cpu'
            else:
                backend_type = 'pytorch'

            # Get backend
            backend = self.get_backend(backend_type)
            
            # Get and verify model path
            model_file = model_config.pytorch_model_file if not settings.use_onnx else model_config.onnx_model_file
            model_path = await paths.get_model_path(model_file)
            
            if not await paths.verify_model_path(model_path):
                raise RuntimeError(f"Model file not found: {model_path}")

            # Pre-cache default voice and use for warmup
            warmup_voice = await voice_manager.load_voice(
                settings.default_voice, device=backend.device)
            logger.info(f"Pre-cached voice {settings.default_voice} for warmup")
            
            # Initialize model with warmup voice
            await self.load_model(model_path, warmup_voice, backend_type)

            # Only pre-cache default voice to avoid memory bloat
            logger.info(f"Using {settings.default_voice} as warmup voice")

            # Get available voices count
            voices = await voice_manager.list_voices()
            voicepack_count = len(voices)

            # Get device info for return
            device = "GPU" if settings.use_gpu else "CPU"
            model = "ONNX" if settings.use_onnx else "PyTorch"

            return device, model, voicepack_count

        except Exception as e:
            logger.error(f"Failed to initialize model with warmup: {e}")
            raise RuntimeError(f"Failed to initialize model with warmup: {e}")

    def get_backend(self, backend_type: Optional[str] = None) -> BaseModelBackend:
        """Get specified backend.
        Args:
            backend_type: Backend type ('pytorch_cpu', 'pytorch_gpu', 'onnx_cpu', 'onnx_gpu'),
                         uses default if None
        Returns:
            Model backend instance
        Raises:
            ValueError: If backend type is invalid
            RuntimeError: If no backends are available
        """
        if not self._backends:
            raise RuntimeError("No backends available")

        if backend_type is None:
            backend_type = self._current_backend
        
        if backend_type not in self._backends:
            raise ValueError(
                f"Invalid backend type: {backend_type}. "
                f"Available backends: {', '.join(self._backends.keys())}"
            )

        return self._backends[backend_type]

    def _determine_backend(self, model_path: str) -> str:
        """Determine appropriate backend based on model file and settings.
        Args:
            model_path: Path to model file
        Returns:
            Backend type to use
        """
        # If ONNX is preferred or model is ONNX format
        if settings.use_onnx or model_path.lower().endswith('.onnx'):
            return 'onnx_gpu' if settings.use_gpu and torch.cuda.is_available() else 'onnx_cpu'
        return 'pytorch'

    async def load_model(
        self,
        model_path: str,
        warmup_voice: Optional[torch.Tensor] = None,
        backend_type: Optional[str] = None
    ) -> None:
        """Load model on specified backend.
        Args:
            model_path: Path to model file
            warmup_voice: Optional voice tensor for warmup, skips warmup if None
            backend_type: Backend to load on, uses default if None
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get absolute model path
            abs_path = await paths.get_model_path(model_path)
            
            # Auto-determine backend if not specified
            if backend_type is None:
                backend_type = self._determine_backend(abs_path)
            
            # Get backend lock
            lock = self._backend_locks[backend_type]
            
            async with lock:
                backend = self.get_backend(backend_type)
                
                # For ONNX backends, use session pool
                if backend_type.startswith('onnx'):
                    pool = self._session_pools[backend_type]
                    backend._session = await pool.get_session(abs_path)
                    self._loaded_models[backend_type] = abs_path
                    logger.info(f"Fetched model instance from {backend_type} pool")
                    
                # For PyTorch backends, load normally
                else:
                    # Check if model is already loaded
                    if (backend_type in self._loaded_models and 
                        self._loaded_models[backend_type] == abs_path and
                        backend.is_loaded):
                        logger.info(f"Fetching existing model instance from {backend_type}")
                        return
                        
                    # Load model
                    await backend.load_model(abs_path)
                    self._loaded_models[backend_type] = abs_path
                    logger.info(f"Initialized new model instance on {backend_type}")
                
                # Run warmup if voice provided
                if warmup_voice is not None:
                    await self._warmup_inference(backend, warmup_voice)
            
        except Exception as e:
            # Clear cached path on failure
            self._loaded_models.pop(backend_type, None)
            raise RuntimeError(f"Failed to load model: {e}")

    async def _warmup_inference(self, backend: BaseModelBackend, voice: torch.Tensor) -> None:
        """Run warmup inference to initialize model.
        
        Args:
            backend: Model backend to warm up
            voice: Voice tensor already loaded on correct device
        """
        try:
            # Import here to avoid circular imports
            from ..services.text_processing import process_text
            
            # Use real text
            text = "Testing text to speech synthesis."
            
            # Process through pipeline
            tokens = process_text(text)
            if not tokens:
                raise ValueError("Text processing failed")
            
            # Run inference
            backend.generate(tokens, voice, speed=1.0)
            logger.debug("Completed warmup inference")
            
        except Exception as e:
            logger.warning(f"Warmup inference failed: {e}")
            raise

    async def generate(
        self,
        tokens: list[int],
        voice: torch.Tensor,
        speed: float = 1.0,
        backend_type: Optional[str] = None
    ) -> torch.Tensor:
        """Generate audio using specified backend.
        
        Args:
            tokens: Input token IDs
            voice: Voice tensor already loaded on correct device
            speed: Speed multiplier
            backend_type: Backend to use, uses default if None
            
        Returns:
            Generated audio tensor
            
        Raises:
            RuntimeError: If generation fails
        """
        backend = self.get_backend(backend_type)
        if not backend.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Generate audio using provided voice tensor
            # No lock needed here since inference is thread-safe
            return backend.generate(tokens, voice, speed)
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def unload_all(self) -> None:
        """Unload models from all backends and clear cache."""
        # Clean up session pools
        for pool in self._session_pools.values():
            pool.cleanup()
            
        # Unload PyTorch backends
        for backend in self._backends.values():
            backend.unload()
            
        self._loaded_models.clear()
        logger.info("Unloaded all models and cleared cache")

    @property
    def available_backends(self) -> list[str]:
        """Get list of available backends.
        """
        return list(self._backends.keys())

    @property
    def current_backend(self) -> str:
        """Get current default backend.
        """
        return self._current_backend


async def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Get global model manager instance.
    Args:
        config: Optional model configuration
    Returns:
        ModelManager instance
    Thread Safety:
        This function should be thread-safe. Lemme know if it unravels on you
    """
    global _manager_instance
    
    # Fast path - return existing instance without lock
    if _manager_instance is not None:
        return _manager_instance
        
    # Slow path - create new instance with lock
    async with _manager_lock:
        # Double-check pattern
        if _manager_instance is None:
            _manager_instance = ModelManager(config)
            await _manager_instance.initialize()
        return _manager_instance
    
