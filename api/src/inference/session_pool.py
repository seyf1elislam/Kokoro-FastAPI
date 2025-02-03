"""Session pooling for model inference."""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set

import torch
from loguru import logger
from onnxruntime import (
    ExecutionMode,
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions
)

from ..core import paths
from ..core.model_config import model_config


@dataclass
class SessionInfo:
    """Session information."""
    session: InferenceSession
    last_used: float
    stream_id: Optional[int] = None


def create_session_options(is_gpu: bool = False) -> SessionOptions:
    """Create ONNX session options.
    
    Args:
        is_gpu: Whether to use GPU configuration
        
    Returns:
        Configured session options
    """
    options = SessionOptions()
    config = model_config.onnx_gpu if is_gpu else model_config.onnx_cpu
    
    # Set optimization level
    if config.optimization_level == "all":
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    elif config.optimization_level == "basic":
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
    else:
        options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    
    # Configure threading
    options.intra_op_num_threads = config.num_threads
    options.inter_op_num_threads = config.inter_op_threads
    
    # Set execution mode
    options.execution_mode = (
        ExecutionMode.ORT_PARALLEL
        if config.execution_mode == "parallel"
        else ExecutionMode.ORT_SEQUENTIAL
    )
    
    # Configure memory optimization
    options.enable_mem_pattern = config.memory_pattern
    
    return options


def create_provider_options(is_gpu: bool = False) -> Dict:
    """Create provider options.
    
    Args:
        is_gpu: Whether to use GPU configuration
        
    Returns:
        Provider configuration
    """
    if is_gpu:
        config = model_config.onnx_gpu
        return {
            "device_id": config.device_id,
            "arena_extend_strategy": config.arena_extend_strategy,
            "gpu_mem_limit": int(config.gpu_mem_limit * torch.cuda.get_device_properties(0).total_memory),
            "cudnn_conv_algo_search": config.cudnn_conv_algo_search,
            "do_copy_in_default_stream": config.do_copy_in_default_stream
        }
    else:
        return {
            "arena_extend_strategy": model_config.onnx_cpu.arena_extend_strategy,
            "cpu_memory_arena_cfg": "cpu:0"
        }


class BaseSessionPool:
    """Base session pool implementation."""
    
    def __init__(self, max_size: int, timeout: int):
        """Initialize session pool.
        
        Args:
            max_size: Maximum number of concurrent sessions
            timeout: Session timeout in seconds
        """
        self._max_size = max_size
        self._timeout = timeout
        self._sessions: Dict[str, SessionInfo] = {}
        self._lock = asyncio.Lock()
        
    async def get_session(self, model_path: str) -> InferenceSession:
        """Get session from pool.
        
        Args:
            model_path: Path to model file
            
        Returns:
            ONNX inference session
            
        Raises:
            RuntimeError: If no sessions available
        """
        async with self._lock:
            # Clean expired sessions
            self._cleanup_expired()
            
            # TODO: Change session tracking to use unique IDs instead of model paths
            # This would allow multiple instances of the same model
            
            # Check if session exists and is valid
            if model_path in self._sessions:
                session_info = self._sessions[model_path]
                session_info.last_used = time.time()
                return session_info.session
                
            # TODO: Modify session limit check to count instances per model path
            # Rather than total sessions across all models
            if len(self._sessions) >= self._max_size:
                raise RuntimeError(
                    f"Maximum number of sessions reached ({self._max_size}). "
                    "Try again later or reduce concurrent requests."
                )
                
            # Create new session
            session = await self._create_session(model_path)
            self._sessions[model_path] = SessionInfo(
                session=session,
                last_used=time.time()
            )
            return session
            
    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        current_time = time.time()
        expired = [
            path for path, info in self._sessions.items()
            if current_time - info.last_used > self._timeout
        ]
        for path in expired:
            logger.info(f"Removing expired session: {path}")
            del self._sessions[path]
            
    async def _create_session(self, model_path: str) -> InferenceSession:
        """Create new session.
        
        Args:
            model_path: Path to model file
            
        Returns:
            ONNX inference session
        """
        raise NotImplementedError
        
    def cleanup(self) -> None:
        """Clean up all sessions."""
        self._sessions.clear()


class StreamingSessionPool(BaseSessionPool):
    """GPU session pool with CUDA streams."""
    
    def __init__(self):
        """Initialize GPU session pool."""
        config = model_config.onnx_gpu
        super().__init__(config.cuda_streams, config.stream_timeout)
        self._available_streams: Set[int] = set(range(config.cuda_streams))
        
    async def get_session(self, model_path: str) -> InferenceSession:
        """Get session with CUDA stream.
        
        Args:
            model_path: Path to model file
            
        Returns:
            ONNX inference session
            
        Raises:
            RuntimeError: If no streams available
        """
        async with self._lock:
            # Clean expired sessions
            self._cleanup_expired()
            
            # Try to find existing session
            if model_path in self._sessions:
                session_info = self._sessions[model_path]
                session_info.last_used = time.time()
                return session_info.session
                
            # Get available stream
            if not self._available_streams:
                raise RuntimeError("No CUDA streams available")
            stream_id = self._available_streams.pop()
            
            try:
                # Create new session
                session = await self._create_session(model_path)
                self._sessions[model_path] = SessionInfo(
                    session=session,
                    last_used=time.time(),
                    stream_id=stream_id
                )
                return session
                
            except Exception:
                # Return stream to pool on failure
                self._available_streams.add(stream_id)
                raise
                
    def _cleanup_expired(self) -> None:
        """Remove expired sessions and return streams."""
        current_time = time.time()
        expired = [
            path for path, info in self._sessions.items()
            if current_time - info.last_used > self._timeout
        ]
        for path in expired:
            info = self._sessions[path]
            if info.stream_id is not None:
                self._available_streams.add(info.stream_id)
            logger.info(f"Removing expired session: {path}")
            del self._sessions[path]
            
    async def _create_session(self, model_path: str) -> InferenceSession:
        """Create new session with CUDA provider."""
        abs_path = await paths.get_model_path(model_path)
        options = create_session_options(is_gpu=True)
        provider_options = create_provider_options(is_gpu=True)
        
        return InferenceSession(
            abs_path,
            sess_options=options,
            providers=["CUDAExecutionProvider"],
            provider_options=[provider_options]
        )


class CPUSessionPool(BaseSessionPool):
    """CPU session pool."""
    
    def __init__(self):
        """Initialize CPU session pool."""
        config = model_config.onnx_cpu
        super().__init__(config.max_instances, config.instance_timeout)
        
    async def _create_session(self, model_path: str) -> InferenceSession:
        """Create new session with CPU provider."""
        abs_path = await paths.get_model_path(model_path)
        options = create_session_options(is_gpu=False)
        provider_options = create_provider_options(is_gpu=False)
        
        return InferenceSession(
            abs_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
            provider_options=[provider_options]
        )