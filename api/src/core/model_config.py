"""Model configuration schemas."""

from pydantic import BaseModel, Field


class ONNXCPUConfig(BaseModel):
    """ONNX CPU runtime configuration."""
    
    # Session pooling
    max_instances: int = Field(4, description="Maximum concurrent model instances")
    instance_timeout: int = Field(60, description="Session timeout in seconds")
    
    # Runtime settings
    num_threads: int = Field(8, description="Number of threads for parallel operations")
    inter_op_threads: int = Field(4, description="Number of threads for operator parallelism")
    execution_mode: str = Field("parallel", description="ONNX execution mode")
    optimization_level: str = Field("all", description="ONNX optimization level")
    memory_pattern: bool = Field(True, description="Enable memory pattern optimization")
    arena_extend_strategy: str = Field("kNextPowerOfTwo", description="Memory arena strategy")

    class Config:
        frozen = True


class ONNXGPUConfig(ONNXCPUConfig):
    """ONNX GPU-specific configuration."""
    
    # CUDA settings
    device_id: int = Field(0, description="CUDA device ID")
    gpu_mem_limit: float = Field(0.5, description="Fraction of GPU memory to use")
    cudnn_conv_algo_search: str = Field("EXHAUSTIVE", description="CuDNN convolution algorithm search")
    
    # Stream management
    cuda_streams: int = Field(2, description="Number of CUDA streams for inference")
    stream_timeout: int = Field(60, description="Stream timeout in seconds")
    do_copy_in_default_stream: bool = Field(True, description="Copy in default CUDA stream")

    class Config:
        frozen = True


class PyTorchCPUConfig(BaseModel):
    """PyTorch CPU backend configuration."""
    
    memory_threshold: float = Field(0.8, description="Memory threshold for cleanup")
    retry_on_oom: bool = Field(True, description="Whether to retry on OOM errors")
    num_threads: int = Field(8, description="Number of threads for parallel operations")
    pin_memory: bool = Field(True, description="Whether to pin memory for faster CPU-GPU transfer")

    class Config:
        frozen = True


class PyTorchGPUConfig(BaseModel):
    """PyTorch GPU backend configuration."""
    
    device_id: int = Field(0, description="CUDA device ID")
    use_triton: bool = Field(True, description="Whether to use Triton for CUDA kernels")
    memory_threshold: float = Field(0.8, description="Memory threshold for cleanup")
    retry_on_oom: bool = Field(True, description="Whether to retry on OOM errors")
    sync_cuda: bool = Field(True, description="Whether to synchronize CUDA operations")
    cuda_streams: int = Field(2, description="Number of CUDA streams for inference")
    stream_timeout: int = Field(60, description="Stream timeout in seconds")

    class Config:
        frozen = True


class ModelConfig(BaseModel):
    """Model configuration."""
    
    # General settings
    model_type: str = Field("pytorch", description="Model type ('pytorch' or 'onnx')")
    device_type: str = Field("auto", description="Device type ('cpu', 'gpu', or 'auto')")
    cache_models: bool = Field(True, description="Whether to cache loaded models")
    cache_voices: bool = Field(True, description="Whether to cache voice tensors")
    voice_cache_size: int = Field(2, description="Maximum number of cached voices")
    
    # Model filenames
    pytorch_model_file: str = Field("kokoro-v0_19-half.pth", description="PyTorch model filename")
    onnx_model_file: str = Field("kokoro-v0_19.onnx", description="ONNX model filename")
    
    # Backend-specific configs
    onnx_cpu: ONNXCPUConfig = Field(default_factory=ONNXCPUConfig)
    onnx_gpu: ONNXGPUConfig = Field(default_factory=ONNXGPUConfig)
    pytorch_cpu: PyTorchCPUConfig = Field(default_factory=PyTorchCPUConfig)
    pytorch_gpu: PyTorchGPUConfig = Field(default_factory=PyTorchGPUConfig)

    class Config:
        frozen = True

    def get_backend_config(self, backend_type: str):
        """Get configuration for specific backend.
        
        Args:
            backend_type: Backend type ('pytorch_cpu', 'pytorch_gpu', 'onnx_cpu', 'onnx_gpu')
            
        Returns:
            Backend-specific configuration
            
        Raises:
            ValueError: If backend type is invalid
        """
        if backend_type not in {
            'pytorch_cpu', 'pytorch_gpu', 'onnx_cpu', 'onnx_gpu'
        }:
            raise ValueError(f"Invalid backend type: {backend_type}")
            
        return getattr(self, backend_type)


# Global instance
model_config = ModelConfig()