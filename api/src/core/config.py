from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    api_title: str = "Kokoro TTS API"
    api_description: str = "API for text-to-speech generation using Kokoro"
    api_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8880

    # Application Settings
    output_dir: str = "output"
    output_dir_size_limit_mb: float = 500.0  # Maximum size of output directory in MB
    default_voice: str = "af_heart"
    use_gpu: bool = True  # Whether to use GPU acceleration if available
    allow_local_voice_saving: bool = False  # Whether to allow saving combined voices locally
    
    # Container absolute paths
    model_dir: str = "/app/api/src/models"  # Absolute path in container
    voices_dir: str = "/app/api/src/voices/v1_0"  # Absolute path in container
    
    # Audio Settings
    sample_rate: int = 24000
    # Text Processing Settings
    target_min_tokens: int = 175  # Target minimum tokens per chunk
    target_max_tokens: int = 250  # Target maximum tokens per chunk
    absolute_max_tokens: int = 450  # Absolute maximum tokens per chunk
    
    gap_trim_ms: int = 250  # Amount to trim from streaming chunk ends in milliseconds

    # Web Player Settings
    enable_web_player: bool = True  # Whether to serve the web player UI
    web_player_path: str = "web"  # Path to web player static files
    cors_origins: list[str] = ["*"]  # CORS origins for web player
    cors_enabled: bool = True  # Whether to enable CORS

    # Temp File Settings for WEB Ui
    temp_file_dir: str = "api/temp_files"  # Directory for temporary audio files (relative to project root)
    max_temp_dir_size_mb: int = 2048  # Maximum size of temp directory (2GB)
    max_temp_dir_age_hours: int = 1  # Remove temp files older than 1 hour
    max_temp_dir_count: int = 3  # Maximum number of temp files to keep

    #extra 
    public_api: bool = False
    public_api_retries: int = 3
    class Config:
        env_file = ".env"


settings = Settings()
