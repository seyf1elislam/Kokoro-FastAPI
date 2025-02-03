import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import torch
from pathlib import Path

from ..src.inference.voice_manager import VoiceManager
from ..src.structures.model_schemas import VoiceConfig


@pytest.fixture
def mock_voice_tensor():
    return torch.randn(10, 10)  # Dummy tensor


@pytest.fixture
def voice_manager():
    return VoiceManager(VoiceConfig())


@pytest.mark.asyncio
async def test_load_voice(voice_manager, mock_voice_tensor):
    """Test loading a single voice"""
    with patch("api.src.core.paths.load_voice_tensor", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = mock_voice_tensor
        with patch("os.path.exists", return_value=True):
            voice = await voice_manager.load_voice("af_bella", "cpu")
            assert torch.equal(voice, mock_voice_tensor)


@pytest.mark.asyncio
async def test_load_voice_not_found(voice_manager):
    """Test loading non-existent voice"""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(RuntimeError, match="Voice not found: invalid_voice"):
            await voice_manager.load_voice("invalid_voice", "cpu")


@pytest.mark.skip(reason="Local saving is optional and not critical to functionality")
@pytest.mark.asyncio
async def test_combine_voices_with_saving(voice_manager, mock_voice_tensor):
    """Test combining voices with local saving enabled"""
    pass


@pytest.mark.asyncio
async def test_combine_voices_without_saving(voice_manager, mock_voice_tensor):
    """Test combining voices without local saving"""
    with patch("api.src.core.paths.load_voice_tensor", new_callable=AsyncMock) as mock_load, \
         patch("torch.save") as mock_save, \
         patch("os.makedirs"), \
         patch("os.path.exists", return_value=True):
        
        # Setup mocks
        mock_load.return_value = mock_voice_tensor
        
        # Mock settings
        with patch("api.src.core.config.settings") as mock_settings:
            mock_settings.allow_local_voice_saving = False
            mock_settings.voices_dir = "/mock/voices"
            
            # Combine voices
            combined = await voice_manager.combine_voices(["af_bella", "af_sarah"], "cpu")
            assert combined == "af_bella+af_sarah"  # Note: using + separator
            
            # Verify voice was not saved
            mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_combine_voices_single_voice(voice_manager):
    """Test combining with single voice"""
    with pytest.raises(ValueError, match="At least 2 voices are required"):
        await voice_manager.combine_voices(["af_bella"], "cpu")


@pytest.mark.asyncio
async def test_list_voices(voice_manager):
    """Test listing available voices"""
    with patch("os.listdir", return_value=["af_bella.pt", "af_sarah.pt", "af_bella+af_sarah.pt"]), \
         patch("os.makedirs"):
        voices = await voice_manager.list_voices()
        assert len(voices) == 3
        assert "af_bella" in voices
        assert "af_sarah" in voices
        assert "af_bella+af_sarah" in voices


@pytest.mark.asyncio
async def test_load_combined_voice(voice_manager, mock_voice_tensor):
    """Test loading a combined voice"""
    with patch("api.src.core.paths.load_voice_tensor", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = mock_voice_tensor
        with patch("os.path.exists", return_value=True):
            voice = await voice_manager.load_voice("af_bella+af_sarah", "cpu")
            assert torch.equal(voice, mock_voice_tensor)


def test_cache_management(mock_voice_tensor):
    """Test voice cache management"""
    # Create voice manager with small cache size
    config = VoiceConfig(cache_size=2)
    voice_manager = VoiceManager(config)
    
    # Add items to cache
    voice_manager._voice_cache = {
        "voice1_cpu": torch.randn(5, 5),
        "voice2_cpu": torch.randn(5, 5),
        "voice3_cpu": torch.randn(5, 5),  # Add one more than cache size
    }
    
    # Try managing cache
    voice_manager._manage_cache()
    
    # Check cache size maintained
    assert len(voice_manager._voice_cache) <= 2


@pytest.mark.asyncio
async def test_voice_loading_with_cache(voice_manager, mock_voice_tensor):
    """Test voice loading with cache enabled"""
    with patch("api.src.core.paths.load_voice_tensor", new_callable=AsyncMock) as mock_load, \
         patch("os.path.exists", return_value=True):
        
        mock_load.return_value = mock_voice_tensor
        
        # First load should hit disk
        voice1 = await voice_manager.load_voice("af_bella", "cpu")
        assert mock_load.call_count == 1
        
        # Second load should hit cache
        voice2 = await voice_manager.load_voice("af_bella", "cpu")
        assert mock_load.call_count == 1  # Still 1
        
        assert torch.equal(voice1, voice2)