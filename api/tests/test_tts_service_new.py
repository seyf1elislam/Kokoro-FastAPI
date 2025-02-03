# import pytest
# import numpy as np
# from unittest.mock import AsyncMock, patch

# @pytest.mark.asyncio
# async def test_generate_audio(tts_service, mock_audio_output, test_voice):
#     """Test basic audio generation"""
#     audio, processing_time = await tts_service.generate_audio(
#         text="Hello world",
#         voice=test_voice,
#         speed=1.0
#     )
    
#     assert isinstance(audio, np.ndarray)
#     assert audio == mock_audio_output.tobytes()
#     assert processing_time > 0
#     tts_service.model_manager.generate.assert_called_once()

# @pytest.mark.asyncio
# async def test_generate_audio_with_combined_voice(tts_service, mock_audio_output):
#     """Test audio generation with a combined voice"""
#     test_voices = ["voice1", "voice2"]
#     combined_id = await tts_service._voice_manager.combine_voices(test_voices)
    
#     audio, processing_time = await tts_service.generate_audio(
#         text="Hello world",
#         voice=combined_id,
#         speed=1.0
#     )
    
#     assert isinstance(audio, np.ndarray)
#     assert np.array_equal(audio, mock_audio_output)
#     assert processing_time > 0

# @pytest.mark.asyncio
# async def test_generate_audio_stream(tts_service, mock_audio_output, test_voice):
#     """Test streaming audio generation"""
#     tts_service.model_manager.generate.return_value = mock_audio_output
    
#     chunks = []
#     async for chunk in tts_service.generate_audio_stream(
#         text="Hello world",
#         voice=test_voice,
#         speed=1.0,
#         output_format="pcm"
#     ):
#         assert isinstance(chunk, bytes)
#         chunks.append(chunk)
    
#     assert len(chunks) > 0
#     tts_service.model_manager.generate.assert_called()

# @pytest.mark.asyncio
# async def test_empty_text(tts_service, test_voice):
#     """Test handling empty text"""
#     with pytest.raises(ValueError) as exc_info:
#         await tts_service.generate_audio(
#             text="",
#             voice=test_voice,
#             speed=1.0
#         )
#     assert "No audio chunks were generated successfully" in str(exc_info.value)

# @pytest.mark.asyncio
# async def test_invalid_voice(tts_service):
#     """Test handling invalid voice"""
#     tts_service._voice_manager.load_voice.side_effect = ValueError("Voice not found")
    
#     with pytest.raises(ValueError) as exc_info:
#         await tts_service.generate_audio(
#             text="Hello world",
#             voice="invalid_voice",
#             speed=1.0
#         )
#     assert "Voice not found" in str(exc_info.value)

# @pytest.mark.asyncio
# async def test_model_generation_error(tts_service, test_voice):
#     """Test handling model generation error"""
#     # Make generate return None to simulate failed generation
#     tts_service.model_manager.generate.return_value = None
    
#     with pytest.raises(ValueError) as exc_info:
#         await tts_service.generate_audio(
#             text="Hello world",
#             voice=test_voice,
#             speed=1.0
#         )
#     assert "No audio chunks were generated successfully" in str(exc_info.value)

# @pytest.mark.asyncio
# async def test_streaming_generation_error(tts_service, test_voice):
#     """Test handling streaming generation error"""
#     # Make generate return None to simulate failed generation
#     tts_service.model_manager.generate.return_value = None
    
#     chunks = []
#     async for chunk in tts_service.generate_audio_stream(
#         text="Hello world",
#         voice=test_voice,
#         speed=1.0,
#         output_format="pcm"
#     ):
#         chunks.append(chunk)
    
#     # Should get no chunks if generation fails
#     assert len(chunks) == 0

# @pytest.mark.asyncio
# async def test_list_voices(tts_service):
#     """Test listing available voices"""
#     voices = await tts_service.list_voices()
#     assert len(voices) == 2
#     assert "voice1" in voices
#     assert "voice2" in voices
#     tts_service._voice_manager.list_voices.assert_called_once()

# @pytest.mark.asyncio
# async def test_combine_voices(tts_service):
#     """Test combining voices"""
#     test_voices = ["voice1", "voice2"]
#     combined_id = await tts_service.combine_voices(test_voices)
#     assert combined_id == "voice1_voice2"
#     tts_service._voice_manager.combine_voices.assert_called_once_with(test_voices)

# @pytest.mark.asyncio
# async def test_chunked_text_processing(tts_service, test_voice, mock_audio_output):
#     """Test processing chunked text"""
#     # Create text that will force chunking by exceeding max tokens
#     long_text = "This is a test sentence." * 100  # Should be way over 500 tokens
    
#     # Don't mock smart_split - let it actually split the text
#     audio, processing_time = await tts_service.generate_audio(
#         text=long_text,
#         voice=test_voice,
#         speed=1.0
#     )
    
#     # Should be called multiple times due to chunking
#     assert tts_service.model_manager.generate.call_count > 1
#     assert isinstance(audio, np.ndarray)
#     assert processing_time > 0