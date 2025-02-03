<p align="center">
  <img src="githubbanner.png" alt="Kokoro TTS Banner">
</p>

# <sub><sub>_`FastKoko`_ </sub></sub>
[![Tests](https://img.shields.io/badge/tests-100%20passed-darkgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-49%25-grey)]()
[![Tested at Model Commit](https://img.shields.io/badge/last--tested--model--commit-a67f113-blue)](https://huggingface.co/hexgrad/Kokoro-82M/tree/c3b0d86e2a980e027ef71c28819ea02e351c2667) [![Try on Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on-Spaces-blue)](https://huggingface.co/spaces/Remsky/Kokoro-TTS-Zero)

> Support for Kokoro-82M v1.0 coming very soon! Dev build on the `v0.1.5-integration` branch

Dockerized FastAPI wrapper for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech model
- OpenAI-compatible Speech endpoint, with inline voice combination, and mapped naming/models for strict systems
- NVIDIA GPU accelerated or CPU inference (ONNX or Pytorch for either)
- very fast generation time
  - ~35x-100x+ real time speed via 4060Ti+
  - ~5x+ real time speed via M3 Pro CPU
- streaming support & tempfile generation, phoneme based dev endpoints
- (new) Integrated web UI on localhost:8880/web
- (new) Debug endpoints for monitoring threads, storage, and session pools


## Get Started

<details>
<summary>Quickest Start (docker run)</summary>


Pre built images are available to run, with arm/multi-arch support, and baked in models
Refer to the core/config.py file for a full list of variables which can be managed via the environment

```bash

docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:v0.1.4 # CPU, or:
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:v0.1.4 #NVIDIA GPU
```


</details>

<details>

<summary>Quick Start (docker compose) </summary>

1. Install prerequisites, and start the service using Docker Compose (Full setup including UI):
   - Install [Docker](https://www.docker.com/products/docker-desktop/)
   - Clone the repository:
        ```bash
        git clone https://github.com/remsky/Kokoro-FastAPI.git
        cd Kokoro-FastAPI

        cd docker/gpu # OR
        # cd docker/cpu # Run this or the above
        docker compose up --build
        # if you are missing any models, run:
        # python ../scripts/download_model.py --type pth  # for GPU
        # python ../scripts/download_model.py --type onnx # for CPU
        ```

        ```bash
        Or directly via UV
        ./start-cpu.sh
        ./start-gpu.sh 
        ```
</details>
<details>
<summary>Direct Run (via uv) </summary>

1. Install prerequisites ():
   - Install [astral-uv](https://docs.astral.sh/uv/)
   - Clone the repository:
        ```bash
        git clone https://github.com/remsky/Kokoro-FastAPI.git
        cd Kokoro-FastAPI

        # if you are missing any models, run:
        # python ../scripts/download_model.py --type pth  # for GPU
        # python ../scripts/download_model.py --type onnx # for CPU
        ```

        Start directly via UV (with hot-reload)
        ```bash
        ./start-cpu.sh OR
        ./start-gpu.sh 
        ```

</details>

<details open>
<summary> Up and Running? </summary>


Run locally as an OpenAI-Compatible Speech Endpoint
    
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8880/v1", api_key="not-needed"
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_sky+af_bella", #single or multiple voicepack combo
    input="Hello world!"
  ) as response:
      response.stream_to_file("output.mp3")
```
  
- The API will be available at http://localhost:8880
- API Documentation: http://localhost:8880/docs

- Web Interface: http://localhost:8880/web
- Gradio UI (deprecating) can be accessed at http://localhost:7860 if enabled in docker compose file (it is a separate image!)

<div align="center" style="display: flex; justify-content: center; gap: 10px;">
  <img src="assets/docs-screenshot.png" width="40%" alt="API Documentation" style="border: 2px solid #333; padding: 10px;">
  <img src="assets/webui-screenshot.png" width="49%" alt="Web UI Screenshot" style="border: 2px solid #333; padding: 10px;">
</div>

</details>

## Features 
<details>
<summary>OpenAI-Compatible Speech Endpoint</summary>

```python
# Using OpenAI's Python library
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")
response = client.audio.speech.create(
    model="kokoro",  
    voice="af_bella+af_sky", # see /api/src/core/openai_mappings.json to customize
    input="Hello world!",
    response_format="mp3"
)

response.stream_to_file("output.mp3")
```
Or Via Requests:
```python
import requests


response = requests.get("http://localhost:8880/v1/audio/voices")
voices = response.json()["voices"]

# Generate audio
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "model": "kokoro",  
        "input": "Hello world!",
        "voice": "af_bella",
        "response_format": "mp3",  # Supported: mp3, wav, opus, flac
        "speed": 1.0
    }
)

# Save audio
with open("output.mp3", "wb") as f:
    f.write(response.content)
```

Quick tests (run from another terminal):
```bash
python examples/assorted_checks/test_openai/test_openai_tts.py # Test OpenAI Compatibility
python examples/assorted_checks/test_voices/test_all_voices.py # Test all available voices
```
</details>

<details>
<summary>Voice Combination</summary>

- Averages model weights of any existing voicepacks
- Saves generated voicepacks for future use
- (new) Available through any endpoint, simply concatenate desired packs with "+"

Combine voices and generate audio:
```python
import requests
response = requests.get("http://localhost:8880/v1/audio/voices")
voices = response.json()["voices"]

# Create combined voice (saves locally on server)
response = requests.post(
    "http://localhost:8880/v1/audio/voices/combine",
    json=[voices[0], voices[1]]
)
combined_voice = response.json()["voice"]

# Generate audio with combined voice (or, simply pass multiple directly with `+` )
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "Hello world!",
        "voice": combined_voice, # or skip the above step with f"{voices[0]}+{voices[1]}"
        "response_format": "mp3"
    }
)
```
<p align="center">
  <img src="assets/voice_analysis.png" width="80%" alt="Voice Analysis Comparison" style="border: 2px solid #333; padding: 10px;">
</p>
</details>

<details>
<summary>Multiple Output Audio Formats</summary>

- mp3
- wav
- opus 
- flac
- aac
- pcm

<p align="center">
<img src="assets/format_comparison.png" width="80%" alt="Audio Format Comparison" style="border: 2px solid #333; padding: 10px;">
</p>

</details>

<details>
<summary>Gradio Web Utility</summary>

Access the interactive web UI at http://localhost:7860 after starting the service. Features include:
- Voice/format/speed selection
- Audio playback and download
- Text file or direct input

If you only want the API, just comment out everything in the docker-compose.yml under and including `gradio-ui`

Currently, voices created via the API are accessible here, but voice combination/creation has not yet been added

Running the UI Docker Service [deprecating]
   - If you only want to run the Gradio web interface separately and connect it to an existing API service:
      ```bash
      docker run -p 7860:7860 \
        -e API_HOST=<api-hostname-or-ip> \
        -e API_PORT=8880 \
      ```

     - Replace `<api-hostname-or-ip>` with:
       - `kokoro-tts` if the UI container is running in the same Docker Compose setup.
       - `localhost` if the API is running on your local machine.
  
### Disabling Local Saving

You can disable local saving of audio files and hide the file view in the UI by setting the `DISABLE_LOCAL_SAVING` environment variable to `true`. This is useful when running the service on a server where you don't want to store generated audio files locally.

When using Docker Compose:
```yaml
environment:
  - DISABLE_LOCAL_SAVING=true
```

When running the Docker image directly:
```bash
docker run -p 7860:7860 -e DISABLE_LOCAL_SAVING=true ghcr.io/remsky/kokoro-fastapi-ui:v0.1.4
```
</details>

<details>
<summary>Streaming Support</summary>

```python
# OpenAI-compatible streaming
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8880/v1", api_key="not-needed")

# Stream to file
with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_bella",
    input="Hello world!"
) as response:
    response.stream_to_file("output.mp3")

# Stream to speakers (requires PyAudio)
import pyaudio
player = pyaudio.PyAudio().open(
    format=pyaudio.paInt16, 
    channels=1, 
    rate=24000, 
    output=True
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_bella",
    response_format="pcm",
    input="Hello world!"
) as response:
    for chunk in response.iter_bytes(chunk_size=1024):
        player.write(chunk)
```

Or via requests:
```python
import requests

response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "Hello world!",
        "voice": "af_bella",
        "response_format": "pcm"
    },
    stream=True
)

for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        # Process streaming chunks
        pass
```

<p align="center">
  <img src="assets/gpu_first_token_timeline_openai.png" width="45%" alt="GPU First Token Timeline" style="border: 2px solid #333; padding: 10px; margin-right: 1%;">
  <img src="assets/cpu_first_token_timeline_stream_openai.png" width="45%" alt="CPU First Token Timeline" style="border: 2px solid #333; padding: 10px;">
</p>

Key Streaming Metrics:
- First token latency @ chunksize
    - ~300ms  (GPU) @ 400 
    - ~3500ms (CPU) @ 200 (older i7)
    - ~<1s    (CPU) @ 200 (M3 Pro)
- Adjustable chunking settings for real-time playback 

*Note: Artifacts in intonation can increase with smaller chunks*
</details>

## Processing Details
<details>
<summary>Performance Benchmarks</summary>

Benchmarking was performed on generation via the local API using text lengths up to feature-length books (~1.5 hours output), measuring processing time and realtime factor. Tests were run on: 
- Windows 11 Home w/ WSL2 
- NVIDIA 4060Ti 16gb GPU @ CUDA 12.1
- 11th Gen i7-11700 @ 2.5GHz
- 64gb RAM
- WAV native output
- H.G. Wells - The Time Machine (full text)

<p align="center">
  <img src="assets/gpu_processing_time.png" width="45%" alt="Processing Time" style="border: 2px solid #333; padding: 10px; margin-right: 1%;">
  <img src="assets/gpu_realtime_factor.png" width="45%" alt="Realtime Factor" style="border: 2px solid #333; padding: 10px;">
</p>

Key Performance Metrics:
- Realtime Speed: Ranges between 35x-100x (generation time to output audio length)
- Average Processing Rate: 137.67 tokens/second (cl100k_base)
</details>
<details>
<summary>GPU Vs. CPU</summary>

```bash
# GPU: Requires NVIDIA GPU with CUDA 12.1 support (~35x-100x realtime speed)
docker compose up --build

# CPU: ONNX optimized inference (~5x+ realtime speed on M3 Pro)
docker compose -f docker-compose.cpu.yml up --build
```
*Note: Overall speed may have reduced somewhat with the structural changes to accomodate streaming. Looking into it* 
</details>

<details>
<summary>Natural Boundary Detection</summary>

- Automatically splits and stitches at sentence boundaries 
- Helps to reduce artifacts and allow long form processing as the base model is only currently configured for approximately 30s output 
</details>

<details>
<summary>Phoneme & Token Routes</summary>

Convert text to phonemes and/or generate audio directly from phonemes:
```python
import requests

def get_phonemes(text: str, language: str = "a"):
    """Get phonemes and tokens for input text"""
    response = requests.post(
        "http://localhost:8880/dev/phonemize",
        json={"text": text, "language": language}  # "a" for American English
    )
    response.raise_for_status()
    result = response.json()
    return result["phonemes"], result["tokens"]

def generate_audio_from_phonemes(phonemes: str, voice: str = "af_bella"):
    """Generate audio from phonemes"""
    response = requests.post(
        "http://localhost:8880/dev/generate_from_phonemes",
        json={"phonemes": phonemes, "voice": voice},
        headers={"Accept": "audio/wav"}
    )
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return None
    return response.content

# Example usage
text = "Hello world!"
try:
    # Convert text to phonemes
    phonemes, tokens = get_phonemes(text)
    print(f"Phonemes: {phonemes}")  # e.g. ðɪs ɪz ˈoʊnli ɐ tˈɛst
    print(f"Tokens: {tokens}")      # Token IDs including start/end tokens

    # Generate and save audio
    if audio_bytes := generate_audio_from_phonemes(phonemes):
        with open("speech.wav", "wb") as f:
            f.write(audio_bytes)
        print(f"Generated {len(audio_bytes)} bytes of audio")
except Exception as e:
    print(f"Error: {e}")
```

See `examples/phoneme_examples/generate_phonemes.py` for a sample script.
</details>

<details>
<summary>Debug Endpoints</summary>

Monitor system state and resource usage with these endpoints:

- `/debug/threads` - Get thread information and stack traces
- `/debug/storage` - Monitor temp file and output directory usage
- `/debug/system` - Get system information (CPU, memory, GPU)
- `/debug/session_pools` - View ONNX session and CUDA stream status

Useful for debugging resource exhaustion or performance issues.
</details>

## Known Issues

<details>
<summary>Versioning & Development</summary>

I'm doing what I can to keep things stable, but we are on an early and rapid set of build cycles here.
If you run into trouble, you may have to roll back a version on the release tags if something comes up, or build up from source and/or troubleshoot + submit a PR. Will leave the branch up here for the last known stable points:

`v0.0.5post1`

Free and open source is a community effort, and I love working on this project, though there's only really so many hours in a day. If you'd like to support the work, feel free to open a PR, buy me a coffee, or report any bugs/features/etc you find during use.

  <a href="https://www.buymeacoffee.com/remsky" target="_blank">
    <img 
      src="https://cdn.buymeacoffee.com/buttons/v2/default-violet.png" 
      alt="Buy Me A Coffee" 
      style="height: 30px !important;width: 110px !important;"
    >
  </a>

  
</details>

<details>
<summary>Linux GPU Permissions</summary>

Some Linux users may encounter GPU permission issues when running as non-root. 
Can't guarantee anything, but here are some common solutions, consider your security requirements carefully

### Option 1: Container Groups (Likely the best option)
```yaml
services:
  kokoro-tts:
    # ... existing config ...
    group_add:
      - "video"
      - "render"
```

### Option 2: Host System Groups
```yaml
services:
  kokoro-tts:
    # ... existing config ...
    user: "${UID}:${GID}"
    group_add:
      - "video"
```
Note: May require adding host user to groups: `sudo usermod -aG docker,video $USER` and system restart.

### Option 3: Device Permissions (Use with caution)
```yaml
services:
  kokoro-tts:
    # ... existing config ...
    devices:
      - /dev/nvidia0:/dev/nvidia0
      - /dev/nvidiactl:/dev/nvidiactl
      - /dev/nvidia-uvm:/dev/nvidia-uvm
```
⚠️ Warning: Reduces system security. Use only in development environments.

Prerequisites: NVIDIA GPU, drivers, and container toolkit must be properly configured.

Visit [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for more detailed information

</details>

## Model and License

<details open>
<summary>Model</summary>

This API uses the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model from HuggingFace. 

Visit the model page for more details about training, architecture, and capabilities. I have no affiliation with any of their work, and produced this wrapper for ease of use and personal projects.
</details>
<details>
<summary>License</summary>
This project is licensed under the Apache License 2.0 - see below for details:

- The Kokoro model weights are licensed under Apache 2.0 (see [model page](https://huggingface.co/hexgrad/Kokoro-82M))
- The FastAPI wrapper code in this repository is licensed under Apache 2.0 to match
- The inference code adapted from StyleTTS2 is MIT licensed

The full Apache 2.0 license text can be found at: https://www.apache.org/licenses/LICENSE-2.0
</details>
