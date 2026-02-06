# Echo-TTS API

FastAPI wrapper around the local Echo-TTS code at `~/dev/echo-tts`.

## Requirements

- Python 3.10+
- CUDA GPU recommended (same requirements as Echo-TTS)
- Local Echo-TTS checkout (default path: `~/dev/echo-tts`)

## Install

```bash
cd ~/dev/echo-tts-api
python -m venv .venv
source .venv/bin/activate
pip install -r ~/dev/echo-tts/requirements.txt
pip install -r requirements.txt
```

## Voice Cloning Source (Default)

This API uses voice reference files from `./voices` by default.
If you do not upload `speaker_audio` and do not pass a specific `voice`, it will use the **first audio file** in `./voices` (sorted by filename).

Supported voice file extensions: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.opus`

## Run

```bash
export ECHO_TTS_ROOT=~/dev/echo-tts
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Model weights are loaded during startup.

## Endpoints

### Native

- `GET /healthz`
- `GET /v1/voices`
- `POST /v1/generate` returns `audio/wav`
- `POST /v1/generate/json` returns base64 audio + metadata

### OpenAI-compatible

- `POST /v1/audio/speech`
  - Accepts OpenAI-like JSON fields (`model`, `input`, `voice`, `response_format`, `speed`)
  - `response_format` supported: `mp3`, `wav`, `flac`, `pcm`

### ElevenLabs-compatible

- `POST /v1/text-to-speech/{voice_id}`
- `POST /v1/text-to-speech/{voice_id}/stream`
  - Accepts ElevenLabs-like JSON fields (`text`, `model_id`, `output_format`, `seed`, `voice_settings`)
  - `output_format` supported: `mp3_...`, `ogg_...`, `pcm_...`, `mp3`, `wav`, `flac`, `ogg`

## Examples

### Native WAV generation

```bash
curl -X POST http://localhost:8000/v1/generate \
  -F 'text_prompt=[S1] Hello from Echo-TTS API.' \
  -F 'num_steps=40' \
  -F 'rng_seed=0' \
  -o output.wav
```

### Native generation with explicit voice file

```bash
curl -X POST http://localhost:8000/v1/generate \
  -F 'text_prompt=[S1] This uses the voice file named scarlett.wav.' \
  -F 'voice=scarlett' \
  -o output.wav
```

### OpenAI-compatible

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-mini-tts",
    "input": "Hello from OpenAI-compatible Echo TTS",
    "voice": "scarlett",
    "response_format": "mp3"
  }' \
  -o speech.mp3
```

### ElevenLabs-compatible

```bash
curl -X POST http://localhost:8000/v1/text-to-speech/scarlett \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hello from ElevenLabs-compatible Echo TTS",
    "output_format": "mp3_44100_128"
  }' \
  -o speech.mp3
```

Use `voice_id=default` (or `first`) to force the first file in `./voices`.

## Useful environment variables

- `ECHO_TTS_ROOT` (default: `~/dev/echo-tts`)
- `ECHO_TTS_DEVICE` (default: `cuda` if available, otherwise `cpu`)
- `ECHO_TTS_MODEL_DTYPE` (`bfloat16`, `float16`, `float32`)
- `ECHO_TTS_FISH_AE_DTYPE` (`bfloat16`, `float16`, `float32`)
- `ECHO_TTS_USE_COMPILE` (`true` or `false`)
- `ECHO_TTS_VOICES_DIR` (default: `./voices`)
- `HF_TOKEN` (optional; needed for private/gated repos)
