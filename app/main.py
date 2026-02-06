from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from .config import load_settings
from .echo_service import EchoTTSService
from .schemas import (
    ElevenLabsTTSRequest,
    GenerationJSONResponse,
    GenerationRequest,
    OpenAIAudioSpeechRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    service = EchoTTSService(settings)
    app.state.echo_service = service
    yield


app = FastAPI(
    title="Echo-TTS API",
    description="HTTP API wrapper around local Echo-TTS inference.",
    version="0.2.0",
    lifespan=lifespan,
)


def get_service(request: Request) -> EchoTTSService:
    service = getattr(request.app.state, "echo_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model service is not ready")
    return service


def _normalize_openai_format(response_format: str | None) -> str:
    fmt = (response_format or "mp3").strip().lower()
    if fmt in {"mp3", "wav", "flac", "pcm", "pcm16"}:
        return fmt
    raise ValueError(
        "Unsupported OpenAI response_format. Supported: mp3, wav, flac, pcm"
    )


def _normalize_elevenlabs_format(output_format: str | None) -> str:
    if output_format is None:
        return "mp3"

    value = output_format.strip().lower()
    if value in {"mp3", "wav", "flac", "pcm", "pcm16"}:
        return value

    if value.startswith("mp3_"):
        return "mp3"
    if value.startswith("pcm_"):
        return "pcm"

    raise ValueError(
        "Unsupported ElevenLabs output_format. Examples: mp3_44100_128, pcm_44100, wav"
    )


def _run_generation(
    service: EchoTTSService,
    payload: GenerationRequest,
    speaker_bytes: bytes | None,
    speaker_filename: str | None,
    output_format: str,
    voice_name: str | None,
):
    try:
        return service.generate(
            request=payload,
            speaker_audio_bytes=speaker_bytes,
            speaker_filename=speaker_filename,
            output_format=output_format,
            voice_name=voice_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/healthz")
def healthz(service: Annotated[EchoTTSService, Depends(get_service)]):
    return {"status": "ok", **service.status()}


@app.get("/v1/voices")
def list_voices(service: Annotated[EchoTTSService, Depends(get_service)]):
    return {"voices": service.list_voices()}


@app.post("/v1/generate", response_class=Response)
async def generate_audio(
    payload: Annotated[GenerationRequest, Depends(GenerationRequest.as_form)],
    service: Annotated[EchoTTSService, Depends(get_service)],
    speaker_audio: UploadFile | None = File(default=None),
):
    speaker_bytes = await speaker_audio.read() if speaker_audio is not None else None
    result = _run_generation(
        service=service,
        payload=payload,
        speaker_bytes=speaker_bytes,
        speaker_filename=speaker_audio.filename if speaker_audio else None,
        output_format="wav",
        voice_name=payload.voice,
    )

    headers = {
        "Content-Disposition": 'inline; filename="echo_tts.wav"',
        "X-Generation-Seconds": f"{result.generation_seconds:.3f}",
        "X-Sample-Rate": str(result.sample_rate),
    }
    if result.voice_name:
        headers["X-Voice-Name"] = result.voice_name

    return Response(content=result.audio_bytes, media_type=result.media_type, headers=headers)


@app.post("/v1/generate/json", response_model=GenerationJSONResponse)
async def generate_audio_json(
    payload: Annotated[GenerationRequest, Depends(GenerationRequest.as_form)],
    service: Annotated[EchoTTSService, Depends(get_service)],
    speaker_audio: UploadFile | None = File(default=None),
):
    speaker_bytes = await speaker_audio.read() if speaker_audio is not None else None
    result = _run_generation(
        service=service,
        payload=payload,
        speaker_bytes=speaker_bytes,
        speaker_filename=speaker_audio.filename if speaker_audio else None,
        output_format="wav",
        voice_name=payload.voice,
    )

    return GenerationJSONResponse(
        normalized_text=result.normalized_text,
        generation_seconds=result.generation_seconds,
        sample_rate=result.sample_rate,
        audio_base64=base64.b64encode(result.audio_bytes).decode("ascii"),
    )


@app.post("/v1/audio/speech", response_class=Response)
async def openai_audio_speech(
    payload: OpenAIAudioSpeechRequest,
    service: Annotated[EchoTTSService, Depends(get_service)],
):
    try:
        output_format = _normalize_openai_format(payload.response_format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    generation_payload = GenerationRequest(
        text_prompt=payload.input,
        voice=payload.voice,
        rng_seed=payload.seed or 0,
    )

    result = _run_generation(
        service=service,
        payload=generation_payload,
        speaker_bytes=None,
        speaker_filename=None,
        output_format=output_format,
        voice_name=payload.voice,
    )

    extension = "pcm" if result.audio_format == "pcm" else result.audio_format
    filename = f"speech.{extension}"
    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "X-Generation-Seconds": f"{result.generation_seconds:.3f}",
        "X-Sample-Rate": str(result.sample_rate),
    }
    if result.voice_name:
        headers["X-Voice-Name"] = result.voice_name

    return Response(content=result.audio_bytes, media_type=result.media_type, headers=headers)


@app.post("/v1/text-to-speech/{voice_id}", response_class=Response)
async def elevenlabs_text_to_speech(
    voice_id: str,
    payload: ElevenLabsTTSRequest,
    service: Annotated[EchoTTSService, Depends(get_service)],
):
    voice_name = None if voice_id.strip().lower() in {"default", "first"} else voice_id
    try:
        output_format = _normalize_elevenlabs_format(payload.output_format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    generation_payload = GenerationRequest(
        text_prompt=payload.text,
        voice=voice_name,
        rng_seed=payload.seed or 0,
    )

    result = _run_generation(
        service=service,
        payload=generation_payload,
        speaker_bytes=None,
        speaker_filename=None,
        output_format=output_format,
        voice_name=voice_name,
    )

    extension = "pcm" if result.audio_format == "pcm" else result.audio_format
    filename = f"elevenlabs_tts.{extension}"
    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "X-Generation-Seconds": f"{result.generation_seconds:.3f}",
        "X-Sample-Rate": str(result.sample_rate),
    }
    if result.voice_name:
        headers["X-Voice-Name"] = result.voice_name

    return Response(content=result.audio_bytes, media_type=result.media_type, headers=headers)


@app.post("/v1/text-to-speech/{voice_id}/stream", response_class=Response)
async def elevenlabs_text_to_speech_stream(
    voice_id: str,
    payload: ElevenLabsTTSRequest,
    service: Annotated[EchoTTSService, Depends(get_service)],
):
    return await elevenlabs_text_to_speech(voice_id, payload, service)
