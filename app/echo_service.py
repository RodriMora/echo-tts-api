from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torchaudio

from .config import Settings
from .schemas import GenerationRequest


SUPPORTED_VOICE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
MEDIA_TYPE_BY_FORMAT = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "pcm": "audio/pcm",
}


@dataclass
class GenerationResult:
    audio_bytes: bytes
    normalized_text: str
    sample_rate: int
    generation_seconds: float
    audio_format: str
    media_type: str
    voice_name: str | None


class EchoTTSService:
    sample_rate = 44_100

    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
        self.settings.voices_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._inference = self._import_inference_module()
        self._load_model_components()

    def _import_inference_module(self):
        if not self.settings.echo_tts_root.exists():
            raise FileNotFoundError(
                f"Echo-TTS repo not found at: {self.settings.echo_tts_root}"
            )

        root = str(self.settings.echo_tts_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        return importlib.import_module("inference")

    def _load_model_components(self) -> None:
        inference = self._inference

        self.model = inference.load_model_from_hf(
            repo_id=self.settings.model_repo_id,
            device=self.settings.device,
            dtype=self.settings.model_dtype,
            token=self.settings.hf_token,
            compile=False,
            delete_blockwise_modules=self.settings.delete_blockwise_modules,
        )
        self.fish_ae = inference.load_fish_ae_from_hf(
            repo_id=self.settings.fish_ae_repo_id,
            device=self.settings.device,
            dtype=self.settings.fish_ae_dtype,
            token=self.settings.hf_token,
            compile=False,
        )
        self.pca_state = inference.load_pca_state_from_hf(
            repo_id=self.settings.pca_repo_id,
            device=self.settings.device,
            filename=self.settings.pca_filename,
            token=self.settings.hf_token,
        )

        if self.settings.use_compile:
            self.model = inference.compile_model(self.model)
            self.fish_ae = inference.compile_fish_ae(self.fish_ae)

    def status(self) -> dict[str, str | bool | int | None]:
        default_voice = self._default_voice_file()
        return {
            "device": self.settings.device,
            "use_compile": self.settings.use_compile,
            "echo_tts_root": str(self.settings.echo_tts_root),
            "voices_dir": str(self.settings.voices_dir),
            "voices_count": len(self._list_voice_files()),
            "default_voice": default_voice.name if default_voice is not None else None,
        }

    def list_voices(self) -> list[dict[str, str | bool]]:
        default_voice = self._default_voice_file()
        default_name = default_voice.name if default_voice is not None else None

        voices: list[dict[str, str | bool]] = []
        for voice_path in self._list_voice_files():
            voices.append(
                {
                    "voice_id": voice_path.stem,
                    "name": voice_path.stem,
                    "filename": voice_path.name,
                    "is_default": voice_path.name == default_name,
                }
            )
        return voices

    def resolve_voice_file(self, voice_name: str | None) -> Path | None:
        available = self._list_voice_files()
        if not available:
            return None

        if voice_name is None or not voice_name.strip():
            return available[0]

        requested = Path(voice_name.strip()).name
        requested_lower = requested.lower()

        if requested_lower in {"default", "first"}:
            return available[0]

        for voice_path in available:
            if voice_path.name.lower() == requested_lower:
                return voice_path

        for voice_path in available:
            if voice_path.stem.lower() == requested_lower:
                return voice_path

        return None

    def generate(
        self,
        request: GenerationRequest,
        speaker_audio_bytes: bytes | None,
        speaker_filename: str | None,
        voice_name: str | None = None,
        output_format: str = "wav",
    ) -> GenerationResult:
        speaker_path: Path | None = None
        speaker_audio: torch.Tensor | None = None
        resolved_voice_name: str | None = None

        if speaker_audio_bytes:
            suffix = Path(speaker_filename or "speaker.wav").suffix or ".wav"
            speaker_path = self._write_temp_file(speaker_audio_bytes, suffix=suffix)
            speaker_audio = self._inference.load_audio(str(speaker_path))
            resolved_voice_name = Path(speaker_filename or "upload.wav").name
        else:
            voice_path = self.resolve_voice_file(voice_name)
            if voice_name and voice_path is None:
                raise ValueError(
                    f"Voice '{voice_name}' not found in {self.settings.voices_dir}."
                )
            if voice_path is not None:
                speaker_audio = self._inference.load_audio(str(voice_path))
                resolved_voice_name = voice_path.name

        speaker_kv_enabled = bool(request.force_speaker and speaker_audio is not None)

        try:
            with self._lock:
                start_time = time.perf_counter()

                sample_fn = partial(
                    self._inference.sample_euler_cfg_independent_guidances,
                    num_steps=request.num_steps,
                    cfg_scale_text=request.cfg_scale_text,
                    cfg_scale_speaker=(
                        request.cfg_scale_speaker if speaker_audio is not None else 0.0
                    ),
                    cfg_min_t=request.cfg_min_t,
                    cfg_max_t=request.cfg_max_t,
                    truncation_factor=request.truncation_factor,
                    rescale_k=request.rescale_k,
                    rescale_sigma=request.rescale_sigma,
                    speaker_kv_scale=(
                        request.speaker_kv_scale if speaker_kv_enabled else None
                    ),
                    speaker_kv_max_layers=(
                        request.speaker_kv_max_layers if speaker_kv_enabled else None
                    ),
                    speaker_kv_min_t=(
                        request.speaker_kv_min_t if speaker_kv_enabled else None
                    ),
                    sequence_length=request.sequence_length,
                )

                audio_out, normalized_text = self._inference.sample_pipeline(
                    model=self.model,
                    fish_ae=self.fish_ae,
                    pca_state=self.pca_state,
                    sample_fn=sample_fn,
                    text_prompt=request.text_prompt,
                    speaker_audio=speaker_audio,
                    rng_seed=request.rng_seed,
                    pad_to_max_text_length=request.pad_to_max_text_length,
                    pad_to_max_speaker_latent_length=request.pad_to_max_speaker_latent_length,
                    normalize_text=request.normalize_text,
                )

                generation_seconds = time.perf_counter() - start_time
                audio_bytes, normalized_format, media_type = self._save_tensor_to_audio_bytes(
                    audio_out[0].cpu(),
                    output_format=output_format,
                )

            return GenerationResult(
                audio_bytes=audio_bytes,
                normalized_text=normalized_text,
                sample_rate=self.sample_rate,
                generation_seconds=generation_seconds,
                audio_format=normalized_format,
                media_type=media_type,
                voice_name=resolved_voice_name,
            )
        finally:
            if speaker_path is not None:
                speaker_path.unlink(missing_ok=True)

    def _save_tensor_to_audio_bytes(
        self,
        audio_tensor: torch.Tensor,
        output_format: str,
    ) -> tuple[bytes, str, str]:
        normalized_format = self._normalize_output_format(output_format)

        if normalized_format == "pcm":
            mono = audio_tensor
            if mono.ndim == 2 and mono.shape[0] == 1:
                mono = mono[0]
            elif mono.ndim == 2:
                mono = mono.mean(dim=0)
            mono = mono.clamp(-1.0, 1.0)
            pcm16 = (mono * 32767.0).to(torch.int16).contiguous().cpu().numpy().tobytes()
            return pcm16, normalized_format, MEDIA_TYPE_BY_FORMAT[normalized_format]

        if normalized_format == "ogg":
            return (
                self._save_ogg_opus_bytes(audio_tensor),
                normalized_format,
                MEDIA_TYPE_BY_FORMAT[normalized_format],
            )

        output_path = self._write_temp_file(b"", suffix=f".{normalized_format}")
        try:
            if normalized_format == "mp3":
                torchaudio.save(
                    str(output_path),
                    audio_tensor,
                    self.sample_rate,
                    format="mp3",
                    encoding="mp3",
                    bits_per_sample=None,
                )
            else:
                torchaudio.save(
                    str(output_path),
                    audio_tensor,
                    self.sample_rate,
                    format=normalized_format,
                )

            return (
                output_path.read_bytes(),
                normalized_format,
                MEDIA_TYPE_BY_FORMAT[normalized_format],
            )
        finally:
            output_path.unlink(missing_ok=True)

    def _save_ogg_opus_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        wav_path = self._write_temp_file(b"", suffix=".wav")
        ogg_path = self._write_temp_file(b"", suffix=".ogg")
        try:
            # Encode to PCM WAV first, then transcode with ffmpeg/libopus for Telegram compatibility.
            torchaudio.save(str(wav_path), audio_tensor, self.sample_rate, format="wav")

            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-ac",
                "1",
                "-ar",
                "48000",
                "-c:a",
                "libopus",
                "-b:a",
                "48k",
                "-vbr",
                "on",
                "-application",
                "voip",
                str(ogg_path),
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise ValueError(
                    "ffmpeg is required for OGG Opus output but was not found in PATH."
                ) from exc

            if proc.returncode != 0:
                detail = proc.stderr.strip() or proc.stdout.strip() or "unknown ffmpeg error"
                raise ValueError(f"Failed to encode OGG Opus audio: {detail}")

            return ogg_path.read_bytes()
        finally:
            wav_path.unlink(missing_ok=True)
            ogg_path.unlink(missing_ok=True)

    def _normalize_output_format(self, output_format: str) -> str:
        fmt = (output_format or "wav").strip().lower()
        if fmt in {"wav", "mp3", "flac", "ogg"}:
            return fmt
        if fmt in {"pcm", "pcm16"}:
            return "pcm"
        raise ValueError("Unsupported output format. Use one of: wav, mp3, flac, ogg, pcm")

    def _list_voice_files(self) -> list[Path]:
        paths: list[Path] = []
        for path in self.settings.voices_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() in SUPPORTED_VOICE_EXTENSIONS:
                paths.append(path)
        return sorted(paths, key=lambda p: p.name.lower())

    def _default_voice_file(self) -> Path | None:
        voices = self._list_voice_files()
        return voices[0] if voices else None

    def _write_temp_file(self, payload: bytes, suffix: str) -> Path:
        fd, path = tempfile.mkstemp(dir=self.settings.temp_dir, suffix=suffix)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        return Path(path)
