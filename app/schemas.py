from __future__ import annotations

from typing import Annotated

from fastapi import Form
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerationRequest(BaseModel):
    text_prompt: str = Field(min_length=1, max_length=4000)
    voice: str | None = None
    rng_seed: int = Field(default=0, ge=0)
    num_steps: int = Field(default=40, ge=1, le=80)
    cfg_scale_text: float = Field(default=3.0, ge=0.0)
    cfg_scale_speaker: float = Field(default=8.0, ge=0.0)
    cfg_min_t: float = Field(default=0.5, ge=0.0, le=1.0)
    cfg_max_t: float = Field(default=1.0, ge=0.0, le=1.0)
    truncation_factor: float | None = Field(default=0.8, gt=0.0)
    rescale_k: float | None = Field(default=None, gt=0.0)
    rescale_sigma: float | None = Field(default=3.0, gt=0.0)
    force_speaker: bool = False
    speaker_kv_scale: float = Field(default=1.5, gt=0.0)
    speaker_kv_max_layers: int | None = Field(default=None, ge=1)
    speaker_kv_min_t: float = Field(default=0.5, ge=0.0, le=1.0)
    sequence_length: int = Field(default=640, ge=1, le=640)
    pad_to_max_text_length: int | None = Field(default=None, ge=1, le=768)
    pad_to_max_speaker_latent_length: int | None = Field(default=None, ge=4, le=6400)
    normalize_text: bool = True

    @model_validator(mode="after")
    def validate_ranges(self) -> "GenerationRequest":
        if self.cfg_min_t > self.cfg_max_t:
            raise ValueError("cfg_min_t must be <= cfg_max_t")
        if self.force_speaker and self.speaker_kv_min_t > 1.0:
            raise ValueError("speaker_kv_min_t must be <= 1.0")
        if self.rescale_k is None:
            self.rescale_sigma = None
        if self.rescale_k is not None and self.rescale_sigma is None:
            raise ValueError("rescale_sigma is required when rescale_k is provided")
        return self

    @classmethod
    def as_form(
        cls,
        text_prompt: Annotated[str, Form(...)],
        voice: Annotated[str | None, Form()] = None,
        rng_seed: Annotated[int, Form()] = 0,
        num_steps: Annotated[int, Form()] = 40,
        cfg_scale_text: Annotated[float, Form()] = 3.0,
        cfg_scale_speaker: Annotated[float, Form()] = 8.0,
        cfg_min_t: Annotated[float, Form()] = 0.5,
        cfg_max_t: Annotated[float, Form()] = 1.0,
        truncation_factor: Annotated[float | None, Form()] = 0.8,
        rescale_k: Annotated[float | None, Form()] = None,
        rescale_sigma: Annotated[float | None, Form()] = 3.0,
        force_speaker: Annotated[bool, Form()] = False,
        speaker_kv_scale: Annotated[float, Form()] = 1.5,
        speaker_kv_max_layers: Annotated[int | None, Form()] = None,
        speaker_kv_min_t: Annotated[float, Form()] = 0.5,
        sequence_length: Annotated[int, Form()] = 640,
        pad_to_max_text_length: Annotated[int | None, Form()] = None,
        pad_to_max_speaker_latent_length: Annotated[int | None, Form()] = None,
        normalize_text: Annotated[bool, Form()] = True,
    ) -> "GenerationRequest":
        return cls(
            text_prompt=text_prompt,
            voice=voice,
            rng_seed=rng_seed,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            force_speaker=force_speaker,
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_max_layers=speaker_kv_max_layers,
            speaker_kv_min_t=speaker_kv_min_t,
            sequence_length=sequence_length,
            pad_to_max_text_length=pad_to_max_text_length,
            pad_to_max_speaker_latent_length=pad_to_max_speaker_latent_length,
            normalize_text=normalize_text,
        )


class OpenAIAudioSpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "echo-tts"
    input: str = Field(min_length=1, max_length=4000)
    voice: str | None = None
    response_format: str = "mp3"
    speed: float = Field(default=1.0, gt=0.0)
    instructions: str | None = None
    seed: int | None = Field(default=None, ge=0)


class ElevenLabsTTSRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1, max_length=4000)
    model_id: str | None = None
    output_format: str | None = "mp3_44100_128"
    seed: int | None = Field(default=None, ge=0)
    voice_settings: dict[str, float | bool | str | int] | None = None


class GenerationJSONResponse(BaseModel):
    normalized_text: str
    generation_seconds: float
    sample_rate: int
    audio_base64: str
