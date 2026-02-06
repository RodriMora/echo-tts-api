from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key not in _DTYPE_MAP:
        supported = ", ".join(sorted(_DTYPE_MAP.keys()))
        raise ValueError(f"Unsupported dtype '{name}'. Supported values: {supported}")
    return _DTYPE_MAP[key]


def _expand_path(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


@dataclass(frozen=True)
class Settings:
    echo_tts_root: Path
    temp_dir: Path
    voices_dir: Path
    device: str
    model_repo_id: str
    fish_ae_repo_id: str
    pca_repo_id: str
    pca_filename: str
    hf_token: str | None
    model_dtype: torch.dtype
    fish_ae_dtype: torch.dtype
    use_compile: bool
    delete_blockwise_modules: bool


def load_settings() -> Settings:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = os.getenv("ECHO_TTS_DEVICE", default_device)

    default_model_dtype = "bfloat16" if device.startswith("cuda") else "float32"

    return Settings(
        echo_tts_root=_expand_path(os.getenv("ECHO_TTS_ROOT", "~/dev/echo-tts")),
        temp_dir=_expand_path(os.getenv("ECHO_TTS_API_TEMP_DIR", "./tmp")),
        voices_dir=_expand_path(os.getenv("ECHO_TTS_VOICES_DIR", "./voices")),
        device=device,
        model_repo_id=os.getenv("ECHO_TTS_MODEL_REPO", "jordand/echo-tts-base"),
        fish_ae_repo_id=os.getenv("ECHO_TTS_FISH_AE_REPO", "jordand/fish-s1-dac-min"),
        pca_repo_id=os.getenv("ECHO_TTS_PCA_REPO", "jordand/echo-tts-base"),
        pca_filename=os.getenv("ECHO_TTS_PCA_FILENAME", "pca_state.safetensors"),
        hf_token=os.getenv("HF_TOKEN"),
        model_dtype=_to_dtype(os.getenv("ECHO_TTS_MODEL_DTYPE", default_model_dtype)),
        fish_ae_dtype=_to_dtype(os.getenv("ECHO_TTS_FISH_AE_DTYPE", "float32")),
        use_compile=_to_bool(os.getenv("ECHO_TTS_USE_COMPILE"), default=False),
        delete_blockwise_modules=_to_bool(
            os.getenv("ECHO_TTS_DELETE_BLOCKWISE_MODULES"), default=True
        ),
    )
