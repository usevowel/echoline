from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VADStreamOptions(BaseModel):
    threshold: float = Field(default=0.5, ge=0, le=1)
    neg_threshold: float | None = Field(default=None, ge=0)
    min_silence_duration_ms: int = Field(default=550, ge=0)
    speech_pad_ms: int = Field(default=0, ge=0)
    sample_rate: int = Field(default=16000)


class VADStreamAudio(BaseModel):
    session_id: str
    audio: str  # base64-encoded PCM16
    timestamp_ms: int = Field(default=0, ge=0)
    options: VADStreamOptions | None = None
    reset_state: bool = False


class VADStreamEvent(BaseModel):
    session_id: str
    type: Literal["speech_start", "speech_end", "probability", "error"]
    timestamp_ms: int
    probability: float | None = None
    state: dict | None = None
    message: str | None = None
