from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from echoline.executors.silero_vad_v5 import MODEL_ID, SileroVADModelManager, VadOptions
from echoline.schemas.vad_stream import VADStreamOptions

if TYPE_CHECKING:
        from echoline.schemas.vad_stream import VADStreamEvent

logger = logging.getLogger(__name__)

WINDOW_SIZE_SAMPLES = 512
MAX_BUFFER_DURATION_MS = 30000


class VADStreamSession:
    def __init__(
        self,
        session_id: str,
        model_manager: SileroVADModelManager,
        config: VADStreamOptions,
    ) -> None:
        self.session_id = session_id
        self.model_manager = model_manager
        self.config = config
        self.audio_buffer: list[float] = []
        self.is_speech_active: bool = False
        self.speech_start_ms: int | None = None
        self.last_speech_prob: float = 0.0
        self.silence_frame_count: int = 0
        self.last_activity: float = time.time()

    def reset(self) -> None:
        self.audio_buffer = []
        self.is_speech_active = False
        self.speech_start_ms = None
        self.last_speech_prob = 0.0
        self.silence_frame_count = 0
        self.last_activity = time.time()

    def update_config(self, config: VADStreamOptions) -> None:
        self.config = config
        self.last_activity = time.time()

    def get_vad_options(self) -> VadOptions:
        return VadOptions(
            threshold=self.config.threshold,
            neg_threshold=self.config.neg_threshold,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
        )

    def get_buffer_duration_ms(self) -> int:
        return len(self.audio_buffer) * 1000 // self.config.sample_rate


class VADStreamManager:
    def __init__(self, model_manager: SileroVADModelManager) -> None:
        self._sessions: dict[str, VADStreamSession] = {}
        self._lock = threading.RLock()
        self._model_manager = model_manager

    def get_or_create_session(self, session_id: str, config: VADStreamOptions | None = None) -> VADStreamSession:
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_activity = time.time()
                if config is not None:
                    session.update_config(config)
                return session

            effective_config = config if config is not None else VADStreamOptions()
            session = VADStreamSession(
                session_id=session_id,
                model_manager=self._model_manager,
                config=effective_config,
            )
            self._sessions[session_id] = session
            logger.info(f"Created VAD stream session: {session_id}")
            return session

    def remove_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Removed VAD stream session: {session_id}")

    def process_chunk(
        self,
        session_id: str,
        audio_bytes: bytes,
        timestamp_ms: int,
    ) -> VADStreamEvent | None:
    from echoline.schemas.vad_stream import VADStreamEvent

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning(f"Session not found: {session_id}")
                return None

            session.last_activity = time.time()

            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            session.audio_buffer.extend(audio_float32.tolist())

            if len(session.audio_buffer) < WINDOW_SIZE_SAMPLES:
                return None

            audio_array = np.array(session.audio_buffer, dtype=np.float32)
            vad_options = session.get_vad_options()

            try:
                with session.model_manager.load_model(MODEL_ID) as model:
                    padded_length = WINDOW_SIZE_SAMPLES - len(audio_array) % WINDOW_SIZE_SAMPLES
                    if padded_length != WINDOW_SIZE_SAMPLES:
                        padded_audio = np.pad(audio_array, (0, padded_length))
                    else:
                        padded_audio = audio_array

                    audio_batch = padded_audio.reshape(1, -1)
                    speech_probs = model(audio_batch).squeeze(0)

                    if len(speech_probs) == 0:
                        return None

                    latest_prob = float(speech_probs[-1])
                    session.last_speech_prob = latest_prob

                    current_time_ms = timestamp_ms

                    if latest_prob >= vad_options.threshold:
                        session.silence_frame_count = 0
                        if not session.is_speech_active:
                            session.is_speech_active = True
                            session.speech_start_ms = current_time_ms
                            logger.debug(
                                f"Session {session_id}: speech_start at {current_time_ms}ms (prob={latest_prob:.3f})"
                            )
                            return VADStreamEvent(
                                session_id=session_id,
                                type="speech_start",
                                timestamp_ms=current_time_ms,
                                probability=latest_prob,
                            )
                    elif session.is_speech_active:
                        session.silence_frame_count += 1
                        silence_duration_ms = (
                            session.silence_frame_count * WINDOW_SIZE_SAMPLES * 1000 // session.config.sample_rate
                        )

                        if silence_duration_ms >= vad_options.min_silence_duration_ms:
                            speech_end_ms = current_time_ms
                            duration_ms = speech_end_ms - (session.speech_start_ms or current_time_ms)
                            session.is_speech_active = False
                            session.silence_frame_count = 0
                            start_ms = session.speech_start_ms
                            session.speech_start_ms = None
                            logger.debug(
                                f"Session {session_id}: speech_end at {speech_end_ms}ms (duration={duration_ms}ms)"
                            )
                            return VADStreamEvent(
                                session_id=session_id,
                                type="speech_end",
                                timestamp_ms=speech_end_ms,
                                probability=latest_prob,
                                state={"audio_start_ms": start_ms, "audio_end_ms": speech_end_ms},
                            )

                    return None

            except Exception:
                logger.exception(f"Error processing VAD chunk for session {session_id}")
                return VADStreamEvent(
                    session_id=session_id,
                    type="error",
                    timestamp_ms=timestamp_ms,
                    message="VAD processing error",
                )

            finally:
                max_samples = session.config.sample_rate * MAX_BUFFER_DURATION_MS // 1000
                if len(session.audio_buffer) > max_samples:
                    session.audio_buffer = session.audio_buffer[-max_samples:]

    def cleanup_expired_sessions(self, timeout_seconds: int = 300) -> int:
        with self._lock:
            current_time = time.time()
            expired_sessions = [
                session_id
                for session_id, session in self._sessions.items()
                if current_time - session.last_activity > timeout_seconds
            ]

            for session_id in expired_sessions:
                del self._sessions[session_id]
                logger.info(f"Cleaned up expired VAD stream session: {session_id}")

            return len(expired_sessions)

    def clear_all_sessions(self) -> int:
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            return count


_manager_holder: dict[str, VADStreamManager] = {}


def set_vad_manager(manager: VADStreamManager) -> None:
    _manager_holder["instance"] = manager


def cleanup_all_sessions() -> int:
    manager = _manager_holder.get("instance")
    if manager is not None:
        return manager.clear_all_sessions()
    return 0
