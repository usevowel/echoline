import base64
from contextlib import suppress
import logging

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)

from echoline.dependencies import ConfigDependency, ExecutorRegistryDependency
from echoline.realtime.utils import verify_websocket_api_key
from echoline.schemas.vad_stream import VADStreamAudio, VADStreamOptions
from echoline.services.vad_stream_manager import VADStreamManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice-activity-detection"])

_manager_state: dict[str, VADStreamManager] = {}


def get_vad_stream_manager(executor_registry: ExecutorRegistryDependency) -> VADStreamManager:
    if "manager" not in _manager_state:
        _manager_state["manager"] = VADStreamManager(executor_registry.vad.model_manager)
    return _manager_state["manager"]


@router.websocket("/v1/vad/stream")
async def vad_stream(
    ws: WebSocket,
    session_id: str,
    config: ConfigDependency,
    executor_registry: ExecutorRegistryDependency,
) -> None:
    manager = get_vad_stream_manager(executor_registry)

    try:
        await verify_websocket_api_key(ws, config)
    except WebSocketException:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return

    await ws.accept()
    logger.info(f"Accepted VAD stream WebSocket connection for session: {session_id}")

    session = manager.get_or_create_session(session_id)

    try:
        while True:
            try:
                raw_message = await ws.receive_json()
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session: {session_id}")
                break

            message_type = raw_message.get("type")

            if message_type == "audio":
                try:
                    audio_data = VADStreamAudio(
                        session_id=session_id,
                        audio=raw_message.get("audio", ""),
                        timestamp_ms=raw_message.get("timestamp_ms", 0),
                        options=raw_message.get("options"),
                        reset_state=raw_message.get("reset_state", False),
                    )
                except Exception:
                    logger.exception(f"Invalid audio message for session: {session_id}")
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "Invalid audio message format",
                        }
                    )
                    continue

                if audio_data.reset_state:
                    session.reset()

                if audio_data.options is not None:
                    session.update_config(audio_data.options)

                try:
                    audio_bytes = base64.b64decode(audio_data.audio)
                except Exception:
                    logger.exception(f"Invalid base64 audio for session: {session_id}")
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "Invalid audio format",
                        }
                    )
                    continue

                if len(audio_bytes) == 0:
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "Empty audio data",
                        }
                    )
                    continue

                if len(audio_bytes) % 2 != 0:
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "Invalid PCM16 audio: odd byte count",
                        }
                    )
                    continue

                event = manager.process_chunk(
                    session_id=session_id,
                    audio_bytes=audio_bytes,
                    timestamp_ms=audio_data.timestamp_ms,
                )

                if event is not None:
                    await ws.send_json(event.model_dump())

            elif message_type == "reset":
                session.reset()
                logger.debug(f"Reset VAD state for session: {session_id}")

            elif message_type == "config":
                options_raw = raw_message.get("options", {})
                try:
                    options = VADStreamOptions(**options_raw)
                    session.update_config(options)
                    logger.debug(f"Updated config for session: {session_id}")
                except Exception:
                    logger.exception(f"Invalid config for session: {session_id}")
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "Invalid config options",
                        }
                    )

            elif message_type == "ping":
                await ws.send_json({"type": "pong"})

            else:
                logger.warning(f"Unknown message type '{message_type}' for session: {session_id}")
                await ws.send_json(
                    {
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                    }
                )

    except Exception:
        logger.exception(f"Error in VAD stream WebSocket for session: {session_id}")
        with suppress(Exception):
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        manager.remove_session(session_id)
        logger.info(f"VAD stream session cleaned up: {session_id}")
