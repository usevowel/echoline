!!! note

## Intro

Unlike other API features. The VAD API isn't OpenAI compatible, as OpenAI doesn't provide a VAD API. Therefore, you cannot use OpenAI SDKs to access this API; You'll need to use an HTTP client like `httpx`(Python), `requests`(Python), `reqwest`(Rust), etc.

There's only 1 supported model for VAD, which is `silero_vad_v5`. This model is packaged in one of the dependencies, so you don't need to download it separately. Because of this, you won't see it when querying local models or listing models from model registry.

Refer to the [../api.md] for additional details such as supported request parameters and response format.

## Batch API

Use the batch endpoint for analyzing complete audio files.

```sh
export ECHOLINE_BASE_URL="http://localhost:8000"


curl "$ECHOLINE_BASE_URL/v1/audio/speech/timestamps" -F "file=@audio.wav"
# [{"start":64,"end":1323}]


curl "$ECHOLINE_BASE_URL/v1/audio/speech/timestamps" -F "file=@audio.wav"  -F "max_speech_duration_s=0.2"
# [{"start":64,"end":256},{"start":288,"end":480},{"start":512,"end":704},{"start":800,"end":992},{"start":1024,"end":1216}]

curl "$ECHOLINE_BASE_URL/v1/audio/speech/timestamps" -F "file=@audio.wav"  -F "max_speech_duration_s=0.2" -F "threshold=0.99"
# [{"start":96,"end":288},{"start":320,"end":512},{"start":544,"end":736},{"start":832,"end":1024},{"start":1056,"end":1248}]
```

## Streaming API (WebSocket)

Use the streaming endpoint for real-time voice activity detection on audio streams. This is useful for clients that need to detect speech boundaries as audio arrives (e.g., voice chat applications, live transcription pipelines).

### Endpoint

```
WS /v1/vad/stream?session_id=<uuid>
```

### Authentication

If an API key is configured, provide it via:

- Query parameter: `?api_key=<key>`
- Authorization header: `Authorization: Bearer <key>`
- X-API-Key header: `X-API-Key: <key>`

### Connection

```python
import websockets
import json
import base64

async def test_vad_stream():
    uri = "ws://localhost:8000/v1/vad/stream?session_id=test-123"
    async with websockets.connect(uri) as ws:
        # Send audio chunk
        audio = get_audio_chunk()  # PCM16 16kHz mono bytes
        await ws.send(json.dumps({
            "type": "audio",
            "audio": base64.b64encode(audio).decode(),
            "timestamp_ms": 1000
        }))

        # Receive VAD events
        async for msg in ws:
            event = json.loads(msg)
            print(f"VAD Event: {event}")
```

### Client-to-Server Messages

| Type | Fields | Description |
|------|--------|-------------|
| `audio` | `audio` (base64 string), `timestamp_ms` (int), `options` (optional object), `reset_state` (optional bool) | Send an audio chunk for VAD processing |
| `reset` | — | Reset the VAD state and audio buffer |
| `config` | `options` (object) | Update VAD configuration for subsequent chunks |
| `ping` | — | Keepalive ping |

### Server-to-Client Messages

| Type | Fields | Description |
|------|--------|-------------|
| `speech_start` | `timestamp_ms`, `probability` | Speech activity detected |
| `speech_end` | `timestamp_ms`, `probability`, `state` | Speech activity ended. `state` contains `audio_start_ms` and `audio_end_ms` |
| `error` | `message` | Error processing audio or invalid request |
| `pong` | — | Response to ping |

### Audio Format

| Property | Value |
|----------|-------|
| Encoding | PCM16 (16-bit signed little-endian) |
| Sample Rate | 16kHz |
| Channels | Mono |
| Transport | Base64-encoded in JSON |

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `threshold` | 0.5 | Speech probability threshold (0.0 - 1.0) |
| `neg_threshold` | null | Silence threshold. Defaults to `max(threshold - 0.15, 0.01)` if null |
| `min_silence_duration_ms` | 550 | Minimum silence duration before triggering speech_end |
| `speech_pad_ms` | 0 | Padding added to speech segments |
| `sample_rate` | 16000 | Audio sample rate (must be 16000 for Silero VAD) |

### Session Management

- Sessions are identified by `session_id` (client-provided string)
- Sessions auto-expire after 5 minutes of inactivity
- Sessions are cleaned up on WebSocket disconnect
- Audio buffer is trimmed to 30 seconds maximum
