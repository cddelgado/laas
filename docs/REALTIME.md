# LAAS Realtime Compatibility Design

LAAS has a local realtime voice WebSocket at
`/v1/local/voice/sessions/{session_id}/realtime`. It is not a full OpenAI
Realtime API implementation yet. It is a compatibility-oriented bridge over the
existing local stack:

- Whisper.cpp-compatible STT for incoming audio
- Gemma text chat for the assistant response
- Kokoro TTS for outgoing audio

Native Gemma audio input/output is not assumed. Audio is transcribed before it
reaches the LLM, and response audio is synthesized after the LLM returns text.

## Session Lifecycle

Create a local voice session first:

```http
POST /v1/local/voice/sessions
```

Then connect:

```text
WS /v1/local/voice/sessions/{session_id}/realtime
```

The server immediately sends:

```json
{
  "type": "session.created",
  "session": {
    "id": "vs_...",
    "object": "local.voice.session",
    "status": "active"
  }
}
```

Close the session with:

```json
{"type":"session.close"}
```

or:

```json
{"type":"close"}
```

The server sends `session.closed`, removes the local session, and closes the
WebSocket.

## Supported Event Subset

| Event | Status | LAAS behavior |
| --- | --- | --- |
| `session.update` | Supported local compatibility subset | Updates `instructions`, `voice`, `response_format`, `language`, `prompt`, `temperature`, `speed`, and `lang`. Replies with `session.updated`. |
| `input_audio_buffer.append` | Supported | Appends base64 audio bytes to the current buffer. Replies with `input_audio_buffer.appended`. |
| `input_audio_buffer.clear` | Supported | Clears the current audio buffer. Replies with `input_audio_buffer.cleared`. |
| `input_audio_buffer.commit` | Supported | Runs one full local voice turn from buffered audio. Replies with `response.completed`. |
| `response.create` | Supported as alias | Runs one full local voice turn from buffered audio. Replies with `response.completed`. |
| `response.cancel` | Supported control event | Replies with `response.cancelled`. No in-flight model cancellation is attempted yet. |
| `voice.turn` | LAAS extension | One-shot event with inline base64 audio. Replies with `response.completed`. |
| `session.close` / `close` | Supported | Ends and removes the local session. |

## Audio Input

`input_audio_buffer.append` expects base64-encoded bytes:

```json
{
  "type": "input_audio_buffer.append",
  "audio": "<base64 audio bytes>"
}
```

After one or more appends, run the turn:

```json
{
  "type": "response.create",
  "filename": "question.wav"
}
```

`filename` is used only to pick a temporary file suffix for the transcription
backend. LAAS does not currently validate container/codec at the WebSocket
layer; the STT backend is responsible for rejecting unreadable audio.

## Output Event

Successful turns reply with:

```json
{
  "type": "response.completed",
  "session_id": "vs_...",
  "turn": {
    "object": "local.voice.turn",
    "transcript": {"text": "..."},
    "response": {"text": "..."},
    "audio": {
      "data": "<base64 audio bytes>",
      "format": "pcm",
      "media_type": "audio/pcm",
      "sample_rate": 24000
    }
  }
}
```

The output shape is local and intentionally mirrors the HTTP voice turn payload.
It is not yet OpenAI Realtime event parity.

## Error Events

Errors are sent as:

```json
{
  "type": "error",
  "error": {
    "message": "...",
    "code": "..."
  }
}
```

Current codes include:

- `not_found`
- `invalid_audio`
- `empty_audio`
- `voice_turn_failed`
- `unsupported_event`

## Unsupported OpenAI Realtime Areas

The following are intentionally out of scope for the current local bridge:

- WebRTC transport
- Hosted ephemeral token/session APIs
- OpenAI Realtime `/v1/realtime` endpoint parity
- Full `conversation.item.*` event graph
- Server-side VAD
- Audio delta streaming
- Native LLM audio input/output
- Tool calls over Realtime
- Model-side interruption or cancellation

## Proposed Implementation Follow-Up

Issue #18 tracks the design. The next implementation issue should cover:

- Add an OpenAI-shaped `/v1/realtime` or `/v1/realtime/sessions` endpoint.
- Translate OpenAI Realtime event names to the local voice session runtime.
- Return explicit unsupported-event errors for unimplemented OpenAI events.
- Add SDK/client smoke coverage for the supported subset.
- Keep `/v1/local/voice/sessions/{session_id}/realtime` as the stable local
  transport.
