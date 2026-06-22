# LAAS Realtime Compatibility Design

LAAS has two realtime voice WebSocket surfaces:

- `WS /v1/local/voice/sessions/{session_id}/realtime`
- `WS /v1/realtime/sessions/{session_id}`

The second route is an OpenAI-shaped compatibility wrapper over the same local
runtime. LAAS is not a full hosted OpenAI Realtime API implementation yet. It is
a compatibility-oriented bridge over the existing local stack:

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

For OpenAI-shaped session and response objects, create the session through:

```http
POST /v1/realtime/sessions
```

Then connect:

```text
WS /v1/realtime/sessions/{session_id}
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
| `conversation.item.create` | Supported text subset | Accepts `message` items with `system`, `user`, or `assistant` roles and text content parts. Replies with `conversation.item.created`. |
| `conversation.item.retrieve` | Supported text subset | Returns a stored text conversation item by `item_id`. |
| `conversation.item.delete` | Supported text subset | Removes a stored text conversation item from the visible item list and backend chat history. |
| `conversation.item.truncate` | Supported text subset | Truncates a stored text content part by `text_end_index`, replaces it with `text`, or acknowledges `audio_end_ms` as a no-op for text items. |
| `input_audio_buffer.append` | Supported | Appends base64 audio bytes to the current buffer. Replies with `input_audio_buffer.appended`. |
| `input_audio_buffer.clear` | Supported | Clears the current audio buffer. Replies with `input_audio_buffer.cleared`. |
| `input_audio_buffer.commit` | Supported | Runs one full local voice turn from buffered audio. The local route replies with `response.completed`; the OpenAI-shaped route emits lifecycle, text, and audio events before `response.completed`. |
| `response.create` | Supported as alias | Runs one full local voice turn from buffered audio, or from accumulated text `conversation.item.create` messages when no audio is buffered. The local route replies with `response.completed`; the OpenAI-shaped route emits lifecycle, text, and audio events before `response.completed`. |
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

## Text Conversation Items

The OpenAI-shaped route accepts text conversation items:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "message",
    "role": "user",
    "content": [
      {"type": "input_text", "text": "Answer this without an audio input."}
    ]
  }
}
```

LAAS stores the text in the voice session's chat history and replies with
`conversation.item.created`. A later `response.create` can run from that stored
text even when the audio buffer is empty. In that case, Whisper is not invoked,
but Kokoro still synthesizes the assistant text into output audio.

Supported content part types are `input_text`, `text`, and `output_text`.
`input_audio` inside `conversation.item.create` is rejected with an explicit
error; send audio through `input_audio_buffer.append` instead.

Stored text items can be read, removed, or truncated:

```json
{"type": "conversation.item.retrieve", "item_id": "item_..."}
{"type": "conversation.item.delete", "item_id": "item_..."}
{"type": "conversation.item.truncate", "item_id": "item_...", "content_index": 0, "text_end_index": 12}
```

Deleting an item removes it from the backend chat history used by later
`response.create` calls. Truncating a text item updates the backend chat history
to the truncated text.

## Session Configuration

`POST /v1/realtime/sessions` and `session.update` accept the practical local
subset of OpenAI-shaped session fields:

- `modalities`: `["audio", "text"]`, `["audio"]`, or `["text"]`
- `input_audio_format`: `pcm`, `wav`, `mp3`, `flac`, `opus`, or `aac`
- `output_audio_format` / `response_format`: `pcm`, `wav`, `mp3`, `flac`,
  `opus`, or `aac`
- `turn_detection`: stored and returned for client compatibility, but local
  server-side VAD is not implemented yet

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

The OpenAI-shaped route sends these events for a completed local turn:

```text
response.created
response.output_item.added
response.output_text.delta
response.output_text.done
response.audio.delta
response.audio.done
response.output_item.done
response.completed
```

`response.audio.delta` contains base64-encoded chunks of the final encoded
audio. With the current Kokoro backend, LAAS emits these chunks after TTS
returns a whole buffer. The event shape is streaming-compatible, but it is not
yet true low-latency chunk synthesis.

The final `response.completed` event contains a `realtime.response` wrapper and
keeps the full LAAS turn payload in `laas_turn`:

```json
{
  "type": "response.completed",
  "response": {
    "id": "resp_...",
    "object": "realtime.response",
    "status": "completed",
    "output": [
      {
        "type": "message",
        "role": "assistant",
        "content": [
          {"type": "output_text", "text": "..."},
          {
            "type": "output_audio",
            "audio": "<base64 audio bytes>",
            "format": "pcm",
            "media_type": "audio/pcm",
            "sample_rate": 24000
          }
        ]
      }
    ]
  },
  "laas_turn": {
    "object": "local.voice.turn",
    "transcript": {"text": "..."},
    "response": {"text": "..."},
    "audio": {"data": "<base64 audio bytes>"}
  }
}
```

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
- Server-side VAD
- Native chunk-by-chunk TTS generation before the local TTS backend returns
- Native LLM audio input/output
- Tool calls over Realtime
- Model-side interruption or cancellation

## Remaining Follow-Up

Issue #18 tracks the design and issue #19 tracks the first OpenAI-shaped
session wrapper. Remaining compatibility work should cover:

- Expand binary/audio conversation item support beyond text item controls.
- Add SDK/client smoke coverage for the supported subset.
- Add audio delta streaming when the local TTS backend can stream chunks.
- Add server-side VAD when a local VAD dependency is selected.
- Keep `/v1/local/voice/sessions/{session_id}/realtime` as the stable local
  transport.
