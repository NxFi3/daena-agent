# Local Voice Mode

Voice is an optional Studio client for the existing Daena agent. It does not
replace the configured language model, tool permissions, conversation or browser/
desktop engines. Ordinary CLI/Studio use imports no speech model or audio device.
There are no paid API calls in this implementation.

## Install and start

```bash
conda activate AI
cd ~/Desktop/daena-latest
python -m pip install -e '.[desktop,voice]'
python -m src.voice doctor
python -m src.studio
```

Click **Voice** beside the composer. The dialog has Conversation, Models and Audio
controls, device selection, a microphone meter and measured recognition/first-PCM
latencies. Opening the dialog enumerates devices; recording starts only after
**Start Voice**. Closing Voice or Studio stops capture and queued playback.
Microphone permission and a working system audio service are required. On Linux,
Voice checks Qt's PipeWire configuration before enumeration; when no backend was
explicitly selected and PipeWire configuration is missing, it uses the available
PulseAudio backend. An explicit unusable PipeWire choice yields an error instead
of changing the user's selection. This preflight runs only when Voice is opened.

## One-time model setup

Choose **Download speech model**, **Download Persian voice**, and **Download
English voice**, or run explicit commands:

```bash
python -m src.voice download whisper-turbo
python -m src.voice download piper-fa
python -m src.voice download piper-en
```

These are public model downloads, not paid API use. Whisper turbo is approximately
1.6GB; voice models are additional downloads. The downloader resolves an immutable
repository revision, streams to temporary files, verifies available remote hashes
and sizes, records local SHA-256 manifests, and retains model cards/licenses.
Downloads are cancellable between network blocks; a stalled read may take up to
its network timeout to finish. Inference never downloads models implicitly.

Models live under the user's cache (`~/.cache/daena/voice` on typical Linux hosts;
see `python -m src.voice models` for the actual paths). Per-workspace preferences
are in `.daena/studio/voice.json`. No microphone recordings are written to disk by
Voice Mode. Recognized text becomes ordinary chat history when submitted.

If Hugging Face is unavailable or returns403 in your location, download the same
model files through an accessible connection and choose their local paths in
Models. A Whisper CTranslate2 directory needs `model.bin`, `config.json`, tokenizer
and vocabulary files. Each Piper voice needs the `.onnx` file and matching
`.onnx.json`. No private access token is required by this implementation.

## Persian and English quality

The default recognition model is multilingual **large-v3-turbo**, with CPU int8,
four CPU threads and beam search3. This keeps the GPU free for a large local
Ollama model. Choose **small** to reduce memory/load time, or explicitly choose a
supported CUDA/compute combination. Models are cached and warmed in background
workers. `auto` detects language; fixed `fa` or `en` can improve short utterances.

The included catalog uses Persian **Amir medium** and English **Lessac high**
Piper voices. Automatic TTS handles script changes with the corresponding voice;
pronunciation of names/code-switching depends on the voice. Quality is not
claimed equivalent to a cloud speech-to-speech model. The Piper software is
GPLv3; individual voice/data licenses vary. Consult the downloaded model cards
and licenses before redistributing voices or derivative products.

Audition a downloaded voice without opening the microphone:

```bash
python -m src.voice synthesize --language fa --text 'سلام، من داینا هستم. چطور می‌توانم کمک کنم؟' --output daena-fa.wav
python -m src.voice synthesize --language en --text 'Hello, I am Daena. How can I help?' --output daena-en.wav
```

The command refuses an existing output filename. Audition mixed-rate voices
separately; Studio resamples them automatically for the selected output device.

## Turn-taking and interruption

- Recognized speech appears in the dialog and composer. Review it and press
  **Send to Daena** or Enter in chat. **Auto-send** is explicit and only submits a
  complete utterance when the agent is idle and the composer has no other draft.
  Busy-agent speech remains for review rather than silently steering active work.
- Endpointing uses20ms frames, short pre-roll and configurable silence detection.
  WebRTC VAD is installed by the voice extra; an adaptive energy fallback is
  available. A duration cap marks a partial utterance for review and suppresses
  its continuing tail until actual silence. **Finish speaking** supplies an
  explicit endpoint when needed.
- **Interrupt** immediately clears buffered speech, invalidates late STT/TTS
  results and requests cancellation of the active conversational agent turn.
  Native inference itself may finish later; it is not forcibly killed, and its
  canceled audio/text cannot be submitted. Restart waits for old native work to
  finish so a second copy of the model is not loaded on top.
- During speaker playback, microphone PCM is gated before recognition; a short
  tail gate prevents echo retriggering. This is half-duplex turn-taking, not
  hardware acoustic echo cancellation or automatic speaker-safe full duplex.
  Use Interrupt to resume talking immediately. Manual mute also suspends capture.
- Only committed final responses are spoken by default. Optional progress speech
  uses committed assistant messages. Raw drafts, reasoning, tool receipts and
  discarded retries are never narrated. Markdown is prepared for speech; fenced
  code and URL targets are omitted. Read-aloud is bounded to8000 characters;
  the complete answer stays in chat.

## Latency and resource limits

Speech recognition and synthesis have independent workers and bounded queues.
Synthesis is sentence-based and PCM is delivered in short chunks with backpressure
and streaming HQ Soxr sample-rate conversion. Resampler tails are flushed to
preserve final phonemes. Microphone/playback format negotiation handles native
rates and channel counts rather than assuming every device accepts16kHz.

The metrics show utterance-end-to-transcription and queued-text-to-first-PCM
**measured on your machine**. They are not claims of audible device latency. End
silence, model warm-up, local language-model reasoning/generation and hardware
buffers add delay. Whisper is an utterance recognizer here, not native streaming
speech-to-speech. A slow35B reasoning model still determines the agent response
wait; lower its reasoning level in existing Studio controls if appropriate.

The feature was verified with scripted recognition/synthesis, synthetic PCM,
resampling tests, cancellation/backpressure tests and Qt offscreen interaction.
Speech packages/APIs were installed/inspected, but this build environment could not
fetch Hugging Face weights or audition a real microphone/speaker. Persian
naturalness and real end-to-end latency still require audition with actual models.

## Primary references

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper): local CTranslate2 transcription and VAD.
- [Whisper language support](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py): Persian `fa` and English `en`.
- [Piper Python API](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md): local chunked synthesis.
- [Published voice catalog](https://github.com/rhasspy/piper/blob/master/VOICES.md) and [voice licensing](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md).
