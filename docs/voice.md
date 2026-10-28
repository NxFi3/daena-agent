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
# Optional higher-quality English voice and raw-audio perception:
python -m pip install -e '.[voice-kokoro,voice-perception]'
python -m src.voice doctor
python -m src.studio
```

For interruption through speakers, Automatic mode needs a local echo-cancellation
backend. On Ubuntu/Debian the simplest installation is the system SpeexDSP library:

```bash
sudo apt install libspeexdsp1
```

Alternatively, install the optional WebRTC backend in the same Python environment:

```bash
python -m pip install -e '.[voice-duplex]'
```

WebRTC is tried first, then SpeexDSP. The WebRTC wrapper publishes Windows x64
wheels for supported Python versions; Linux may require a native source build
with a C++ compiler and the wrapper's SWIG/Meson build prerequisites. The system
SpeexDSP library avoids that Python extension build on Ubuntu. These are local
signal-processing libraries, with no API key, paid service or speech-model
download. If neither backend is available, Automatic mode keeps safe manual
turn-taking and shows that status instead of silently opening speaker echo to
recognition. Headphones mode works without either backend when wearing headphones.

Open **Settings → Voice** to choose output voices, microphone recognition,
devices and optional acoustic perception. Settings are created lazily and opening
them never starts recording or a speech model. Choose **Apply voice settings** to
save the selected configuration; changes to an active Voice session are validated
before it restarts. Voice preferences are separate
from the Agent profile and remain editable while the Agent works; runtime profile
controls stay locked until its turn finishes.

Click **Voice** beside the main chat composer to open the conversation screen.
Its central animated circle shows listening, thinking and speaking activity.
Opening this screen starts the selected Voice configuration when its local files
are ready. Microphone input requires microphone permission; text-only input does
not. **End Voice** or closing the conversation screen stops capture and queued
speech. The screen's **Voice settings** button opens the main Settings page
without ending the active Voice session.

Changing models or devices with **Apply voice settings** restarts Voice only,
without canceling the Agent's active turn. The current speech/input epoch is
replaced and late recognition from it is discarded; a pending committed Agent
reply is retained. Native inference may need to finish before restart can
complete. Model downloads continue when Voice ends; closing Studio cancels its
transfers while retaining resumable partial bytes.

**Hands-free** on the Voice screen sends complete recognized utterances without
pressing Send. It is on for new configurations; a previously saved off setting is
preserved. Toggle it off to review recognized words first. The same preference is
available under **Settings → Voice → Conversation** and can change while the Agent
works. A separate unsent chat draft is preserved and prevents automatic submission.

A working system audio service is required. On Linux, Voice checks Qt's PipeWire
configuration before device access; when no backend was explicitly selected and
PipeWire configuration is missing, it uses the available PulseAudio backend. An
explicit unusable PipeWire choice yields an error instead of changing the user's
selection. Ordinary Studio use does not run this audio preflight.

## Text only: use Voice without Whisper

In microphone mode, tap the central circle or choose **Type instead** to reveal the text composer
and switch away from microphone input. **Enter** or **Send** submits your message
to the existing Agent and its committed answer is spoken. **Shift+Enter** inserts
a newline. The main chat remains the complete conversation history; the Voice
screen shows the current interaction. Typed input stays usable while the Agent
works and follows the existing busy-turn handling instead of starting a second
Agent run.

You can also select **Settings → Voice → Conversation → Input → Text only · no
Whisper or microphone** and apply it as the preferred mode.
If no configured Whisper model is installed and no input mode was previously
selected, Voice offers this mode automatically. It needs only the selected local
output voice files; it never loads Whisper/VAD, opens a microphone or runs acoustic
perception. Output-device selection and the existing speech quality/speed controls
remain available. **Use microphone** explicitly switches back to speech input;
this requires a configured recognition model. A mode change replaces the Voice
input epoch so recognition from the old microphone session cannot arrive later
as a new typed message.

For a direct voice audition, use the separate text editor in **Settings → Voice
→ Conversation** and choose **Speak text**, or press **Enter** there. This reads
the literal text without calling the Agent, changing the main chat composer or
needing an LLM server. **Shift+Enter** inserts a newline. This preview remains
available while the Agent works. If Voice is off, Speak text starts output-only
Voice, so a missing recognition model cannot block an audition. A committed
Agent reply can interrupt a preview so the answer is not lost behind it.

Ordinary typed messages in the main chat also get spoken responses while Voice
is active. A new Agent turn interrupts any direct preview rather than waiting
behind its audio. Models warm in a background worker and stay cached, and PCM
streams through bounded queues.
Playback begins as soon as the first synthesized chunk is available; actual
latency depends on the voice, text length and CPU. Only committed replies are
narrated; a slow Agent still adds its own response delay.

Language routing is automatic for every microphone utterance and every
spoken reply, even when an older saved setting says English or Persian. Whisper
detects the spoken language per utterance; the reply router selects configured
voices from the reply's writing system and can switch between Persian and English
within one response. Available voices are loaded without requiring every language
at startup. For example, **Kokoro model + voices** can speak English without an
English Piper voice, while Persian still requires its own configured Piper voice.
If a reply needs a language with no local voice, Voice reports that specific
missing voice and stays active. It never silently downloads or substitutes a
different language. Microphone mode still needs a local Whisper model.

## One-time model setup

In **Settings → Voice → Models**, choose **Download selected speech model**,
**Download Persian voice**, and **Download English voice** as needed, or run
explicit commands:

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

Interrupted downloads now retain partial bytes under the voice cache's
`downloads/` directory. Retrying the same model, including after restarting
Studio, uses HTTP Range to continue from the saved offset. Transient network
errors get up to three attempts; each read has a 30-second timeout. Progress
reports connecting, resuming, retrying and verified cache reuse. Completed files
are verified before publication; a known checksum mismatch discards invalid
bytes. A server that ignores Range or changes its validator is rejected while
preserving the old prefix, rather than silently restarting a large transfer.
Only immutable commit/asset URLs share partials; a different revision has a
separate cache. Two processes cannot write the same transfer concurrently.
Verified transfers are cached separately from the installed model files, so
installation requires additional disk space. This cannot restore partial files
already deleted by older Daena versions.

Models live under the user's cache (`~/.cache/daena/voice` on typical Linux hosts;
see `python -m src.voice models` for the actual paths). Per-workspace preferences
are in `.daena/studio/voice.json`. No microphone recordings are written to disk by
Voice Mode. Recognized text becomes ordinary chat history when submitted.

If Hugging Face is unavailable or returns403 in your location, download the same
model files through an accessible connection and choose their local paths in
**Settings → Voice → Models**. A Whisper CTranslate2 directory needs `model.bin`, `config.json`, tokenizer
and vocabulary files. Each Piper voice needs the `.onnx` file and matching
`.onnx.json`. No private access token is required by this implementation.

## Persian and English quality

The default recognition model is multilingual **large-v3-turbo**, with CPU int8,
four CPU threads and beam search3. This keeps the GPU free for a large local
Ollama model. Choose **small** to reduce memory/load time, or explicitly choose a
supported CUDA/compute combination. Models are cached and warmed in background
workers. Language is detected automatically for each utterance so switching between
Persian and English does not require changing a global language setting.

The default catalog voices are Persian **Amir medium** and English **Lessac high**
Piper voices. You can select **Kokoro** as the English engine instead. Automatic TTS handles script changes with the corresponding voice;
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

## Optional Kokoro English

Install `.[voice-kokoro]`, then choose **Kokoro** under **Settings → Voice → Models** and download its
model plus voices. The catalog offers the full-precision v1.0 ONNX model (~326MB)
and matching voices archive (~28MB) from the official Kokoro ONNX release. You
can also select these files from disk. Start never downloads anything.

```bash
python -m src.voice download kokoro-en
python -m src.voice synthesize --engine kokoro --language en --voice af_heart --text 'Hello, I am Daena. How can I help?' --output daena-kokoro.wav
```

Kokoro officially has **no Persian voice**. A configured Persian Piper model
handles Persian, including mixed-script replies. Old or missing Piper paths do
not block Kokoro's English output. Choose a bundled English voice or enter its
exact ID from your voices archive; British voices use the British phonemizer.
Speaking speed changes the engine's duration controls, rather than changing the
playback sample rate and pitch.

Kokoro uses two CPU inference threads by default, separately from Whisper's
thread setting. More threads are not necessarily faster, especially while Ollama
is running. Models are loaded once per selected configuration. Kokoro is an
optional dependency so installing or using Daena without Voice remains unchanged.
The engine is MIT; the Kokoro weights are Apache-2.0. Other dependency licenses
still apply.

## Acoustic perception: speakers and surrounding sounds

Install `.[voice-perception]`, open **Settings → Voice → Perception**, and explicitly enable local
audio perception. Download the speaker model (~26.5MB) and/or the sound-event
model. The sound archive is ~111MB; only its int8 model (~26MB) and labels are
installed. Background sound monitoring is a separate option. These are local
Sherpa ONNX models, without PyTorch or a paid API.

```bash
python -m src.voice download speaker-embedding
python -m src.voice download audio-events
# Optional diagnostic, without recognition or microphone capture:
python -m src.voice analyze room-16khz-mono-pcm16.wav --ambient
```

- **Speaker grouping uses the waveform**, not the transcript. Voices receive
  anonymous session labels such as `speaker_1`; similar later utterances can
  match that label. Short, silent or ambiguous samples remain uncertain. This
  does not determine a real person's name, authenticate anyone or identify all
  speakers inside an overlapping recording. The chosen VoxCeleb model is not a
  Persian-specific diarization benchmark.
- Similarity is a cosine score, **not a confidence probability**. The conservative
  default match threshold is .88; changing microphone, language or background
  noise may split the same person's voice into separate labels. You can adjust
  the threshold; uncertain is preferable to an invented identity. At most eight
  speaker centroids are held in memory and they reset for each Voice session.
  Observation payloads include a fresh Voice-session ID so labels in separate
  sessions cannot be mistaken for a persistent identity.
- Sound tagging analyzes raw two-second windows even if speech VAD finds nothing.
  It can recognize fixed AudioSet classes such as laughter, music or finger
  snapping. Low scores produce **unknown**. This is a closed-set classifier,
  not a reliable detector of every new sound in the world. Demographic speech
  categories are projected to generic speech/singing rather than personal traits.
- Processing has its own bounded worker. Slow inference drops old pending sound
  windows; errors leave transcription and playback working. Analysis uses at
  most eight seconds of each utterance. No waveform or speaker embedding is
  written to disk. If you send a recognized request, bounded matching speech
  evidence and recent ambient labels can accompany it as uncertain sensor data.
  These labels then remain in ordinary conversation history with the request.
  They never become instructions or trigger an agent task on their own.
- To avoid associating a whole mixed draft with one speaker, combined drafts do
  not get a single-speaker label. Ordinary typed messages receive no microphone
  observations. Auto-send waits at most 400ms for matching acoustic evidence;
  unavailable or late evidence does not block the conversation. Muting clears
  partial ambient windows, and canceled/session-old results are discarded.

The advanced catalog pins official GitHub release asset IDs, checks file sizes
and available SHA-256 digests, and stores source/license notes beside the files.
Archives are streamed and only known regular model/label files are copied; no
general archive extraction is used. A failed download does not replace a working
installation. GitHub access/rate limits can still prevent downloads; selecting
the same local files works without API access or credentials.

## Turn-taking and interruption

Choose **Settings → Voice → Audio → Interrupt by speaking** and apply the selection:

- **Automatic** uses local WebRTC or SpeexDSP echo cancellation. Microphone capture
  continues during replies when a backend is active. The canceller receives the
  audio actually consumed by the playback device, rather than future queued TTS.
  If no backend is available, capture pauses during replies and **Stop reply**
  remains available. The Audio page displays the applied backend and status
  separately from any edited, unapplied preference.
- **Headphones** keeps microphone capture open without software echo cancellation.
  Wear headphones so Daena's output does not re-enter the microphone; software
  cannot determine acoustic isolation reliably from a device name.
- **Manual** pauses microphone input during replies. Use **Stop reply** to stop the
  answer and resume listening.

With an interruptible mode active, speaking over a reply stops queued playback,
cancels the old speech epoch and asks an active conversational Agent turn to stop.
The microphone utterance that triggered interruption is preserved, including its
beginning. Late audio and final messages from the interrupted turn cannot restart
speech. If **Hands-free** is enabled, the complete next utterance is submitted once
the interrupted Agent has finished cleanup; it does not start a concurrent Agent.
Edited drafts, session changes, muting, truncated utterances or turning Hands-free
off prevent that automatic submission and leave words available for review.

- Recognized speech appears in the Voice screen and composer. With Hands-free off,
  review it and press **Send** or Enter in chat. Complete utterances can otherwise
  auto-submit when the Agent is idle and there is no unrelated draft. Speech during
  an interrupted turn follows the deferred submission described above.
- Endpointing uses20ms frames, short pre-roll and configurable silence detection.
  WebRTC VAD is installed by the voice extra; an adaptive energy fallback is
  available. A duration cap marks a partial utterance for review and suppresses
  its continuing tail until actual silence. Silence duration and utterance limits
  are configured under **Settings → Voice → Audio**.
- During model warmup, live speech is not queued until recognition is ready.
  Wait for **Listening** before talking; the meter can be active while models
  prepare. This prevents normal startup from filling the speech queue.
- **Stop reply** immediately clears buffered speech, invalidates late STT/TTS
  results and requests cancellation of the active conversational agent turn.
  Native inference itself may finish later; it is not forcibly killed, and its
  canceled audio/text cannot be submitted. Restart waits for old native work to
  finish so a second copy of the model is not loaded on top.
- Echo cancellation and voice-activity detection are best effort. Room acoustics,
  speaker volume, microphone placement, device buffering and Bluetooth latency
  affect interruption and recognition. The adaptive filter needs a short startup
  period; a speaker or noisy room still needs testing on your hardware. Headphones
  avoid most acoustic coupling. Manual mute always suspends microphone input.
- Only committed final responses are spoken by default. Optional progress speech
  uses committed assistant messages. Raw drafts, reasoning, tool receipts and
  discarded retries are never narrated. Markdown is prepared for speech; fenced
  code and URL targets are omitted. Read-aloud is bounded to8000 characters;
  the complete answer stays in chat.

## Spoken response style

While Voice is active, Daena adds a temporary system instruction alongside its
ordinary instructions: use natural Persian or English, answer directly, normally
in one to three short sentences, and leave room for the next turn. It discourages
repeated introductions, filler and long recaps. Details remain available when
requested or required for correctness; code, commands and detailed evidence can
still appear in chat with a brief spoken explanation.

This changes presentation, not task scope, tool permissions, verification or the
Agent's abilities. It is an application instruction, not a hard response-token cap
or post-generation truncation. It applies to subsequent model requests while Voice
is active and is removed when Voice ends; it is not saved into conversation
history or the Agent's persistent profile. The configured model still determines
how well it follows that instruction and how quickly it responds.

## No sound: test the output path

Open **Settings → Voice → Audio**, choose the intended speakers/headphones and
apply the selection. **Refresh devices** updates the available device list;
it does not start microphone capture. **Test speaker** sends a short diagnostic
tone without loading Whisper or a TTS model. These controls remain available
while the Agent runs.

If the tone is inaudible, check the selected system output, application volume,
mute state and your desktop audio service before changing voice model files.
If the tone is audible but a text preview fails, inspect the error in Voice
settings and check the selected engine, matching local model/voice files and
supported language. The preview text is not sent to the Agent.

The playback state/error handling supports current Qt/PySide6 audio enums; output
errors are surfaced rather than treating queued PCM as successful playback. A
software message that a tone was sent cannot prove a physical speaker was heard.
Native tests can capture known Qt output through a PulseAudio null-sink monitor
to verify the application-to-audio-server path; they do not replace a listening
test on your actual device.

## Latency and resource limits

Speech recognition and synthesis have independent workers and bounded queues.
Synthesis is sentence-based and PCM is delivered in short chunks with backpressure
and streaming HQ Soxr sample-rate conversion. Resampler tails are flushed to
preserve final phonemes. Microphone/playback format negotiation handles native
rates and channel counts rather than assuming every device accepts16kHz.

The metrics show utterance-end-to-transcription, agent response time, queued-text-to-first-PCM
**measured on your machine**. They are not claims of audible device latency. End
silence, model warm-up, local language-model reasoning/generation and hardware
buffers add delay. Whisper is an utterance recognizer here, not native streaming
speech-to-speech. A slow35B reasoning model still determines the agent response
wait; lower its reasoning level in existing Studio controls if appropriate.
Whisper beam search and CPU threads are now exposed: beam1 trades some search
quality for speed; beam3 remains the default. Shorter endpoint silence reduces
turn delay but can split a hesitant utterance. The playback dispatch metric is
the first Qt playback notification after an agent turn, not physical speaker latency.

Verification includes scripted STT/TTS, real PCM/resampling, cancellation and
backpressure, Qt offscreen interaction, installed-wheel integration, plus actual
local Kokoro and speaker/sound ONNX inference, and a native Qt/PulseAudio monitor
loopback check. Example CPU measurements on this
build host: 2.84 seconds of Kokoro speech took 1.51 seconds with two threads;
combined speaker/sound analysis on speech fixtures took roughly 112–231ms;
laughter/finger-snap fixtures took roughly 96–108ms. These are fixtures, not
end-to-end microphone benchmarks or guarantees for another machine. Longer
sentences, model cold start and concurrent Ollama generation add latency. There
was no human listening evaluation or live microphone/speaker audition. Persian
naturalness and conversational timing still require testing on your hardware.

To run optional native model tests after explicitly provisioning the official
files, set `DAENA_TEST_VOICE_MODELS` to their directory and run the voice tests.
Regular CI checks native package imports and uses scripted models; it does not
download large weights automatically. This pipeline passes acoustic evidence to
the text agent; it is not an end-to-end audio language model or ChatGPT-equivalent
full-duplex voice system.

## Primary references

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper): local CTranslate2 transcription and VAD.
- [Whisper language support](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py): Persian `fa` and English `en`.
- [Piper Python API](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md): local chunked synthesis.
- [Published voice catalog](https://github.com/rhasspy/piper/blob/master/VOICES.md) and [voice licensing](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md).
- [Kokoro ONNX API](https://github.com/thewh1teagle/kokoro-onnx) and [official model files](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1).
- [Sherpa speaker embeddings](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speaker-identification.py) and [sound tagging](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/audio-tagging-from-a-file.py).
- [WebRTC audio-processing wrapper](https://pypi.org/project/aec-audio-processing/) and [SpeexDSP](https://www.speex.org/docs/) for optional local acoustic echo cancellation.
