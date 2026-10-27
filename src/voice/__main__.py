"""Explicit local model setup and offline voice audition; no microphone on import."""

import argparse
import importlib.util
import json
from pathlib import Path
import wave


def main(argv=None):
    from .models import installed_models, model_catalog, download_model, voice_paths

    parser = argparse.ArgumentParser(description="Daena local Persian / English speech")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Check optional packages and installed models without recording"
    )
    commands.add_parser(
        "models", help="List explicit download keys and installed paths"
    )
    download = commands.add_parser(
        "download", help="Explicitly download one public speech model"
    )
    download.add_argument("key", choices=[row["key"] for row in model_catalog()])
    audition = commands.add_parser(
        "synthesize", help="Render local speech to a WAV for audition"
    )
    audition.add_argument("--text", required=True)
    audition.add_argument("--language", choices=["auto", "fa", "en"], default="auto")
    audition.add_argument("--fa-model", type=Path)
    audition.add_argument("--en-model", type=Path)
    audition.add_argument("--engine", choices=["piper", "kokoro"], default="piper")
    audition.add_argument("--kokoro-model", type=Path)
    audition.add_argument("--kokoro-voices", type=Path)
    audition.add_argument("--voice", default="af_heart")
    audition.add_argument("--speed", type=float, default=1.0)
    audition.add_argument("--cpu-threads", type=int, default=2)
    audition.add_argument("--output", type=Path, required=True)
    analyze = commands.add_parser(
        "analyze", help="Analyze a local 16 kHz mono PCM16 WAV without transcribing it"
    )
    analyze.add_argument("wav", type=Path)
    analyze.add_argument("--speaker-model", type=Path)
    analyze.add_argument("--sound-model", type=Path)
    analyze.add_argument("--sound-labels", type=Path)
    analyze.add_argument(
        "--ambient",
        action="store_true",
        help="Skip speaker embeddings for non-speech audio",
    )
    args = parser.parse_args(argv)
    if args.command == "synthesize" and not 0.7 <= args.speed <= 1.4:
        parser.error("Speech speed must be between 0.7 and 1.4")

    if args.command == "doctor":
        from .duplex import backend_status

        print(
            json.dumps(
                {
                    "packages": {
                        name: importlib.util.find_spec(name) is not None
                        for name in (
                            "faster_whisper",
                            "piper",
                            "webrtcvad",
                            "soxr",
                            "kokoro_onnx",
                            "sherpa_onnx",
                        )
                    },
                    "models": installed_models(),
                    "duplex": backend_status(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "models":
        print(
            json.dumps(
                {"catalog": model_catalog(), "installed": installed_models()},
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "download":
        print(json.dumps(download_model(args.key), ensure_ascii=False, indent=2))
    elif args.command == "analyze":
        from .perception import LocalAudioPerception

        models = installed_models()
        speaker = args.speaker_model or models.get("speaker-embedding", {}).get(
            "path", ""
        )
        sound = args.sound_model or models.get("audio-events", {}).get("path", "")
        labels = args.sound_labels or (
            Path(sound).parent / "class_labels_indices.csv" if sound else ""
        )
        if not speaker and not sound:
            parser.error("Select/download a speaker or sound-event model first")
        with wave.open(str(args.wav), "rb") as recording:
            if (
                recording.getframerate(),
                recording.getnchannels(),
                recording.getsampwidth(),
                recording.getcomptype(),
            ) != (16000, 1, 2, "NONE"):
                parser.error("Analyze expects an uncompressed 16 kHz mono PCM16 WAV")
            total_frames = recording.getnframes()
            pcm = recording.readframes(16000 * 8)
        provider = LocalAudioPerception(str(speaker), str(sound), str(labels))
        result = provider.analyze(pcm, speech=not args.ambient)
        result["file_truncated"] = total_frames > 16000 * 8
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    else:
        from .providers import PiperProvider
        from .streaming import speech_segments

        if args.output.exists():
            parser.error("Output already exists; choose a new WAV filename")
        paths = voice_paths()
        if args.fa_model:
            paths["fa"] = args.fa_model
        if args.en_model:
            paths["en"] = args.en_model
        provider = PiperProvider(
            paths, language=args.language, length_scale=1 / args.speed
        )
        if args.engine == "kokoro":
            from .kokoro import KokoroProvider, BilingualSynthesizer

            models = installed_models()
            model = args.kokoro_model or models.get("kokoro-en", {}).get("path", "")
            voices = args.kokoro_voices or (
                Path(model).parent / "voices-v1.0.bin" if model else ""
            )
            provider = BilingualSynthesizer(
                english=KokoroProvider(
                    model,
                    voices,
                    voice=args.voice,
                    speed=args.speed,
                    cpu_threads=args.cpu_threads,
                ),
                persian=PiperProvider(
                    {"fa": paths.get("fa", "")},
                    language="fa",
                    length_scale=1 / args.speed,
                ),
                language=args.language,
            )
        # Mixing different native voice sample rates needs resampling. For
        # audition use one explicit language, or a consistent native rate.
        try:
            with wave.open(str(args.output), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                rate = None
                for text in speech_segments(args.text):
                    for chunk in provider.synthesize(text, args.language):
                        if rate is None:
                            rate = chunk.sample_rate
                            output.setframerate(rate)
                        if rate != chunk.sample_rate:
                            raise ValueError(
                                "Mixed voice sample rates: audition fa/en separately; Studio resamples for playback"
                            )
                        output.writeframesraw(chunk.pcm16)
                if rate is None:
                    raise ValueError("No speakable text")
        except Exception:
            args.output.unlink(missing_ok=True)
            raise
        print(str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
