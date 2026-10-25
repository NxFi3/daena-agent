"""Explicit local model setup and offline voice audition; no microphone on import."""

import argparse
import importlib.util
import json
from pathlib import Path
import wave


def main(argv=None):
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
    download.add_argument(
        "key", choices=["whisper-turbo", "whisper-small", "piper-fa", "piper-en"]
    )
    audition = commands.add_parser(
        "synthesize", help="Render local speech to a WAV for audition"
    )
    audition.add_argument("--text", required=True)
    audition.add_argument("--language", choices=["auto", "fa", "en"], default="auto")
    audition.add_argument("--fa-model", type=Path)
    audition.add_argument("--en-model", type=Path)
    audition.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from .models import installed_models, model_catalog, download_model, voice_paths

    if args.command == "doctor":
        print(
            json.dumps(
                {
                    "packages": {
                        name: importlib.util.find_spec(name) is not None
                        for name in ("faster_whisper", "piper", "webrtcvad", "soxr")
                    },
                    "models": installed_models(),
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
        provider = PiperProvider(paths, language=args.language)
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
