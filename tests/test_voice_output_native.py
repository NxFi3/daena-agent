"""Qt enum compatibility and optional real playback→Pulse monitor verification."""

from enum import Enum
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
from src.studio.voice_audio import AudioDevices, _audio_enum_is
from tests.test_voice_audio import Audio, Device, FakeSink, Format, Media


def test_qt_audio_alias_wrappers_start_playback_and_do_not_report_false_errors(
    qt_app, monkeypatch
):
    import src.studio.voice_audio as module

    # PySide 6.12 exposes distinct QAudio/QtAudio Python classes for the same
    # underlying C++ members. Equal-looking names/hashes do not imply equality.
    State = Enum(
        "State",
        {"StoppedState": 0, "ActiveState": 1, "IdleState": 2, "SuspendedState": 3},
    )
    StateAlias = Enum(
        "State",
        {"StoppedState": 0, "ActiveState": 1, "IdleState": 2, "SuspendedState": 3},
    )
    Error = Enum("Error", {"NoError": 0, "IOError": 2})
    ErrorAlias = Enum("Error", {"NoError": 0, "IOError": 2})
    assert State.StoppedState != StateAlias.StoppedState
    assert _audio_enum_is(State.StoppedState, StateAlias.StoppedState)
    assert not _audio_enum_is(State.StoppedState, Error.NoError)

    class Enums:
        StoppedState, ActiveState, IdleState, SuspendedState = list(State)
        NoError = Error.NoError

    class AliasSink(FakeSink):
        def __init__(self, *args):
            super().__init__(*args)
            self.current = StateAlias.StoppedState
            self.current_error = ErrorAlias.NoError

        def error(self):
            return self.current_error

        def start(self, source):
            self.source = source
            self.starts += 1
            self.current = StateAlias.ActiveState
            self.stateChanged.emit(self.current)

        def stop(self):
            self.current = StateAlias.StoppedState
            self.stateChanged.emit(self.current)

    monkeypatch.setattr(
        module, "_qt_audio", lambda: (Enums, Format, AliasSink, None, Media)
    )
    audio = AudioDevices()
    started, errors = [], []
    audio.playback_started.connect(lambda: started.append(True))
    audio.fault.connect(errors.append)
    assert audio.play_pcm(b"\1\0" * 1600, 16000)
    sink = audio._sink
    assert sink.starts == 1 and started == [True] and not errors
    sink.current_error = ErrorAlias.IOError
    sink.current = StateAlias.StoppedState
    sink.stateChanged.emit(sink.current)
    assert errors and audio._sink is None
    audio.close()


def test_playback_started_requires_native_active_state(qt_app, monkeypatch):
    import src.studio.voice_audio as module

    class PendingSink(FakeSink):
        def start(self, source):
            self.source = source
            self.starts += 1
            # Asynchronous backend setup: PCM is queued, not playing yet.

    monkeypatch.setattr(
        module, "_qt_audio", lambda: (Audio, Format, PendingSink, None, Media)
    )
    audio = AudioDevices()
    started = []
    audio.playback_started.connect(lambda: started.append(True))
    assert audio.play_pcm(b"\1\0" * 1600, 16000)
    assert not started
    audio._sink.current = Audio.ActiveState
    audio._sink.stateChanged.emit(Audio.ActiveState)
    assert started == [True]
    audio.close()


def test_output_preflight_validates_selected_speaker_without_microphone(
    qt_app, monkeypatch
):
    import src.studio.voice_audio as module

    class OutputMedia(Media):
        @staticmethod
        def audioInputs():
            pytest.fail("Output preflight queried microphones")

        @staticmethod
        def defaultAudioInput():
            pytest.fail("Output preflight queried microphones")

    monkeypatch.setattr(
        module, "_qt_audio", lambda: (Audio, Format, FakeSink, None, OutputMedia)
    )
    audio = AudioDevices()
    assert audio.prepare_output() == {
        "id": b"test-device".hex(),
        "name": "Test device",
        "sample_rate": 16000,
        "channels": 1,
        "volume": 1.0,
    }
    assert audio._source is None and audio._sink is None
    audio.set_output_device("removed-device")
    with pytest.raises(RuntimeError, match="no longer available"):
        audio.prepare_output()
    audio.close()


def test_real_qt_output_reaches_pulse_monitor_without_microphone(tmp_path):
    if os.environ.get("DAENA_TEST_AUDIO_LOOPBACK") != "1":
        pytest.skip(
            "Enable DAENA_TEST_AUDIO_LOOPBACK=1 for owned PulseAudio null-sink integration"
        )
    bindir = os.environ.get("DAENA_TEST_PULSE_BIN", "")
    pulse = str(Path(bindir) / "pulseaudio") if bindir else shutil.which("pulseaudio")
    parec = str(Path(bindir) / "parec") if bindir else shutil.which("parec")
    if not pulse or not parec:
        pytest.skip("PulseAudio and parec are required for native loopback")
    with tempfile.TemporaryDirectory(prefix="daena-pulse-") as directory:
        runtime = Path(directory)
        socket = runtime / "native"
        env = dict(
            os.environ,
            PULSE_SERVER=f"unix:{socket}",
            XDG_RUNTIME_DIR=directory,
            QT_AUDIO_BACKEND="pulseaudio",
            QT_QPA_PLATFORM="offscreen",
        )
        command = [
            pulse,
            "--daemonize=no",
            "--exit-idle-time=-1",
            "--use-pid-file=no",
            "--disable-shm=yes",
            "-n",
        ]
        if os.environ.get("DAENA_TEST_PULSE_MODULES"):
            command.append("--dl-search-path=" + os.environ["DAENA_TEST_PULSE_MODULES"])
        command += [
            "-L",
            f"module-native-protocol-unix socket={socket} auth-anonymous=1",
            "-L",
            "module-null-sink sink_name=daena_voice_test rate=48000 channels=2",
        ]
        with (tmp_path / "pulse.log").open("wb") as log:
            server = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            recorder = None
            try:
                deadline = time.monotonic() + 5
                while not socket.exists():
                    if server.poll() is not None or time.monotonic() >= deadline:
                        pytest.fail((tmp_path / "pulse.log").read_text())
                    time.sleep(0.02)
                with (tmp_path / "monitor.pcm").open("wb") as recording:
                    recorder = subprocess.Popen(
                        [
                            parec,
                            "--device=daena_voice_test.monitor",
                            "--format=s16le",
                            "--rate=48000",
                            "--channels=2",
                            "--raw",
                        ],
                        env=env,
                        stdout=recording,
                        stderr=subprocess.DEVNULL,
                    )
                    result = subprocess.run(
                        [sys.executable, "-c", NATIVE_PLAYER],
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    assert result.returncode == 0, result.stdout + result.stderr
                    receipt = json.loads(result.stdout.strip().splitlines()[-1])
                    assert (
                        receipt["errors"] == []
                        and receipt["started"] > 0
                        and receipt["queued"] == 0
                    )
                    recorder.terminate()
                    recorder.wait(timeout=3)
                    recorder = None
                samples = np.fromfile(tmp_path / "monitor.pcm", dtype="<i2")
                assert samples.size and np.count_nonzero(samples) > 48000
                assert np.max(np.abs(samples.astype(np.int32))) > 5000
                nonzero = samples[np.abs(samples.astype(np.int32)) > 100]
                assert np.sqrt(np.mean(nonzero.astype(np.float64) ** 2)) > 5000
            finally:
                if recorder is not None:
                    recorder.terminate()
                    recorder.wait(timeout=3)
                server.terminate()
                server.wait(timeout=3)


NATIVE_PLAYER = r"""
import json
import numpy as np
from PySide6.QtCore import QCoreApplication, QTimer
from src.studio.voice_audio import AudioDevices
app=QCoreApplication([])
audio=AudioDevices()
receipt={"errors":[],"started":0}
audio.fault.connect(receipt["errors"].append)
def started(): receipt["started"]+=1
audio.playback_started.connect(started)
audio.prepare_output()
frames=[(np.sin(np.arange(i*2400,(i+1)*2400)*2*np.pi*440/24000)*20000).astype("<i2").tobytes() for i in range(20)]
def feed():
    if frames: assert audio.play_pcm(frames.pop(0),24000)
    else: audio.finish_playback(); timer.stop()
timer=QTimer(); timer.setInterval(100); timer.timeout.connect(feed)
QTimer.singleShot(300,timer.start)
QTimer.singleShot(5000,app.quit)
app.exec()
receipt["queued"]=audio.queued_seconds
audio.close()
print(json.dumps(receipt))
"""
