"""Real, isolated Qt microphone capture continues through native voice output.

The microphone is an owned PulseAudio monitor, never a host microphone. This
checks I/O scheduling and gating; acoustic AEC quality is tested separately.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

pytest.importorskip("PySide6")


def test_real_qt_duplex_captures_while_output_plays_without_opening_host_mic(tmp_path):
    if os.environ.get("DAENA_TEST_AUDIO_LOOPBACK") != "1":
        pytest.skip("Enable DAENA_TEST_AUDIO_LOOPBACK=1 for isolated native duplex I/O")
    bindir = os.environ.get("DAENA_TEST_PULSE_BIN", "")
    pulse = str(Path(bindir) / "pulseaudio") if bindir else shutil.which("pulseaudio")
    if not pulse:
        pytest.skip("PulseAudio is required for owned monitor-source integration")
    with tempfile.TemporaryDirectory(prefix="daena-duplex-") as directory:
        socket = Path(directory) / "native"
        env = dict(
            os.environ,
            PULSE_SERVER=f"unix:{socket}",
            PULSE_SOURCE="daena_voice_mic",
            PULSE_SINK="daena_voice_test",
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
            "-L",
            "module-remap-source master=daena_voice_test.monitor source_name=daena_voice_mic "
            "source_properties=device.description=Daena_Test_Microphone",
        ]
        with (tmp_path / "pulse-duplex.log").open("wb") as log:
            server = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                deadline = time.monotonic() + 5
                while not socket.exists():
                    if server.poll() is not None or time.monotonic() >= deadline:
                        pytest.fail((tmp_path / "pulse-duplex.log").read_text())
                    time.sleep(0.02)
                result = subprocess.run(
                    [sys.executable, "-c", NATIVE_DUPLEX],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                assert result.returncode == 0, result.stdout + result.stderr
                receipt = json.loads(result.stdout.strip().splitlines()[-1])
                assert receipt["errors"] == []
                assert receipt["half"]["during"] == 0
                assert receipt["duplex"]["during"] >= 12, receipt
                assert receipt["duplex"]["maximum"] > 0.05
                # Native server/source latency can shift monitor samples beyond
                # the sink's ActiveState; require substantial overlap without
                # pretending that state is a physical acoustic clock.
                assert receipt["duplex"]["nonzero"] >= 3200, receipt
            finally:
                server.terminate()
                server.wait(timeout=3)


NATIVE_DUPLEX = r"""
import json
import numpy as np
from PySide6.QtCore import QCoreApplication, QTimer
from src.studio.voice_audio import AudioDevices

app = QCoreApplication([])
receipt = {"errors": [], "half": {"during": 0, "maximum": 0, "nonzero": 0},
           "duplex": {"during": 0, "maximum": 0, "nonzero": 0}}
state = {"audio": None, "phase": "half", "index": 0}
feed_timer = QTimer()
feed_timer.setInterval(100)

def capture(pcm, owner, phase):
    if owner._playing:
        values = np.frombuffer(pcm, dtype='<i2').astype(np.float64) / 32768
        row = receipt[phase]
        row["during"] += 1
        row["maximum"] = max(row["maximum"], float(np.sqrt(np.mean(values * values))))
        row["nonzero"] += int(np.count_nonzero(np.abs(values) > 0.01))

def feed():
    audio = state["audio"]
    if state["index"] >= 24:
        audio.finish_playback()
        feed_timer.stop()
        return
    index = state["index"]
    values = np.arange(index * 2400, (index + 1) * 2400)
    pcm = (np.sin(values * 2 * np.pi * 440 / 24000) * 20000).astype('<i2').tobytes()
    assert audio.play_pcm(pcm, 24000)
    state["index"] += 1

def start_phase(mode):
    audio = AudioDevices()
    state["audio"], state["index"] = audio, 0
    phase = state["phase"]
    audio.fault.connect(receipt["errors"].append)
    audio.configure_duplex(mode)
    audio.pcm_ready.connect(lambda pcm, owner=audio, key=phase: capture(pcm, owner, key))
    audio.prepare_output()
    assert audio.start_capture()
    def begin_output():
        # Start with a short cushion so timer jitter does not turn this into an
        # underrun test and mark every 100ms boundary as output-idle.
        feed()
        feed()
        feed_timer.start()
    QTimer.singleShot(250, begin_output)

def second_phase():
    feed_timer.stop()
    state["audio"].close()
    state["phase"] = "duplex"
    start_phase("headphones")

feed_timer.timeout.connect(feed)
QTimer.singleShot(100, lambda: start_phase("off"))
QTimer.singleShot(3500, second_phase)
QTimer.singleShot(7000, app.quit)
app.exec()
feed_timer.stop()
state["audio"].close()
print(json.dumps(receipt))
"""
