"""Run with an isolated wheel installation, outside the source checkout."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from PIL import Image
import computer_use
from computer_use import ComputerUseTool
from computer_use.contracts import Capture

assert not Path(computer_use.__file__).resolve().is_relative_to(Path(__file__).parents[1])
assert importlib.util.find_spec("src") is None, (
    "Use an isolated environment without Daena to verify the standalone adapter."
)


class Backend:
    name = "wheel-fixture"

    def ready(self, session):
        pass

    def windows(self, session):
        return []

    def capture(self, session, window, output, **kwargs):
        Image.new("RGB", (200, 100), "#151d2e").save(output)
        return Capture(str(output), "wheel-cap", None, "desktop")

    def dispatch(self, session, action, args):
        return {"status": "dispatched"}


with TemporaryDirectory() as directory:
    tool = ComputerUseTool(
        Backend(), workspace=directory, lock_path=Path(directory) / "desktop.lock"
    )
    result = tool.execute("capture", scope="desktop", mode="vision")
    assert result.success and result.content["width"] == 200
    assert tool.get_definition()["function"]["name"] == "computer_use"
    click = tool.execute(
        "click", snapshot_id=result.content["snapshot_id"], coordinate=[40, 30], seconds=0
    )
    assert click.success and click.content["dispatch_status"] == "dispatched"
    assert Path(click.metadata["image_ref"]).is_file()
    command = subprocess.run(
        [sys.executable, "-m", "computer_use", "--workspace", directory, "doctor"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    report = json.loads(command.stdout)
    assert "success" in report and "Traceback" not in command.stderr
print("INSTALLED_COMPUTER_WHEEL_OK")
