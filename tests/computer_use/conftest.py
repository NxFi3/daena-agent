from PIL import Image
import pytest

from src.tools.builtin.computer_use.contracts import Capture, Rect, Window
from src.tools.builtin.computer_use.computer_use import ComputerUseTool
from src.tools.builtin.computer_use.runtime import ComputerError


class FakeBackend:
    name = "fixture"

    def __init__(self):
        self.window = Window(
            123, 99, "Fixture", "Fixture app", Rect(100, 80, 640, 480), True
        )
        self.rows = [
            {
                "element_token": "name",
                "role": "entry",
                "label": "Name",
                "value": "",
                "frame": {"x": 20, "y": 30, "w": 200, "h": 40},
                "actions": ["set_value"],
            },
            {
                "element_token": "save",
                "role": "button",
                "label": "Save",
                "frame": {"x": 20, "y": 100, "w": 90, "h": 40},
                "actions": ["click"],
            },
            {
                "element_token": "enabled",
                "role": "check box",
                "label": "Enabled",
                "checked": False,
                "frame": {"x": 150, "y": 100, "w": 120, "h": 40},
                "actions": ["click"],
            },
        ]
        self.calls = []
        self.captures = 0
        self.color = "#192438"
        self.failure = None
        self.truncated = False
        self.capture_failure = False
        self.frame = "window"

    def ready(self, session):
        pass

    def windows(self, session):
        return [self.window]

    def capture(self, session, window, output, *, pixels, accessibility):
        if self.capture_failure:
            raise ComputerError("capture_failed", "fixture capture failed")
        self.captures += 1
        if pixels:
            Image.new("RGB", (640, 480), self.color).save(output)
        return Capture(
            str(output) if pixels else None,
            f"cap-{self.captures}",
            self.window if window else None,
            "window" if window else "desktop",
            list(self.rows) if accessibility else [],
            self.frame,
            truncated=self.truncated,
        )

    def dispatch(self, session, action, arguments):
        self.calls.append((action, dict(arguments)))
        if self.failure:
            raise self.failure
        if action == "type_text":
            self.rows[0]["value"] = arguments["text"]
        if action == "click" and arguments.get("element_token") == "enabled":
            self.rows[2]["checked"] = not self.rows[2]["checked"]
        return {"status": "dispatched"}

    def doctor(self):
        return {"backend": self.name, "installed": True}


@pytest.fixture
def engine(tmp_path):
    backend = FakeBackend()
    tool = ComputerUseTool(
        backend, workspace=tmp_path, lock_path=tmp_path / "desktop.lock"
    )
    return tool, backend


def capture(tool, **kwargs):
    result = tool.execute("capture", **kwargs)
    assert result.success, result.content
    return result.content["snapshot_id"]
