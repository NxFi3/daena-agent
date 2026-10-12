from threading import Event

from cli.app import DEFAULT_BUSY_MODE, _route_busy_text


class FakeConsole:
    def __init__(self):
        self.messages = []

    def print(self, message):
        self.messages.append(str(message))


def test_busy_mode_defaults_to_steer_so_a_stalled_task_does_not_block_new_input():
    assert DEFAULT_BUSY_MODE == "steer"
    stop_event = Event()
    state = {"stop_event": stop_event}
    console = FakeConsole()

    _route_busy_text(console, state, DEFAULT_BUSY_MODE, "use computer use tool")

    assert state["pending_followups"] == ["use computer use tool"]
    assert stop_event.is_set()
    assert "message queued as task #1" in console.messages[-1]


def test_queue_mode_still_allows_users_to_wait_for_the_current_task():
    console = FakeConsole()
    stop_event = Event()
    state = {"stop_event": stop_event}

    _route_busy_text(console, state, "queue", "first question")

    assert state["pending_followups"] == ["first question"]
    assert not stop_event.is_set()
    assert "will finish first" in console.messages[-1]


def test_multiple_busy_messages_are_queued_in_order():
    console = FakeConsole()
    state = {"stop_event": Event()}

    _route_busy_text(console, state, "queue", "first question")
    _route_busy_text(console, state, "queue", "second question")

    assert state["pending_followups"] == ["first question", "second question"]
    assert "#1" in console.messages[0]
    assert "#2" in console.messages[1]


def test_steer_mode_interrupts_but_preserves_the_redirect_as_a_fresh_task():
    console = FakeConsole()
    stop_event = Event()
    state = {"stop_event": stop_event}

    _route_busy_text(console, state, "steer", "bebin bluebank chie?")

    assert stop_event.is_set()
    assert state["pending_followups"] == ["bebin bluebank chie?"]
    assert "message queued as task #1" in console.messages[-1]
