"""Stable Unicode keymaps with an EWMH client-processing barrier.

libxdo's per-character temporary-keycode reuse can race clients which lazily
refresh XKB mappings (notably Qt). Map each distinct missing keysym to its own
unused code for the entire chunk, and wait for the target's _NET_WM_PING reply
before restoring those codes. No clipboard or website-specific behavior.
"""

from contextlib import closing
import select
import time
import uuid

from .runtime import ComputerError, checkpoint


def keysym(char):
    value = ord(char)
    return value if value <= 255 else 0x01000000 | value


def chunks(text, missing, capacity):
    current, symbols = [], set()
    for char in text:
        symbol = keysym(char)
        if symbol in missing and symbol not in symbols and len(symbols) == capacity:
            yield "".join(current), symbols
            current, symbols = [], set()
        current.append(char)
        if symbol in missing:
            symbols.add(symbol)
    if current:
        yield "".join(current), symbols


def ping(display, window, *, dispatched=False):
    from Xlib import X
    from Xlib.protocol.event import ClientMessage

    protocol = display.intern_atom("WM_PROTOCOLS")
    atom = display.intern_atom("_NET_WM_PING")
    nonce = uuid.uuid4().int & 0xFFFFFFFF
    message = ClientMessage(
        window=window, client_type=protocol, data=(32, [atom, nonce, window.id, 0, 0])
    )
    window.send_event(message, event_mask=0)
    display.flush()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not dispatched:
            checkpoint()
        while display.pending_events():
            event = display.next_event()
            if event.type == X.ClientMessage and event.client_type == protocol:
                values = event.data[1]
                if list(values[:3]) == [atom, nonce, window.id]:
                    return
        select.select(
            [display.fileno()], [], [], min(0.05, max(0, deadline - time.monotonic()))
        )
    raise ComputerError(
        "input_timeout" if dispatched else "target_unresponsive",
        "The target did not acknowledge X11 input processing; text is never replayed.",
        dispatched=dispatched,
    )


def type_stable(text, window_id, send):
    """Call send(chunk) with private, stable Unicode mappings; always restore them."""
    from Xlib import X
    from Xlib.display import Display

    with closing(Display()) as display:
        missing = {
            keysym(c)
            for c in text
            if ord(c) > 127 and not display.keysym_to_keycode(keysym(c))
        }
        if not missing:
            send(text)
            return
        if window_id is None:
            raise ComputerError(
                "target_required",
                "Unicode X11 typing requires an explicit application window. Capture that window first.",
            )
        window = display.create_resource_object("window", int(window_id))
        if display.intern_atom("_NET_WM_PING") not in (window.get_wm_protocols() or []):
            raise ComputerError(
                "unsupported_text_input",
                "This X11 application cannot acknowledge temporary Unicode keymaps. Use browser_use for web forms or the Cua native text backend.",
            )
        root = display.screen().root
        # Qt replies with StructureNotify; other EWMH clients use
        # SubstructureNotify. Neither mask claims window-manager ownership.
        root.change_attributes(
            event_mask=X.SubstructureNotifyMask | X.StructureNotifyMask
        )
        first, last = display.display.info.min_keycode, display.display.info.max_keycode
        mapping = display.get_keyboard_mapping(first, last - first + 1)
        spare = [
            (first + i, tuple(row)) for i, row in enumerate(mapping) if not any(row)
        ]
        if not spare:
            raise ComputerError(
                "unsupported_text_input",
                "No unused X11 keycodes are available for reliable Unicode typing. Use browser_use or Cua native text input.",
            )
        for text_chunk, symbols in chunks(text, missing, len(spare)):
            changed = []
            try:
                for symbol, (code, original) in zip(sorted(symbols), reversed(spare)):
                    replacement = (symbol,) * len(original)
                    display.change_keyboard_mapping(code, [replacement])
                    # XKB canonicalizes repeated groups and may add case pairs.
                    # Compare against what the server installed, not our request.
                    installed = tuple(display.get_keyboard_mapping(code, 1)[0])
                    changed.append((code, original, installed))
                display.sync()
                ping(display, window)
                try:
                    send(text_chunk)
                except BaseException:
                    # Complete the server/client barrier even after cancellation,
                    # so pending key events do not see restored/reused keycodes.
                    try:
                        ping(display, window, dispatched=True)
                    except ComputerError:
                        pass
                    raise
                ping(display, window, dispatched=True)
            finally:
                for code, original, replacement in changed:
                    current = tuple(display.get_keyboard_mapping(code, 1)[0])
                    if current == replacement:
                        display.change_keyboard_mapping(code, [original])
                display.sync()
