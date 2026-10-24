---
name: desktop-control
description: Control existing desktop windows and native applications with computer_use, screenshots, keyboard input and pointer gestures, including drawing on unlabelled surfaces.
---

Discover computer_use and read its schema. When the user refers to an existing application, list_windows first and capture its observed window_id. A desktop application's executable name need not match its title. Do not invent protocols, executable names or desktop-file paths. An empty window list is a backend observation, not proof that an application is uninstalled. Read doctor and window_diagnostics for missing dependencies, display permissions, or native Wayland windows invisible to X11. Explain an installation or session blocker instead of repeatedly trying the same unavailable capability.

Capture returns a snapshot_id. Use it for every input and use the next observation's ID afterward. Treat window IDs, snapshot IDs, and element IDs as distinct. Select controls by observed opaque ID or exact labels. Missing accessibility labels do not prevent pointer input on an observed image: use screenshot coordinates. Window capture coordinates start at the image's top-left; do not add the desktop window position. Zoom crops include a mapping back to native pixels. If no usable image is available, obtain one before choosing visual targets.

Use drag with from_coordinate and to_coordinate for a stroke or shape gesture. A click only clicks. For connected strokes, review each resulting observation before continuing. Choose foreground delivery explicitly when the backend requires it; do not repeatedly request unsupported background delivery. A capture does not activate or focus a window. Never substitute an unrelated toolbar operation for a missing surface target.

After a timeout, capture and inspect before deciding whether to retry: the action may already have happened. Delivered input and changed pixels do not establish the user's goal; inspect the result and report limitations honestly. Preserve the user's existing content unless changing it is authorized. Ask for a missing application or manual setup when the backend cannot observe the intended target.
