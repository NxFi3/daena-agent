"""Session-owned Chromium state. All methods run on one asyncio loop."""

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import uuid

from .contracts import BrowserError
from .dom import COLLECT
from .executable import resolve_executable


class BrowserSession:
    def __init__(self, workspace, session, config):
        self.workspace = Path(workspace).resolve()
        key = hashlib.sha256(str(session).encode()).hexdigest()[:24]
        self.root = self.workspace / ".daena" / "browser_use" / key
        self.config = config
        self.pw = self.browser = self.context = self.page = None
        self.attached = False
        self.pages, self.frames, self.refs = {}, {}, {}
        self.dialogs, self.downloads = {}, {}
        self.latest = None
        self.lock = asyncio.Lock()
        self.delivery = "not_sent"

    def _private_root(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    async def start(self):
        if self.context is not None:
            return
        if self.pw is not None:
            await self.close()
        if importlib.util.find_spec("playwright") is None:
            raise BrowserError(
                "dependency_missing",
                "Install python -m pip install -e '.[browser]' then python -m playwright install chromium.",
            )
        from playwright.async_api import async_playwright

        self._private_root()
        self.pw = await async_playwright().start()
        try:
            if self.config.get("cdp_url"):
                self.browser = await self.pw.chromium.connect_over_cdp(
                    self.config["cdp_url"], timeout=10000
                )
                self.attached = True
                if not self.browser.contexts:
                    raise BrowserError(
                        "no_browser_context",
                        "The configured CDP browser has no available context.",
                    )
                self.context = self.browser.contexts[0]
            else:
                options = dict(
                    headless=self.config.get("headless", False),
                    accept_downloads=True,
                    viewport=None,
                    timeout=15000,
                )
                executable, source = resolve_executable(self.config, self.pw.chromium.executable_path)
                # Let Playwright choose its bundled headed/headless variant.
                if source != "playwright":
                    options["executable_path"] = executable
                self.executable = executable
                self.executable_source = source
                self.context = await self.pw.chromium.launch_persistent_context(
                    str(self.root / "profile"), **options
                )
            self.context.on("page", self.register_page)
            self.context.on("close", self.context_closed)
            for page in self.context.pages:
                self.register_page(page)
            if not self.context.pages:
                self.register_page(await self.context.new_page())
            # Attach without selecting unrelated user work. open will create a
            # new tab until an existing tab is explicitly selected by its ID.
            self.page = None if self.attached else self.context.pages[0]
        except BaseException:
            await self.close()
            raise

    def context_closed(self, *_):
        self.context = self.page = None
        self.latest = None
        self.refs.clear()

    def register_page(self, page):
        if page in self.pages.values():
            return
        tab = "tab_" + uuid.uuid4().hex[:10]
        self.pages[tab] = page
        page.on("dialog", lambda dialog: self.dialogs.__setitem__(tab, dialog))
        page.on(
            "download",
            lambda download: self.downloads.__setitem__(
                "download_" + uuid.uuid4().hex[:10], (tab, download)
            ),
        )

    def tab_id(self, page):
        self.register_page(page)
        return next(k for k, p in self.pages.items() if p == page)

    async def tab_list(self):
        result = []
        for tab, page in list(self.pages.items()):
            if page.is_closed():
                continue
            try:
                title = await asyncio.wait_for(page.title(), 1)
            except Exception:
                title = ""
            result.append(
                dict(
                    tab_id=tab,
                    url=page.url,
                    title=title[:300],
                    selected=page == self.page,
                )
            )
        return result

    async def select_page(self, tab=None):
        if tab is not None:
            page = self.pages.get(tab)
            if page is None or page.is_closed():
                raise BrowserError(
                    "tab_not_found",
                    "Choose a live tab_id from tabs.",
                    tabs=await self.tab_list(),
                )
            if page != self.page:
                await self.clear_refs()
                self.frames = {}
                self.latest = None
            self.page = page
        if self.page is None or self.page.is_closed():
            alive = [p for p in self.pages.values() if not p.is_closed()]
            if len(alive) != 1:
                raise BrowserError(
                    "tab_required",
                    "Use open(url) or select_tab with an explicit tab_id.",
                    tabs=await self.tab_list(),
                )
            self.page = alive[0]
        return self.page

    async def clear_refs(self):
        handles = [entry[0] for entry in self.refs.values()]
        self.refs = {}
        if handles:
            await asyncio.gather(
                *(h.dispose() for h in handles), return_exceptions=True
            )

    async def observe(self, request):
        observed_start = time.perf_counter()
        page = await self.select_page()
        tab = self.tab_id(page)
        self.latest = None
        await self.clear_refs()
        self.frames = {}
        observation = dict(
            snapshot_id="web_" + uuid.uuid4().hex[:12],
            tab_id=tab,
            url=page.url,
            tabs=await self.tab_list(),
            observed_at=time.time(),
            frames=[],
            elements=[],
            text="",
            total_controls=0,
        )
        dialog = self.dialogs.get(tab)
        if dialog:
            observation["dialog"] = dict(
                type=dialog.type,
                message=dialog.message[:1000],
                default_value=dialog.default_value[:200],
            )
            observation["next_step"] = (
                "Resolve this dialog with dialog(accept=true/false) before interacting with the page."
            )
            self.latest = observation
            return observation
        records, errors = [], []
        collections = {}
        for index, frame in enumerate(page.frames):
            frame_id = "frame_" + str(index)
            self.frames[frame_id] = frame
            observation["frames"].append(
                dict(frame_id=frame_id, url=frame.url, name=frame.name)
            )
            remaining = min(max(0, 1200 - len(records)), max(1, 1200 // len(page.frames)))
            collection = None
            try:
                if index:
                    owner = await frame.frame_element()
                    try:
                        shown = await owner.is_visible()
                    finally:
                        await owner.dispose()
                    observation["frames"][-1]["visible"] = shown
                    if not shown:
                        continue
                collection = await frame.evaluate_handle(COLLECT, remaining)
                data = await collection.evaluate(
                    "(x) => ({records:x.records,total:x.total,text:x.text,ready_state:x.ready_state})"
                )
                # Keep a single collection handle per frame. Materialize
                # individual element handles only after filtering/pagination;
                # most pages display 40 of up to 1200 collected controls.
                collections[frame_id] = collection
                observation["total_controls"] += data["total"]
                observation["frames"][-1].update(
                    total_controls=data["total"],
                    text=data["text"][:1000],
                    ready_state=data["ready_state"],
                )
                if index == 0:
                    observation["text"] = data["text"][:4000]
                    observation["ready_state"] = data["ready_state"]
                for n, record in enumerate(data["records"]):
                    records.append(dict(_node_index=n, frame_id=frame_id, **record))
            except Exception as exc:
                errors.append(dict(frame_id=frame_id, error=str(exc)[:180]))
            finally:
                # Retained collections are disposed after the visible refs
                # have been created below.
                if collection and frame_id not in collections:
                    await collection.dispose()
        # Interleave frames before pagination so a large main document cannot
        # hide embedded forms. Within each frame put exposed controls first.
        from itertools import zip_longest
        groups = []
        for frame_id in self.frames:
            group = [r for r in records if r["frame_id"] == frame_id]
            group.sort(key=lambda r: (not r.get("in_viewport", True), bool(r.get("occluded")), not r.get("focused", False)))
            groups.append(group)
        records = [r for row in zip_longest(*groups) for r in row if r is not None]
        query = request.get("query", "").casefold()
        matches = [
            r
            for r in records
            if not query or query in json.dumps(r, ensure_ascii=False).casefold()
        ]
        offset, limit = request.get("offset", 0), request.get("limit", 40)
        visible_records = matches[offset : offset + limit]
        elements = []
        try:
            for record in visible_records:
                frame_id = record["frame_id"]
                source = collections[frame_id]
                handle = (await source.evaluate_handle(
                    "(x, index) => x.nodes[index]", record["_node_index"]
                )).as_element()
                if handle is None:
                    continue
                ref = observation["snapshot_id"] + "_" + str(len(elements))
                self.refs[ref] = (handle, page)
                elements.append({
                    **{k: v for k, v in record.items() if k != "_node_index"},
                    "ref": ref,
                })
        finally:
            await asyncio.gather(
                *(item.dispose() for item in collections.values()),
                return_exceptions=True,
            )
        observation.update(
            elements=elements,
            offset=offset,
            matched_count=len(matches),
            collected_count=len(records),
            truncated=len(records) < observation["total_controls"],
            next_offset=offset + limit if offset + limit < len(matches) else None,
        )
        if errors:
            observation["frame_errors"] = errors
        observation["observation_metrics"] = {
            "duration_ms": round((time.perf_counter() - observed_start) * 1000, 1),
            "element_handles": len(self.refs),
            "scanned_controls": observation["total_controls"],
            "returned_controls": len(elements),
        }
        self.latest = observation
        return observation

    async def target(self, spec, *, wait=False):
        page = await self.select_page()
        if "ref" in spec:
            entry = self.refs.get(spec["ref"])
            if entry is None or entry[1] != page:
                raise BrowserError(
                    "stale_reference",
                    "Use a ref from the latest observation, or a semantic target such as role+name.",
                )
            handle = entry[0]
            if not await handle.evaluate("(e) => e.isConnected"):
                raise BrowserError(
                    "detached_target",
                    "The observed element was replaced. Observe again or use a semantic target.",
                )
            return handle
        frame = (
            page.main_frame
            if not spec.get("frame_id")
            else self.frames.get(spec["frame_id"])
        )
        if frame is None or frame.is_detached():
            raise BrowserError(
                "frame_not_found", "Use a frame_id from the latest observation."
            )
        if "role" in spec:
            loc = frame.get_by_role(
                spec["role"],
                **({"name": spec["name"], "exact": True} if "name" in spec else {}),
            )
        elif "label" in spec:
            loc = frame.get_by_label(spec["label"], exact=True)
        elif "placeholder" in spec:
            loc = frame.get_by_placeholder(spec["placeholder"], exact=True)
        elif "text" in spec:
            loc = frame.get_by_text(spec["text"], exact=True)
        else:
            loc = frame.locator(spec["css"])
        count = await loc.count()
        if count > 1:
            candidates = await loc.evaluate_all(
                '(es) => es.slice(0,10).map(e=>({tag:e.tagName, text:(e.innerText || e.getAttribute("aria-label") || "").slice(0,150)}))'
            )
            raise BrowserError(
                "target_ambiguous",
                f"Target matches {count} elements. Choose an observed ref or a more specific target.",
                candidates=candidates,
            )
        # Locator actions wait for a missing node to appear, then enforce strict uniqueness.
        return loc

    def workspace_file(self, path, *, exists=False):
        candidate = (self.workspace / path).resolve()
        if not candidate.is_relative_to(self.workspace):
            raise BrowserError(
                "workspace_boundary",
                "Browser file transfer paths must remain inside the workspace.",
            )
        if exists and not candidate.is_file():
            raise BrowserError(
                "file_not_found", "Upload path is not a regular workspace file."
            )
        return candidate

    async def perform(self, request):
        action = request["action"]
        self.delivery = "not_sent"
        if action == "doctor":
            executable = source = error = None
            try:
                bundled = self.pw.chromium.executable_path if self.pw else None
                if bundled is None and importlib.util.find_spec("playwright") is not None:
                    from playwright.async_api import async_playwright
                    async with async_playwright() as probe:
                        bundled = probe.chromium.executable_path
                executable, source = resolve_executable(self.config, bundled)
            except BrowserError as exc:
                error = str(exc)
            return dict(
                executable_path=executable,
                executable_source=source,
                executable_error=error,
                playwright_installed=importlib.util.find_spec("playwright") is not None,
                running=self.context is not None,
                headed=not self.config.get("headless", False),
                mode="cdp" if self.config.get("cdp_url") else "managed",
                setup="python -m pip install -e '.[browser]' && python -m playwright install chromium",
            )
        if action == "close":
            await self.close()
            return dict(closed=True, detached=self.attached)
        if action == "open":
            await self.start()
            if request.get("new_tab") or self.page is None or self.page.is_closed():
                self.page = await self.context.new_page()
                self.register_page(self.page)
        elif self.context is None:
            if action == "tabs" and self.config.get("cdp_url"):
                await self.start()
            elif action == "tabs":
                return dict(
                    tabs=[],
                    next_step="Use open with a URL to launch the managed browser.",
                )
            else:
                raise BrowserError(
                    "browser_not_started",
                    "Use open(url) to launch the browser and navigate in one call.",
                )
        if action == "tabs":
            return dict(tabs=await self.tab_list())
        if action == "downloads":
            return dict(
                downloads=[
                    dict(download_id=k, tab_id=v[0], filename=v[1].suggested_filename)
                    for k, v in self.downloads.items()
                ]
            )
        page = await self.select_page(request.get("tab_id"))
        timeout = request.get("timeout_ms", 10000)
        page.set_default_timeout(timeout)
        page.set_default_navigation_timeout(timeout)
        extra = {}
        read_only = action in {"observe", "select_tab", "read", "screenshot", "wait"}
        if action in {"observe", "select_tab"}:
            if action == "select_tab":
                await page.bring_to_front()
            return dict(observation=await self.observe(request))
        if self.dialogs.get(self.tab_id(page)) and action != "dialog":
            raise BrowserError(
                "dialog_pending",
                "Resolve the pending JavaScript dialog explicitly with dialog(accept=true/false).",
            )
        target = await self.target(request["target"]) if "target" in request else None
        # Resolve every target/path before dispatch; never retry an uncertain input.
        destination = (
            await self.target(request["destination"])
            if "destination" in request
            else None
        )
        uploads = [
            str(self.workspace_file(p, exists=True)) for p in request.get("paths", [])
        ]
        download_path = (
            self.workspace_file(request["path"]) if action == "save_download" else None
        )
        if action == "save_download" and (
            request["download_id"] not in self.downloads or download_path.exists()
        ):
            raise BrowserError(
                "invalid_download",
                "Choose an existing download_id and a new destination path; files are never overwritten.",
            )
        if not read_only:
            self.latest = None
            self.delivery = "unknown"
        if action == "open":
            response = await page.goto(request["url"], wait_until="domcontentloaded")
            extra["http_status"] = response.status if response else None
        elif action in {"back", "forward", "reload"}:
            method = {
                "back": page.go_back,
                "forward": page.go_forward,
                "reload": page.reload,
            }[action]
            response = await method(wait_until="domcontentloaded")
            extra["http_status"] = response.status if response else None
        elif action == "close_tab":
            await page.close()
            self.page = None
            await self.clear_refs()
            self.latest = None
            self.delivery = "completed"
            return dict(tabs=await self.tab_list())
        elif action in {"click", "hover", "fill", "press", "select", "check", "upload"}:
            if action == "click":
                await target.click(
                    timeout=timeout,
                    no_wait_after=True,
                    button=request.get("button", "left"),
                    click_count=request.get("click_count", 1),
                    **(
                        {"position": request["position"]}
                        if "position" in request
                        else {}
                    ),
                )
            elif action == "hover":
                await target.hover(timeout=timeout)
            elif action == "fill":
                await target.fill(request["text"], timeout=timeout)
            elif action == "press":
                if target:
                    await target.press(
                        request["text"], timeout=timeout, no_wait_after=True
                    )
                else:
                    await page.keyboard.press(request["text"])
            elif action == "select":
                await target.select_option(value=request["values"], timeout=timeout)
            elif action == "check":
                await target.set_checked(request["checked"], timeout=timeout)
            elif action == "upload":
                await target.set_input_files(uploads, timeout=timeout)
        elif action == "drag":
            # Hover auto-waits and scrolls the source before pressing. Scroll the
            # destination only after pickup, so a distant target cannot move the
            # source out of the viewport before the drag begins.
            await target.hover(timeout=timeout)
            await page.mouse.down()
            try:
                await destination.scroll_into_view_if_needed(timeout=timeout)
                end = await destination.bounding_box()
                if not end:
                    raise BrowserError(
                        "target_not_visible", "Drag destination is not visible."
                    )
                await page.mouse.move(
                    end["x"] + end["width"] / 2, end["y"] + end["height"] / 2, steps=12
                )
            finally:
                await page.mouse.up()
        elif action == "scroll":
            if target:
                await target.hover(timeout=timeout)
            await page.mouse.wheel(
                request.get("delta_x", 0), request.get("delta_y", 600)
            )
        elif action == "dialog":
            dialog = self.dialogs.get(self.tab_id(page))
            if dialog is None:
                raise BrowserError("no_dialog", "No pending dialog on this tab.")
            if request["accept"]:
                await dialog.accept(request.get("text", ""))
            else:
                await dialog.dismiss()
            self.dialogs.pop(self.tab_id(page), None)
        elif action == "save_download":
            download_path.parent.mkdir(parents=True, exist_ok=True)
            download = self.downloads[request["download_id"]][1]
            # Reserve the destination exclusively before save_as to avoid overwrites.
            with download_path.open("xb"):
                pass
            try:
                await download.save_as(str(download_path))
            except BaseException:
                download_path.unlink(missing_ok=True)
                raise
            extra["saved_path"] = str(download_path)
        elif action == "read":
            text = (
                await target.inner_text(timeout=timeout)
                if target
                else await page.locator("body").inner_text(timeout=timeout)
            )
            offset, limit = request.get("offset", 0), request.get("limit", 40) * 100
            extra.update(
                text=text[offset : offset + limit],
                offset=offset,
                total_characters=len(text),
                next_offset=offset + limit if offset + limit < len(text) else None,
            )
        elif action == "screenshot":
            self._private_root()
            image = self.root / ("screen_" + uuid.uuid4().hex + ".png")
            await page.screenshot(path=str(image), timeout=timeout, scale="css")
            extra["image_ref"] = str(image)
            extra["screenshot_coordinates"] = (
                "One image pixel per CSS viewport pixel. click.position is relative to the target padding box, not the viewport."
            )
        elif action == "wait":
            state = request.get(
                "state",
                "visible" if target or "text" in request else "domcontentloaded",
            )
            if "text" in request and not target:
                target = page.get_by_text(request["text"], exact=True)
            if target:
                if hasattr(target, "wait_for"):
                    await target.wait_for(state=state, timeout=timeout)
                else:
                    if state not in {"visible", "hidden"}:
                        raise BrowserError(
                            "invalid_arguments",
                            "Element refs support wait visible/hidden; use semantic locators for attached/detached.",
                        )
                    await target.wait_for_element_state(state, timeout=timeout)
            else:
                if state not in {"load", "domcontentloaded"}:
                    raise BrowserError(
                        "invalid_arguments",
                        "Page wait requires load/domcontentloaded; element states require target.",
                    )
                await page.wait_for_load_state(state, timeout=timeout)
            extra["verification"] = dict(
                status="verified",
                condition={
                    k: v for k, v in request.items() if k in {"state", "target", "text"}
                },
            )
        if not read_only:
            self.delivery = "completed"
        # Navigation can replace documents between frames. An observation failure
        # must not turn a delivered operation into an instruction to repeat it.
        try:
            observation = await self.observe(request)
            if extra.get("image_ref"):
                observation["image_ref"] = extra.pop("image_ref")
            if not read_only:
                observation["last_action"] = dict(action=action, delivery=self.delivery)
            extra["observation"] = observation
        except Exception as exc:
            self.latest = None
            await self.clear_refs()
            extra["observation_error"] = str(exc)[:500]
            extra["next_step"] = (
                "Observe the current page. The operation is not automatically retried."
            )
        return extra

    async def close(self):
        self.latest = None
        await self.clear_refs()
        try:
            if self.context and not self.attached:
                await self.context.close()
        finally:
            if self.pw:
                await self.pw.stop()
            self.pw = self.browser = self.context = self.page = None
            self.pages.clear()
            self.frames.clear()
            self.dialogs.clear()
            self.downloads.clear()
