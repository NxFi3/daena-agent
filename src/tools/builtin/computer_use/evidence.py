"""Keep desktop receipts valid JSON inside the host's 8 KB tool-message limit."""

from copy import deepcopy
import json


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False))


def element_page(elements, *, offset=0, limit=40, budget=3500):
    shown = []
    for element in elements[offset : offset + limit]:
        row = element.public()
        if shown and encoded_size([*shown, row]) > budget:
            break
        shown.append(row)
    end = offset + len(shown)
    return {
        "elements": shown,
        "matched_count": len(elements),
        "shown_count": len(shown),
        "offset": offset,
        "next_offset": end if end < len(elements) else None,
        "truncated": end < len(elements) or offset > 0,
    }


def _grounding(content):
    """Fallback receipt that retains coordinates/image identity before diagnostics."""
    scalar_keys = {
        "action",
        "snapshot_id",
        "dispatch_status",
        "delivery_mode",
        "image_ref",
        "width",
        "height",
        "coordinate_frame",
        "origin",
        "scale_x",
        "scale_y",
        "mapping",
        "input_allowed",
        "snapshot_consumed",
        "next_snapshot_id",
        "snapshot_id_used",
        "success",
        "mode",
    }
    result = {
        key: deepcopy(value) for key, value in content.items() if key in scalar_keys
    }
    window = content.get("window")
    if isinstance(window, dict):
        result["window"] = {
            key: value
            for key, value in window.items()
            if key in {"pid", "window_id", "bounds", "active"}
        }
    for key in ("verification", "visual_delta", "accessibility_delta"):
        value = content.get(key)
        if isinstance(value, dict):
            result[key] = {
                name: deepcopy(item)
                for name, item in value.items()
                if name
                in {
                    "status",
                    "kind",
                    "changed_fraction",
                    "changed_bounds",
                    "before_size",
                    "after_size",
                    "compared_controls",
                    "changed_controls",
                    "newly_observed",
                    "no_longer_observed",
                    "ambiguous_identities",
                }
            }
    for key in ("error", "recovery"):
        value = content.get(key)
        if isinstance(value, str):
            result[key] = value[:500]
        elif isinstance(value, dict):
            result[key] = {
                name: str(item)[:500]
                for name, item in value.items()
                if name in {"code", "message"}
            }
    capabilities = content.get("capabilities")
    if isinstance(capabilities, dict):
        result["capabilities"] = {
            key: deepcopy(capabilities[key])
            for key in (
                "backend",
                "preferred_delivery_mode",
                "delivery_modes",
                "drag_path",
                "pointer_move",
            )
            if key in capabilities
        }
    target = content.get("target")
    if target is not None:
        result["target"] = (
            deepcopy(target)
            if encoded_size(target) <= 500
            else {"details_omitted": True}
        )
    for key in ("observation", "last_input"):
        if isinstance(content.get(key), dict):
            result[key] = _grounding(content[key])
    if "elements" in content:
        result.update(
            elements=[],
            shown_count=0,
            next_offset=content.get("offset", 0),
            truncated=True,
        )
    result["details_omitted"] = True
    return result


def compact_evidence(content, budget=6000):
    result = deepcopy(content)
    # Backend receipts and duplicate raw-image hashes are kept in content/trace.
    result.pop("backend_receipt", None)

    def bound(value, key=None):
        if isinstance(value, dict):
            return {k: bound(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [bound(v) for v in value[:100]]
        # A shortened image path is a different (usually nonexistent) file.
        if isinstance(value, str) and len(value) > 900 and key != "image_ref":
            return value[:900] + "… [truncated]"
        return value

    result = bound(result)
    while encoded_size(result) > budget:
        choices = []

        def collect(value):
            if not isinstance(value, dict):
                return
            for key, item in value.items():
                if (
                    isinstance(item, list)
                    and item
                    and key
                    not in {
                        "coordinate",
                        "start",
                        "end",
                        "from_coordinate",
                        "to_coordinate",
                        "origin",
                        "range",
                        "changed_bounds",
                        "before_size",
                        "after_size",
                        "attached_image_dimensions",
                        "native_input_size",
                        "attached_size",
                        "native_origin",
                        "image_scale",
                    }
                ):
                    choices.append((encoded_size(item), value, key))
                elif isinstance(item, dict):
                    collect(item)

        collect(result)
        if not choices:
            core = _grounding(result)
            if encoded_size(core) <= budget:
                return core
            # At very small budgets, keep the frame's identity and mapping first.
            # Explicitly signal an unavailable image rather than sending a broken path.
            for key in (
                "recovery",
                "accessibility_delta",
                "visual_delta",
                "verification",
                "last_input",
                "observation",
                "error",
            ):
                core.pop(key, None)
                if encoded_size(core) <= budget:
                    return core
            if core.get("image_ref") and encoded_size(core) > budget:
                core.pop("image_ref")
                core["image_unavailable"] = (
                    "Image reference exceeded the evidence budget; it was not truncated."
                )
            return core
        _, parent, key = max(choices, key=lambda row: row[0])
        parent[key].pop()
        if key == "elements":
            parent.update(shown_count=len(parent[key]), truncated=True)
            parent["next_offset"] = parent.get("offset", 0) + len(parent[key])
        else:
            parent[key + "_truncated"] = True
    return result
