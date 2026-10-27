"""Workspace runtime overrides preserve future memory configuration fields."""

import pytest

from src.studio.state import StudioStore, default_config, validate_config


def test_studio_runtime_preserves_memory_options_without_mutating_profile(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    store = StudioStore(tmp_path / "workspace")
    config = default_config()
    config["memory"] = {"database": "old.sqlite3", "retrieval": {"enabled": False}}
    runtime = store.runtime_config(config)
    assert runtime["memory"] == {
        "database": str(store.root / "history.sqlite3"),
        "retrieval": {"enabled": False},
    }
    runtime["memory"]["retrieval"]["enabled"] = True
    assert config["memory"] == {
        "database": "old.sqlite3",
        "retrieval": {"enabled": False},
    }


def test_studio_rejects_nonobject_memory_config_before_starting_agent():
    config = default_config()
    config["memory"] = "invalid"
    with pytest.raises(ValueError, match="memory must be an object"):
        validate_config(config)
