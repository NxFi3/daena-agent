from src.tools.builtin.search.tool import Search


def test_search_finds_matches_and_reports_line_numbers(tmp_path):
    (tmp_path / "a.py").write_text(
        "def first():\n    return 1\n\ndef target():\n    return 2\n",
        encoding="utf-8",
    )
    (tmp_path / "b.txt").write_text(
        "nothing here\n",
        encoding="utf-8",
    )

    result = Search().execute(
        query="target",
        path=str(tmp_path),
        max_results=10,
        context_lines=1,
    )

    assert result.success is True
    assert result.content["result_count"] == 1
    assert "a.py:4" in result.content["content"]
    assert "def target()" in result.content["content"]


def test_search_accepts_invalid_regex_as_literal_text(tmp_path):
    (tmp_path / "app.js").write_text(
        "const value = foo[bar];\n",
        encoding="utf-8",
    )

    result = Search().execute(
        query="foo[bar]",
        path=str(tmp_path),
        max_results=10,
    )

    assert result.success is True
    assert result.content["result_count"] == 1


def test_search_skips_common_generated_directories(tmp_path):
    generated = tmp_path / "node_modules"
    generated.mkdir()
    (generated / "package.js").write_text("SECRET = true\n", encoding="utf-8")
    (tmp_path / "src.py").write_text("SECRET = false\n", encoding="utf-8")

    result = Search().execute(
        query="SECRET",
        path=str(tmp_path),
        max_results=10,
    )

    assert result.success is True
    assert result.content["result_count"] == 1
    assert "src.py" in result.content["content"]
    assert "node_modules" not in result.content["content"]
