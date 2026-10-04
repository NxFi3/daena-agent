from src.tools.builtin.readfile.tool import ReadFile


def _numbered_file(tmp_path, count=10):
    target = tmp_path / "sample.txt"
    target.write_text(
        "".join(f"line{i}\n" for i in range(1, count + 1)),
        encoding="utf-8",
    )
    return target


def test_valid_range_is_respected(tmp_path):
    target = _numbered_file(tmp_path)

    result = ReadFile().execute(
        file_path=str(target),
        start_line=2,
        end_line=4,
    )

    assert result.success is True
    assert result.content["start_line"] == 2
    assert result.content["end_line"] == 4
    assert "line2" in result.content["content"]
    assert "line4" in result.content["content"]
    assert "line5" not in result.content["content"]
    assert "note" not in result.content


def test_reversed_range_reads_to_end_instead_of_failing(tmp_path):
    target = _numbered_file(tmp_path)

    result = ReadFile().execute(
        file_path=str(target),
        start_line=8,
        end_line=3,
    )

    assert result.success is True
    assert result.content["start_line"] == 8
    assert result.content["end_line"] == 10
    assert "line8" in result.content["content"]
    assert "line10" in result.content["content"]
    assert "line7" not in result.content["content"]
    assert "note" in result.content


def test_start_beyond_end_of_file_still_reports_total_lines(tmp_path):
    target = _numbered_file(tmp_path, count=3)

    result = ReadFile().execute(
        file_path=str(target),
        start_line=50,
    )

    assert result.success is False
    assert result.content["total_lines"] == 3
