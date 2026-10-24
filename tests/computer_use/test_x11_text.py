from src.tools.builtin.computer_use.x11_text import chunks, keysym


def test_unicode_chunks_preserve_text_without_reusing_a_missing_code_within_chunk():
    text = "Prefix سلام Ελληνικά 中文 日本語 😀 End"
    missing = {keysym(c) for c in text if ord(c) > 127}
    parts = list(chunks(text, missing, 3))
    assert "".join(value for value, _ in parts) == text
    assert len(parts) > 1
    for value, symbols in parts:
        assert len(symbols) <= 3
        assert {keysym(c) for c in value if keysym(c) in missing} == symbols
