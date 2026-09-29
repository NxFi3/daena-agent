from types import SimpleNamespace
from unittest.mock import patch

from src.models.ToolResult import ToolResult
from src.tools.builtin.web.httpclient import HttpResponse
from src.tools.builtin.web.webfetch import WebFetch


def test_pdf_content_type_is_supported():
    assert WebFetch._classify("application/pdf", b"%PDF-1.7") == "pdf"
    assert WebFetch._classify("application/octet-stream", b"%PDF-1.7") == "pdf"


def test_pdf_is_extracted_with_pypdf():
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        is_encrypted = False

        def __init__(self, *_args, **_kwargs):
            self.pages = [
                FakePage("Hello from PDF"),
                FakePage("Second page"),
            ]
            self.metadata = SimpleNamespace(title="PDF title")

    response = HttpResponse(
        url="https://example.com/paper.pdf",
        status=200,
        content_type="application/pdf",
        charset=None,
        body=b"%PDF-1.7 fake",
        truncated=False,
    )

    with patch(
        "src.tools.builtin.web.webfetch.http_get",
        return_value=response,
    ), patch(
        "pypdf.PdfReader",
        FakeReader,
    ):
        result = WebFetch().execute(
            url=response.url,
            max_chars=6000,
        )

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.content["extractor"] == "pypdf"
    assert result.content["title"] == "PDF title"
    assert "Hello from PDF" in result.content["content"]
    assert "[Page 2]" in result.content["content"]


def test_pdf_without_text_layer_reports_vision_fallback():
    class FakeReader:
        is_encrypted = False

        def __init__(self, *_args, **_kwargs):
            self.pages = [SimpleNamespace(extract_text=lambda: "")]
            self.metadata = None

    response = HttpResponse(
        url="https://example.com/scanned.pdf",
        status=200,
        content_type="application/pdf",
        charset=None,
        body=b"%PDF-1.7 fake",
        truncated=False,
    )

    with patch(
        "src.tools.builtin.web.webfetch.http_get",
        return_value=response,
    ), patch(
        "pypdf.PdfReader",
        FakeReader,
    ):
        result = WebFetch().execute(url=response.url)

    assert result.success is True
    assert "vision/OCR" in result.content["note"]
