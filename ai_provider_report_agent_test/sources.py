"""Structured data for the three providers.

The data was collected from the official documentation pages that were
searched for during the project creation.  The URLs are included so the
report can reference the source.
"""

from .models import Provider, ToolCallFormat, ToolResultFormat

providers = [
    Provider(
        name="Ollama",
        doc_url="https://docs.ollama.com/capabilities/tool-calling",
        tool_call_format=ToolCallFormat(
            representation="tool_calls array with objects containing index, type 'function', function name and arguments as JSON object",
            id_field="index",
            arguments="json object",  # correct per official docs
        ),
        tool_result_format=ToolResultFormat(
            role="tool",
            content="JSON object",
            tool_call_id_field="index",
        ),
    ),
    Provider(
        name="OpenAI",
        doc_url="https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling",
        tool_call_format=ToolCallFormat(
            representation="tool_calls array with objects containing id, type 'function', function name and arguments as JSON string",
            id_field="id",
            arguments="json string",
        ),
        tool_result_format=ToolResultFormat(
            role="tool",
            content="JSON string",
            tool_call_id_field="tool_call_id",
        ),
    ),
    Provider(
        name="OpenRouter",
        doc_url="https://openrouter.ai/docs/guides/features/tool-calling",
        tool_call_format=ToolCallFormat(
            representation="OpenAI compatible: tool_calls array with objects containing id, type 'function', function name and arguments as JSON string",
            id_field="id",
            arguments="json string",
        ),
        tool_result_format=ToolResultFormat(
            role="tool",
            content="JSON string",
            tool_call_id_field="tool_call_id",
        ),
    ),
]

