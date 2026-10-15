# SmolLM3 local provider

Daena includes an optional local provider named smollm3 for text-only SmolLM3 GGUF models. It uses llama-cpp-python and translates SmolLM3's documented XML tool calls into Daena's normal LLMResult.tool_calls format. Daena's existing dispatcher remains responsible for validating and executing tools.

## Install

Install the optional runtime dependency in the same Python environment that runs Daena:

```bash
python -m pip install llama-cpp-python
```

For CUDA acceleration, install a llama-cpp-python build compatible with the host's CUDA/toolchain. The default provider configuration is CPU-only.

## Configure

Set llm.provider to smollm3 and point model_name at the actual GGUF file:

```json
{
  "llm": {
    "provider": "smollm3",
    "provider_config": {
      "model_name": "/absolute/path/to/SmolLM3-3B-q4.gguf",
      "generation_config": {
        "temperature": 0.3,
        "top_p": 0.95,
        "max_tokens": 512,
        "n_ctx": 4096,
        "n_threads": 4,
        "n_batch": 128,
        "n_gpu_layers": 0,
        "think": false
      }
    }
  }
}
```

Alternatively, set DAENA_SMOLLM3_MODEL to the GGUF path. A configured model_name takes precedence.

## Notes and limitations

- The model is loaded lazily on the first request.
- The provider is text-only; image input raises a clear error.
- Tool calls are parsed from <tool_call>{...}</tool_call>; malformed calls are ignored with a warning rather than executed.
- The provider does not execute tools itself. Tool execution, policy checks, workspace boundaries, and result normalization remain in Daena.
- Inference is currently non-token-streaming. When a stream callback is supplied, completion events are emitted after inference.
- GGUF variants can carry different templates or tokenizer metadata. Validate the exact quantized file with provider tests and a small end-to-end tool call before relying on it for unattended tasks.
