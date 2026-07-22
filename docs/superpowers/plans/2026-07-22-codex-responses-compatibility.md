# Codex Responses Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pinned SGLang RunPod worker preserve valid Responses API SSE frames and optionally disable Qwen thinking only inside the Responses adapter.

**Architecture:** Replace line-at-a-time SSE rewriting with an incremental, frame-aware proxy. Launch SGLang through a small worker-owned wrapper that patches only the Responses module's internal `ChatCompletionRequest` factory when `RESPONSES_DISABLE_THINKING` is explicitly enabled.

**Tech Stack:** Python 3, `unittest`, aiohttp streaming, RunPod Python SDK, SGLang v0.5.15.post1, Docker.

## Global Constraints

- Keep SGLang pinned exactly to `v0.5.15.post1`.
- Keep `RESPONSES_DISABLE_THINKING` disabled by default.
- Apply non-thinking mode only to the SGLang Responses adapter, never direct Chat Completions or native generation.
- Do not mutate user prompts or append `/no_think`.
- Preserve RunPod worker count, scaling, idle settings, and model caching settings.
- Do not edit installed SGLang package files.
- Do not upgrade the CUDA image or dependency versions.
- Stage and commit only files changed for this compatibility work; preserve unrelated working-tree changes.

---

### Task 1: Preserve Complete SSE Frames

**Files:**
- Modify: `utils.py`
- Modify: `tests/test_utils.py`

**Interfaces:**
- Consumes: `response.content`, an async iterator of arbitrarily split UTF-8 byte chunks.
- Produces: `format_sse_frame(lines: list[str]) -> str` and `async_process_stream(response)`.
- Preserves: `async_process_response(response, is_stream, route)` and existing structured errors.

- [ ] **Step 1: Write failing Responses SSE tests**

Add these tests to `ResponseProcessingTests`:

~~~python
def test_stream_response_preserves_responses_event_frame(self):
    response = FakeResponse(
        lines=[
            b"event: response.created\n",
            b'data: {"type":"response.created"}\n',
            b"\n",
        ]
    )
    result = asyncio.run(collect_response(response, is_stream=True))
    self.assertEqual(
        result,
        [
            "event: response.created\n"
            'data: {"type": "response.created"}\n\n'
        ],
    )

def test_stream_response_handles_coalesced_and_fragmented_frames(self):
    response = FakeResponse(
        lines=[
            b"event: response.output_text.delta\nda",
            b'ta: {"delta":"h',
            b'i"}\n\nevent: response.completed\n',
            b'data: {"type":"response.completed"}',
        ]
    )
    result = asyncio.run(collect_response(response, is_stream=True))
    self.assertEqual(
        result,
        [
            "event: response.output_text.delta\n"
            'data: {"delta": "hi"}\n\n',
            "event: response.completed\n"
            'data: {"type": "response.completed"}\n\n',
        ],
    )

def test_stream_response_preserves_non_json_data(self):
    response = FakeResponse(lines=[b"data: plain text\n\n"])
    result = asyncio.run(collect_response(response, is_stream=True))
    self.assertEqual(result, ["data: plain text\n\n"])
~~~

Update the existing Chat Completions stream fixture to include blank-line frame delimiters:

~~~python
response = FakeResponse(
    lines=[
        b'data: {"token":"hello"}\n\n',
        b"data: [DONE]\n\n",
    ]
)
~~~

- [ ] **Step 2: Run focused tests and verify the bug**

Run:

~~~bash
python3 -m unittest tests.test_utils.ResponseProcessingTests.test_stream_response_preserves_responses_event_frame tests.test_utils.ResponseProcessingTests.test_stream_response_handles_coalesced_and_fragmented_frames tests.test_utils.ResponseProcessingTests.test_stream_response_preserves_non_json_data -v
~~~

Expected: the Responses event tests fail because `event:` is rewritten as `data:` and lines are emitted independently.

- [ ] **Step 3: Implement incremental SSE frame parsing**

Replace the streaming helpers in `utils.py` with:

~~~python
import codecs
import json


def _format_sse_data(value: str) -> str:
    if value.startswith(" "):
        value = value[1:]
    if value == "[DONE]":
        return "data: [DONE]"
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return f"data: {value}" if value else "data:"
    return f"data: {json.dumps(data)}"


def format_sse_frame(lines: list[str]) -> str:
    """Format one complete SSE frame without changing its field types."""
    formatted = []
    for line in lines:
        field, separator, value = line.partition(":")
        if field == "data":
            formatted.append(_format_sse_data(value if separator else ""))
        else:
            formatted.append(line)
    return "\n".join(formatted) + "\n\n"


def format_sse_chunk(chunk: str) -> str:
    """Backward-compatible formatter for a single-field SSE frame."""
    return format_sse_frame([chunk.strip()])


async def async_process_stream(response):
    """Yield complete SSE frames from arbitrarily split upstream bytes."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    text_buffer = ""
    frame_lines = []

    async for raw_chunk in response.content:
        text_buffer += decoder.decode(raw_chunk)
        while "\n" in text_buffer:
            line, text_buffer = text_buffer.split("\n", 1)
            if line.endswith("\r"):
                line = line[:-1]
            if line:
                frame_lines.append(line)
            elif frame_lines:
                yield format_sse_frame(frame_lines)
                frame_lines = []

    text_buffer += decoder.decode(b"", final=True)
    if text_buffer:
        if text_buffer.endswith("\r"):
            text_buffer = text_buffer[:-1]
        frame_lines.append(text_buffer)
    if frame_lines:
        yield format_sse_frame(frame_lines)
~~~

Leave `async_process_response` unchanged apart from its existing call to `async_process_stream`.

- [ ] **Step 4: Run all response tests**

Run: `python3 -m unittest tests.test_utils -v`

Expected: all response-processing tests pass.

- [ ] **Step 5: Commit the SSE fix**

~~~bash
git add utils.py tests/test_utils.py
git commit -m "fix: preserve responses SSE frames"
~~~

---

### Task 2: Add the Opt-in Responses Thinking Shim

**Files:**
- Create: `sglang_launcher.py`
- Create: `tests/test_sglang_launcher.py`

**Interfaces:**
- Produces: `responses_thinking_disabled(env=None) -> bool`.
- Produces: `wrap_chat_completion_request(factory) -> callable`.
- Produces: `install_responses_compatibility(module=None) -> None`.
- Produces: `main(env=None) -> None`.

- [ ] **Step 1: Write failing shim tests**

Create `tests/test_sglang_launcher.py`:

~~~python
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sglang_launcher


class SGLangLauncherTests(unittest.TestCase):
    def test_responses_thinking_is_disabled_only_for_true_values(self):
        for value in ("true", "TRUE", "1", "yes", "YES"):
            with self.subTest(value=value):
                self.assertTrue(
                    sglang_launcher.responses_thinking_disabled(
                        {"RESPONSES_DISABLE_THINKING": value}
                    )
                )
        for value in ("", "false", "0", "no"):
            with self.subTest(value=value):
                self.assertFalse(
                    sglang_launcher.responses_thinking_disabled(
                        {"RESPONSES_DISABLE_THINKING": value}
                    )
                )

    def test_wrapper_injects_false_without_mutating_existing_kwargs(self):
        def factory(**kwargs):
            return kwargs

        existing = {"preserve_thinking": True}
        wrapped = sglang_launcher.wrap_chat_completion_request(factory)
        result = wrapped(model="model", chat_template_kwargs=existing)

        self.assertEqual(existing, {"preserve_thinking": True})
        self.assertEqual(
            result["chat_template_kwargs"],
            {"preserve_thinking": True, "enable_thinking": False},
        )

    def test_wrapper_preserves_explicit_future_enable_thinking_value(self):
        factory = Mock(side_effect=lambda **kwargs: kwargs)
        wrapped = sglang_launcher.wrap_chat_completion_request(factory)
        result = wrapped(chat_template_kwargs={"enable_thinking": True})
        self.assertEqual(
            result["chat_template_kwargs"], {"enable_thinking": True}
        )

    def test_install_patches_only_supplied_module_reference(self):
        original = Mock(side_effect=lambda **kwargs: kwargs)
        module = SimpleNamespace(ChatCompletionRequest=original)

        sglang_launcher.install_responses_compatibility(module)
        result = module.ChatCompletionRequest(model="model")

        self.assertIsNot(module.ChatCompletionRequest, original)
        self.assertEqual(
            result["chat_template_kwargs"], {"enable_thinking": False}
        )

    @patch("sglang_launcher.runpy.run_module")
    @patch("sglang_launcher.install_responses_compatibility")
    def test_main_installs_shim_only_when_enabled(self, install, run_module):
        sglang_launcher.main({"RESPONSES_DISABLE_THINKING": "true"})
        install.assert_called_once_with()
        run_module.assert_called_once_with(
            "sglang.launch_server", run_name="__main__"
        )

    @patch("sglang_launcher.runpy.run_module")
    @patch("sglang_launcher.install_responses_compatibility")
    def test_main_skips_shim_by_default(self, install, run_module):
        sglang_launcher.main({})
        install.assert_not_called()
        run_module.assert_called_once_with(
            "sglang.launch_server", run_name="__main__"
        )


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Verify the new module is missing**

Run: `python3 -m unittest tests.test_sglang_launcher -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'sglang_launcher'`.

- [ ] **Step 3: Implement the worker-owned launcher**

Create `sglang_launcher.py`:

~~~python
import os
import runpy


TRUE_VALUES = {"true", "1", "yes"}


def responses_thinking_disabled(env=None):
    """Return whether the opt-in Responses compatibility mode is enabled."""
    env = os.environ if env is None else env
    return env.get("RESPONSES_DISABLE_THINKING", "").lower() in TRUE_VALUES


def wrap_chat_completion_request(factory):
    """Inject Qwen's non-thinking argument into Responses requests."""
    def compatible_factory(*args, **kwargs):
        template_kwargs = dict(kwargs.get("chat_template_kwargs") or {})
        template_kwargs.setdefault("enable_thinking", False)
        kwargs["chat_template_kwargs"] = template_kwargs
        return factory(*args, **kwargs)

    compatible_factory._responses_disable_thinking = True
    return compatible_factory


def install_responses_compatibility(module=None):
    """Patch only SGLang's Responses-local request factory reference."""
    if module is None:
        from sglang.srt.entrypoints.openai import serving_responses as module

    current_factory = module.ChatCompletionRequest
    if getattr(current_factory, "_responses_disable_thinking", False):
        return
    module.ChatCompletionRequest = wrap_chat_completion_request(current_factory)


def main(env=None):
    if responses_thinking_disabled(env):
        install_responses_compatibility()
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
~~~

- [ ] **Step 4: Run shim tests**

Run: `python3 -m unittest tests.test_sglang_launcher -v`

Expected: all shim tests pass without importing real SGLang in the local test process.

- [ ] **Step 5: Commit the shim**

~~~bash
git add sglang_launcher.py tests/test_sglang_launcher.py
git commit -m "feat: add responses thinking compatibility shim"
~~~

---

### Task 3: Integrate the Launcher and Publish Configuration

**Files:**
- Modify: `engine.py`
- Modify: `tests/test_engine.py`
- Modify: `Dockerfile`
- Modify: `.runpod/hub.json`
- Modify: `README.md`
- Modify: `tests/test_configuration.py`

**Interfaces:**
- Produces: an engine command beginning with `python3 <absolute path>/sglang_launcher.py`, followed by unchanged SGLang flags.
- Publishes: boolean `RESPONSES_DISABLE_THINKING`, default `false`.

- [ ] **Step 1: Write failing integration tests**

Add to `tests/test_engine.py`:

~~~python
def test_build_command_starts_worker_owned_launcher(self):
    engine = SGlangEngine(
        env={
            "MODEL_NAME": "Qwen/Qwen3-8B",
            "HOST": "127.0.0.1",
            "PORT": "31000",
        }
    )
    command = engine.build_command()
    expected_launcher = str(
        Path(__file__).resolve().parents[1] / "sglang_launcher.py"
    )
    self.assertEqual(
        command[:6],
        [
            "python3",
            expected_launcher,
            "--host",
            "127.0.0.1",
            "--port",
            "31000",
        ],
    )
~~~

Import `Path` from `pathlib` and update the existing command-prefix assertion to expect the launcher path.

Add to `tests/test_configuration.py`:

~~~python
def test_image_includes_responses_launcher_and_documents_opt_in(self):
    dockerfile = (ROOT / "Dockerfile").read_text()
    readme = (ROOT / "README.md").read_text()
    hub = json.loads((ROOT / ".runpod" / "hub.json").read_text())
    env = {entry["key"]: entry for entry in hub["config"]["env"]}

    self.assertIn("sglang_launcher.py", dockerfile)
    self.assertIn("RESPONSES_DISABLE_THINKING", readme)
    self.assertIn("RESPONSES_DISABLE_THINKING", env)
    self.assertEqual(
        env["RESPONSES_DISABLE_THINKING"]["input"]["default"], False
    )
    self.assertEqual(
        env["RESPONSES_DISABLE_THINKING"]["input"]["type"], "boolean"
    )
~~~

- [ ] **Step 2: Verify integration tests fail**

Run: `python3 -m unittest tests.test_engine tests.test_configuration -v`

Expected: launcher-prefix and configuration assertions fail.

- [ ] **Step 3: Route engine startup through the launcher**

In `engine.py`, add:

~~~python
from pathlib import Path

SGLANG_LAUNCHER = str(Path(__file__).resolve().with_name("sglang_launcher.py"))
~~~

Replace only the command prefix in `build_command()`:

~~~python
command = [
    "python3",
    SGLANG_LAUNCHER,
    "--host",
    self.host,
    "--port",
    str(self.port),
]
~~~

- [ ] **Step 4: Include the launcher in the image**

Use this Dockerfile copy line:

~~~dockerfile
COPY handler.py engine.py utils.py sglang_launcher.py download_model.py test_input.json ./
~~~

- [ ] **Step 5: Add RunPod Hub metadata**

Add this object near the parser settings in `.runpod/hub.json`:

~~~json
{
  "key": "RESPONSES_DISABLE_THINKING",
  "input": {
    "name": "Disable Responses Thinking",
    "type": "boolean",
    "description": "Disable Qwen-style thinking only for the SGLang Responses API adapter",
    "default": false,
    "required": false,
    "advanced": true
  }
}
~~~

- [ ] **Step 6: Document the option**

Add this README Worker Settings row:

~~~markdown
| `RESPONSES_DISABLE_THINKING` | Disable Qwen-style thinking only for the Responses adapter | false |
~~~

Add below the tool/reasoning section:

~~~markdown
### Responses compatibility for Qwen tool calling

`RESPONSES_DISABLE_THINKING` is an opt-in compatibility switch for Qwen-style
chat templates served through SGLang's `/v1/responses` adapter. Set it to
`true` only when automatic tool calls remain in reasoning instead of emitting
a function call. It injects
`chat_template_kwargs.enable_thinking=false` into the adapter's internal chat
request. The default is `false`, and direct `/v1/chat/completions` requests are
not changed.
~~~

- [ ] **Step 7: Run integration tests**

Run: `python3 -m unittest tests.test_engine tests.test_configuration -v`

Expected: all engine and configuration tests pass.

- [ ] **Step 8: Commit integration and docs**

~~~bash
git add engine.py tests/test_engine.py Dockerfile .runpod/hub.json README.md tests/test_configuration.py
git commit -m "feat: expose responses compatibility mode"
~~~

---

### Task 4: Verify the Complete Worker Change

**Files:**
- Verify only; do not change endpoint or worker settings.

**Interfaces:**
- Consumes: completed SSE proxy, shim, engine command, image metadata, and all existing tests.
- Produces: local evidence that the worker change is internally consistent.

- [ ] **Step 1: Run the full suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: every discovered test passes.

- [ ] **Step 2: Compile changed Python modules**

~~~bash
python3 -m py_compile handler.py engine.py utils.py sglang_launcher.py tests/test_utils.py tests/test_engine.py tests/test_sglang_launcher.py tests/test_configuration.py
~~~

Expected: exit status 0 and no output.

- [ ] **Step 3: Check whitespace and scope**

~~~bash
git diff --check HEAD~3..HEAD
git status --short
~~~

Expected: the diff check exits 0. Status shows only pre-existing unrelated user changes and generated `__pycache__/`; compatibility files are committed.

- [ ] **Step 4: Report the deployment handoff without changing RunPod**

Report the three implementation commits and these later endpoint values:

~~~text
RESPONSES_DISABLE_THINKING=true
MODEL_NAME=QuantTrio/Qwen3.6-35B-A3B-AWQ
~~~

State that worker count, scaling, idle, caching, selected image, and endpoint environment were not changed during local implementation.
