# Codex Responses Compatibility Design

**Date:** 2026-07-22
**Status:** Approved for implementation planning

## Context

The worker exposes SGLang v0.5.15.post1 through RunPod Serverless. The target
model is `QuantTrio/Qwen3.6-35B-A3B-AWQ`, and Codex connects to custom model
providers through the OpenAI Responses API.

Two compatibility gaps were reproduced against the running endpoint:

1. SGLang emits valid Responses API Server-Sent Events (SSE), but the worker
   reformats each upstream line independently. An upstream frame such as
   `event: response.created` followed by `data: {...}` becomes two `data:`
   records, which is not a valid Responses API SSE frame.
2. The Qwen chat template enables thinking unless
   `chat_template_kwargs.enable_thinking` is explicitly `false`. SGLang's
   v0.5.15 Responses adapter creates an internal `ChatCompletionRequest`
   without forwarding that template argument. In live tests, automatic tool
   selection remained in repetitive reasoning instead of producing a tool
   call.

OpenAI does not generally recommend disabling reasoning for tool requests.
The non-thinking behavior in this design is therefore an explicit,
provider-specific compatibility option rather than the worker default.

## Goals

- Preserve valid upstream Responses API SSE event framing through RunPod.
- Add an opt-in compatibility mode that disables Qwen thinking only for the
  SGLang Responses adapter.
- Keep the default worker behavior unchanged.
- Avoid mutating user prompts or vendoring a complete Qwen chat template.
- Keep Chat Completions, Completions, and native generation behavior unchanged.
- Cover the compatibility behavior with focused unit tests before changing
  implementation code.

## Non-goals

- Changing RunPod worker count, scaling, idle settings, or model caching.
- Upgrading SGLang or changing the pinned CUDA image.
- Implementing a new Responses API translation layer in the worker.
- Disabling reasoning globally for all routes or all models.
- Automatically deploying a new endpoint version as part of the code change.

## Considered Approaches

### 1. Targeted Responses adapter shim (selected)

Start SGLang through a small worker-owned launcher. When the compatibility
environment variable is enabled, the launcher wraps the
`ChatCompletionRequest` factory referenced by SGLang's Responses adapter and
injects:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

The adapter-local reference is patched, so direct Chat Completions requests
remain unaffected. The SGLang installation itself is not edited.

This approach is narrow, reversible, testable, and compatible with the pinned
server version.

### 2. Custom Qwen chat template

A bundled template could default `enable_thinking` to false. This is simpler at
runtime, but it changes template behavior globally and couples the generic
worker to one model family. It is not selected.

### 3. Upgrade SGLang

A later SGLang release may improve Responses reasoning parameter mapping. An
upgrade would require repeating image, CUDA, GDN, parser, cold-start, and tool
calling validation. It is outside this focused compatibility fix.

## Configuration

Add one boolean environment variable:

```text
RESPONSES_DISABLE_THINKING=false
```

Accepted true values follow the worker's existing convention: `true`, `1`, and
`yes`, case-insensitively.

- Unset or false: run SGLang without installing the compatibility shim.
- True: install the shim before SGLang initializes its OpenAI serving objects.

The deployed Qwen endpoint can set this option to `true`. Other deployments
retain standard SGLang behavior unless they explicitly opt in.

## Runtime Design

### SGLang launcher

The engine continues to assemble the same SGLang command-line arguments, but
starts a worker-owned launcher module. The launcher:

1. Reads `RESPONSES_DISABLE_THINKING`.
2. If disabled, immediately runs the normal `sglang.launch_server` module.
3. If enabled, imports SGLang's Responses serving module.
4. Replaces only that module's `ChatCompletionRequest` reference with a small
   factory wrapper.
5. The wrapper copies any existing `chat_template_kwargs`, sets
   `enable_thinking=false` only when absent, and invokes the original factory.
6. Runs the normal SGLang launcher.

Using `setdefault` preserves a future explicit value if SGLang begins
forwarding one. The shim can then be retired after an upstream upgrade and
compatibility validation.

### SSE processing

Streaming is processed as SSE frames, not independent text lines.

1. Decode upstream bytes incrementally as UTF-8.
2. Accumulate lines until the SSE blank-line delimiter.
3. Preserve valid SSE fields such as `event:`, `data:`, `id:`, and `retry:`.
4. Normalize JSON only inside `data:` fields when possible.
5. Emit the complete frame as one worker stream item with exactly one trailing
   blank line.
6. Flush a final unterminated frame when the upstream stream closes.

For Chat Completions streams containing only `data:` fields, the externally
visible format remains `data: <payload>\n\n`. Responses streams retain their
paired `event:` and `data:` fields.

## Error Handling

- Upstream non-2xx responses keep the existing structured worker error shape.
- Invalid JSON in an SSE `data:` field is preserved as text instead of failing
  the stream.
- A shim installation failure is fatal during server startup and should report
  the pinned SGLang compatibility mismatch clearly. Silently running with a
  requested compatibility option disabled would make failures harder to
  diagnose.

## Testing Strategy

Tests are added before implementation and must initially fail for the expected
reason.

### SSE tests

- Preserve a Responses frame containing both `event:` and JSON `data:`.
- Keep Chat Completions `data:` frames and `[DONE]` behavior compatible.
- Handle multiple frames in one upstream chunk.
- Handle a frame split across multiple upstream chunks.
- Flush a final frame without a blank-line terminator.
- Preserve non-JSON data instead of raising.

### Compatibility shim tests

- Leave the original request factory unchanged when the option is false.
- Inject `enable_thinking=false` when the option is true.
- Preserve other existing `chat_template_kwargs`.
- Do not overwrite an explicit future `enable_thinking` value.
- Patch only the Responses adapter reference.
- Verify the engine launches through the worker-owned launcher while retaining
  all existing SGLang CLI arguments.

### Regression tests

- Run the complete existing unit-test suite.
- Run formatting or static checks already available in the repository.
- After a new image is intentionally built and selected, test Responses
  non-streaming, Responses streaming, automatic tool calling, and ordinary Chat
  Completions against the live endpoint.

## Rollout and Validation

1. Implement and verify locally without changing RunPod endpoint settings.
2. Build a new immutable worker image.
3. Update only the endpoint image and set
   `RESPONSES_DISABLE_THINKING=true` for the Qwen endpoint.
4. Do not alter worker count or scaling settings.
5. Wait for existing workers to recycle normally or perform a separately
   approved rollout action.
6. Validate the four live API cases listed above.
7. Configure Codex to use the RunPod Responses endpoint only after those checks
   pass.

## Risks and Mitigations

- **Pinned internal SGLang reference changes:** the shim fails clearly at
  startup and is covered by a version-specific test seam.
- **SSE chunk boundaries differ in production:** incremental buffering tests
  include fragmented and coalesced chunks.
- **Compatibility option leaks into other routes:** the patch targets only the
  Responses module's local factory reference.
- **Future upstream support conflicts with the shim:** `setdefault` preserves
  an explicit upstream value, and the option remains opt-in.
- **Unrelated working-tree changes are included:** commits stage only files
  created or changed for this task.
