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
    if current_factory.__dict__.get("_responses_disable_thinking", False):
        return
    module.ChatCompletionRequest = wrap_chat_completion_request(current_factory)


def main(env=None):
    if responses_thinking_disabled(env):
        install_responses_compatibility()
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
