import importlib.abc
import importlib.machinery
import os
import runpy
import sys


TRUE_VALUES = {"true", "1", "yes"}
RESPONSES_MODULE = "sglang.srt.entrypoints.openai.serving_responses"


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


class _ResponsesPatchLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        return create_module(spec) if create_module else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        install_responses_compatibility(module)


class _ResponsesPatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target):
        self.target = target

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _ResponsesPatchLoader(spec.loader)
        return spec


def install_responses_import_hook():
    """Install the Responses compatibility shim when its module loads."""
    loaded_module = sys.modules.get(RESPONSES_MODULE)
    if loaded_module is not None:
        install_responses_compatibility(loaded_module)
        return

    if any(
        isinstance(finder, _ResponsesPatchFinder)
        and finder.target == RESPONSES_MODULE
        for finder in sys.meta_path
    ):
        return
    sys.meta_path.insert(0, _ResponsesPatchFinder(RESPONSES_MODULE))


def main(env=None):
    if responses_thinking_disabled(env):
        install_responses_import_hook()
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
