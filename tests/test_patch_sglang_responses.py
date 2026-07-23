import importlib.util
import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class PatchSGLangResponsesTests(unittest.TestCase):
    def test_patcher_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("patch_sglang_responses")
        )

    def test_patcher_exposes_source_transform(self):
        module = importlib.import_module("patch_sglang_responses")
        self.assertTrue(hasattr(module, "patch_source"))

    def test_source_transform_adds_runtime_flag_to_responses_request(self):
        module = importlib.import_module("patch_sglang_responses")
        source = (
            "import logging\n"
            "\n"
            "logger = logging.getLogger(__name__)\n"
            "\n"
            "        chat_request = ChatCompletionRequest(\n"
            "            model=request.model,\n"
            "            stop=request.stop,\n"
            "        )\n"
        )

        patched = module.patch_source(source)

        self.assertIn("import os\n", patched)
        self.assertIn(
            "def _responses_chat_template_kwargs(model_type, request):", patched
        )
        self.assertIn(
            'os.environ.get("RESPONSES_DISABLE_THINKING", "").lower()',
            patched,
        )
        self.assertIn(
            "chat_template_kwargs=_responses_chat_template_kwargs(",
            patched,
        )

    def test_poolside_responses_explicitly_enable_thinking_and_parser_state(self):
        module = importlib.import_module("patch_sglang_responses")
        source = (
            "import logging\n"
            "\n"
            "logger = logging.getLogger(__name__)\n"
            "\n"
            "        chat_request = ChatCompletionRequest(\n"
            "            stop=request.stop,\n"
            "        )\n"
            "\n"
            "    def _is_thinking_enabled_for_request("
            "self, request: ResponsesRequest) -> bool:\n"
            "        \"\"\"Whether to start the reasoning detector in thinking mode.\"\"\"\n"
            "        if not self.reasoning_parser:\n"
            "            return False\n"
        )

        patched = module.patch_source(source)

        self.assertIn(
            "self.reasoning_parser, request",
            patched,
        )
        self.assertIn(
            "if self.reasoning_parser == \"poolside_v1\":",
            patched,
        )
        self.assertIn(
            "return _responses_thinking_enabled(self.reasoning_parser, request)",
            patched,
        )

    def test_poolside_thinking_is_enabled_unless_request_explicitly_disables_it(self):
        module = importlib.import_module("patch_sglang_responses")

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(module.responses_thinking_enabled("poolside_v1", None))
            self.assertTrue(module.responses_thinking_enabled("poolside_v1", "low"))
            self.assertFalse(module.responses_thinking_enabled("poolside_v1", "none"))
            self.assertIsNone(module.responses_thinking_enabled("qwen3", "low"))

    def test_legacy_disable_flag_still_wins_for_poolside(self):
        module = importlib.import_module("patch_sglang_responses")

        with mock.patch.dict(
            os.environ, {"RESPONSES_DISABLE_THINKING": "true"}, clear=True
        ):
            self.assertFalse(module.responses_thinking_enabled("poolside_v1", "high"))

    def test_patcher_exposes_file_transform(self):
        module = importlib.import_module("patch_sglang_responses")
        self.assertTrue(hasattr(module, "patch_file"))

    def test_file_transform_rewrites_pinned_adapter_source(self):
        module = importlib.import_module("patch_sglang_responses")
        source = (
            "import logging\n"
            "\n"
            "logger = logging.getLogger(__name__)\n"
            "\n"
            "        chat_request = ChatCompletionRequest(\n"
            "            stop=request.stop,\n"
            "        )\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "serving_responses.py"
            path.write_text(source)

            module.patch_file(path)

            patched = path.read_text()
        self.assertIn(
            "chat_template_kwargs=_responses_chat_template_kwargs(",
            patched,
        )

    def test_transformed_source_is_valid_python(self):
        module = importlib.import_module("patch_sglang_responses")
        source = (
            "import logging\n"
            "\n"
            "logger = logging.getLogger(__name__)\n"
            "\n"
            "def make_request(request):\n"
            "    chat_request = ChatCompletionRequest(\n"
            "            stop=request.stop,\n"
            "        )\n"
        )
        patched = module.patch_source(source)

        try:
            compile(patched, "<patched-serving-responses>", "exec")
        except SyntaxError as error:
            self.fail(f"Patched source did not compile: {error}")


if __name__ == "__main__":
    unittest.main()
