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
