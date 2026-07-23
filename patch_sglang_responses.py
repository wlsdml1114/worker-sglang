"""Patch the pinned SGLang Responses adapter during the image build."""

import os
from pathlib import Path


IMPORT_ANCHOR = "import logging\n"
LOGGER_ANCHOR = "logger = logging.getLogger(__name__)\n"
REQUEST_ANCHOR = "            stop=request.stop,\n        )\n"
THINKING_ANCHOR = (
    "    def _is_thinking_enabled_for_request(self, request: ResponsesRequest) -> bool:\n"
    '        """Whether to start the reasoning detector in thinking mode."""\n'
)
TOOL_PARSE_ANCHOR = (
    "                    content, call_info_list = parser.parse_non_stream(content)\n"
    "                    for call_info in call_info_list:\n"
)

HELPER = (
    "\n\n"
    "def _responses_thinking_enabled(model_type, request):\n"
    '    disabled = os.environ.get("RESPONSES_DISABLE_THINKING", "").lower()\n'
    '    if disabled in {"true", "1", "yes"}:\n'
    "        return False\n"
    '    if model_type != "poolside_v1":\n'
    "        return None\n"
    "    reasoning = getattr(request, \"reasoning\", None)\n"
    "    effort = getattr(reasoning, \"effort\", None)\n"
    '    return effort not in {"none", "no_think"}\n'
    "\n\n"
    "def _responses_chat_template_kwargs(model_type, request):\n"
    "    enabled = _responses_thinking_enabled(model_type, request)\n"
    "    if enabled is not None:\n"
    '        return {"enable_thinking": enabled}\n'
    "    return None\n"
    "\n\n"
    "def _responses_sampling_value(model_type, value, parameter):\n"
    "    if value is not None:\n"
    "        return value\n"
    '    if model_type != "poolside_v1":\n'
    "        return None\n"
    '    return {"temperature": 0.7, "top_p": 0.95}[parameter]\n'
)


def responses_thinking_enabled(model_type, effort):
    disabled = os.environ.get("RESPONSES_DISABLE_THINKING", "").lower()
    if disabled in {"true", "1", "yes"}:
        return False
    if model_type != "poolside_v1":
        return None
    return effort not in {"none", "no_think"}


def responses_sampling_value(model_type, value, parameter):
    if value is not None:
        return value
    if model_type != "poolside_v1":
        return None
    return {"temperature": 0.7, "top_p": 0.95}[parameter]


def patch_source(source):
    for anchor in (IMPORT_ANCHOR, LOGGER_ANCHOR, REQUEST_ANCHOR):
        if source.count(anchor) != 1:
            raise ValueError(f"Expected one SGLang source anchor: {anchor!r}")

    source = source.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + "import os\n", 1)
    source = source.replace(LOGGER_ANCHOR, LOGGER_ANCHOR + HELPER, 1)
    source = source.replace(
        REQUEST_ANCHOR,
        "            stop=request.stop,\n"
        "            temperature=_responses_sampling_value("
        'self.reasoning_parser, request.temperature, "temperature"),\n'
        "            top_p=_responses_sampling_value("
        'self.reasoning_parser, request.top_p, "top_p"),\n'
        "            chat_template_kwargs=_responses_chat_template_kwargs(\n"
        "                self.reasoning_parser, request\n"
        "            ),\n"
        "        )\n",
        1,
    )
    if THINKING_ANCHOR in source:
        source = source.replace(
            THINKING_ANCHOR,
            THINKING_ANCHOR
            + '        if self.reasoning_parser == "poolside_v1":\n'
            + "            return _responses_thinking_enabled("
            + "self.reasoning_parser, request)\n",
            1,
        )
    if TOOL_PARSE_ANCHOR in source:
        source = source.replace(
            TOOL_PARSE_ANCHOR,
            "                    raw_tool_content = content\n"
            "                    content, call_info_list = "
            "parser.parse_non_stream(content)\n"
            "                    if not call_info_list:\n"
            "                        content = raw_tool_content\n"
            "                    for call_info in call_info_list:\n",
            1,
        )
    return source


def patch_file(path):
    path = Path(path)
    path.write_text(patch_source(path.read_text(encoding="utf-8")), encoding="utf-8")
    return path
