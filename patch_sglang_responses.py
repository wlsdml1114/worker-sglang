"""Patch the pinned SGLang Responses adapter during the image build."""

import os
import re
import shlex
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
TOOL_PARSER_ANCHOR = (
    "        if (\n"
    "            content\n"
    "            and chat_tools\n"
    "            and self.tool_call_parser\n"
    '            and request.tool_choice != "none"\n'
    "        ):\n"
    "            parser = FunctionCallParser(\n"
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
    "\n\n"
    "def _normalize_poolside_tool_aliases(model_type, content, chat_tools):\n"
    '    if model_type != "poolside_v1":\n'
    "        return content\n"
    "    tool_names = {\n"
    '        getattr(getattr(tool, "function", None), "name", None)\n'
    "        for tool in chat_tools\n"
    "    }\n"
    '    if "exec_command" not in tool_names:\n'
    "        return content\n"
    "\n"
    "    def replace_read_alias(match):\n"
    "        arguments = dict(re.findall(\n"
    '            r"<arg_key>(path|file_path|workdir)</arg_key>\\s*"\n'
    '            r"<arg_value>(.*?)</arg_value>",\n'
    "            match.group(2), flags=re.DOTALL,\n"
    "        ))\n"
    '        path = arguments.get("path") or arguments.get("file_path")\n'
    "        if not path:\n"
    "            return match.group(0)\n"
    "        command = \"sed -n '1,240p' -- \" + shlex.quote(path)\n"
    '        replacement = "<tool_call>exec_command<arg_key>cmd</arg_key>"\n'
    '        replacement += f"<arg_value>{command}</arg_value>"\n'
    '        if arguments.get("workdir"):\n'
    '            replacement += "<arg_key>workdir</arg_key>"\n'
    '            replacement += f"<arg_value>{arguments[\'workdir\']}</arg_value>"\n'
    '        return replacement + "</tool_call>"\n'
    "\n"
    "    return re.sub(\n"
    '        r"<tool_call>(read|read_file)(.*?)</tool_call>",\n'
    "        replace_read_alias, content, flags=re.DOTALL,\n"
    "    )\n"
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


def normalize_poolside_tool_aliases(content, tool_names):
    if "exec_command" not in tool_names:
        return content

    def replace_read_alias(match):
        arguments = dict(
            re.findall(
                r"<arg_key>(path|file_path|workdir)</arg_key>\s*"
                r"<arg_value>(.*?)</arg_value>",
                match.group(2),
                flags=re.DOTALL,
            )
        )
        path = arguments.get("path") or arguments.get("file_path")
        if not path:
            return match.group(0)
        command = "sed -n '1,240p' -- " + shlex.quote(path)
        replacement = (
            "<tool_call>exec_command"
            "<arg_key>cmd</arg_key>"
            f"<arg_value>{command}</arg_value>"
        )
        if arguments.get("workdir"):
            replacement += (
                "<arg_key>workdir</arg_key>"
                f"<arg_value>{arguments['workdir']}</arg_value>"
            )
        return replacement + "</tool_call>"

    return re.sub(
        r"<tool_call>(read|read_file)(.*?)</tool_call>",
        replace_read_alias,
        content,
        flags=re.DOTALL,
    )


def patch_source(source):
    for anchor in (IMPORT_ANCHOR, LOGGER_ANCHOR, REQUEST_ANCHOR):
        if source.count(anchor) != 1:
            raise ValueError(f"Expected one SGLang source anchor: {anchor!r}")

    source = source.replace(
        IMPORT_ANCHOR, IMPORT_ANCHOR + "import os\nimport re\nimport shlex\n", 1
    )
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
    if TOOL_PARSER_ANCHOR in source:
        source = source.replace(
            TOOL_PARSER_ANCHOR,
            TOOL_PARSER_ANCHOR.replace(
                "            parser = FunctionCallParser(\n",
                "            content = _normalize_poolside_tool_aliases(\n"
                "                self.reasoning_parser, content, chat_tools\n"
                "            )\n"
                "            parser = FunctionCallParser(\n",
            ),
            1,
        )
    return source


def patch_file(path):
    path = Path(path)
    path.write_text(patch_source(path.read_text(encoding="utf-8")), encoding="utf-8")
    return path
