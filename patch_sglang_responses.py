"""Patch the pinned SGLang Responses adapter during the image build."""

from pathlib import Path


IMPORT_ANCHOR = "import logging\n"
LOGGER_ANCHOR = "logger = logging.getLogger(__name__)\n"
REQUEST_ANCHOR = "            stop=request.stop,\n        )\n"

HELPER = (
    "\n\n"
    "def _responses_chat_template_kwargs():\n"
    '    disabled = os.environ.get("RESPONSES_DISABLE_THINKING", "").lower()\n'
    '    if disabled in {"true", "1", "yes"}:\n'
    '        return {"enable_thinking": False}\n'
    "    return None\n"
)


def patch_source(source):
    for anchor in (IMPORT_ANCHOR, LOGGER_ANCHOR, REQUEST_ANCHOR):
        if source.count(anchor) != 1:
            raise ValueError(f"Expected one SGLang source anchor: {anchor!r}")

    source = source.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + "import os\n", 1)
    source = source.replace(LOGGER_ANCHOR, LOGGER_ANCHOR + HELPER, 1)
    source = source.replace(
        REQUEST_ANCHOR,
        "            stop=request.stop,\n"
        "            chat_template_kwargs=_responses_chat_template_kwargs(),\n"
        "        )\n",
        1,
    )
    return source


def patch_file(path):
    path = Path(path)
    path.write_text(patch_source(path.read_text(encoding="utf-8")), encoding="utf-8")
    return path
