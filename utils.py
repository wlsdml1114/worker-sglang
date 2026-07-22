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
    chunk = chunk.strip()
    if chunk.startswith("data:"):
        chunk = chunk[5:]
    return _format_sse_data(chunk) + "\n\n"


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


async def async_process_response(response, is_stream, route):
    """Yield a successful OpenAI response or one structured upstream error."""
    if not 200 <= response.status < 300:
        yield {
            "error": f"Request to {route} failed with status {response.status}",
            "details": await response.text(),
        }
        return

    if is_stream:
        async for chunk in async_process_stream(response):
            yield chunk
        return

    async for raw_line in response.content:
        line = raw_line.decode("utf-8").strip()
        if line:
            yield line
