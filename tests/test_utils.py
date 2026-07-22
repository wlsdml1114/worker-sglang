import asyncio
import unittest

from utils import async_process_response, format_sse_chunk


class FakeContent:
    def __init__(self, lines):
        self._lines = lines

    def __aiter__(self):
        self._iterator = iter(self._lines)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration as error:
            raise StopAsyncIteration from error


class FakeResponse:
    def __init__(self, status=200, lines=(), body=""):
        self.status = status
        self.content = FakeContent(lines)
        self._body = body

    async def text(self):
        return self._body


async def collect_response(response, is_stream=False, route="/v1/chat/completions"):
    return [
        item
        async for item in async_process_response(response, is_stream, route)
    ]


class ResponseProcessingTests(unittest.TestCase):
    def test_format_sse_chunk_formats_raw_json_payload(self):
        self.assertEqual(
            format_sse_chunk('{"token":"hello"}'),
            'data: {"token": "hello"}\n\n',
        )

    def test_format_sse_chunk_formats_raw_done_marker(self):
        self.assertEqual(format_sse_chunk("[DONE]"), "data: [DONE]\n\n")

    def test_non_success_response_yields_structured_error(self):
        response = FakeResponse(status=500, body="boom")

        result = asyncio.run(collect_response(response))

        self.assertEqual(
            result,
            [
                {
                    "error": (
                        "Request to /v1/chat/completions failed with status 500"
                    ),
                    "details": "boom",
                }
            ],
        )

    def test_non_stream_response_yields_non_empty_decoded_lines(self):
        response = FakeResponse(lines=[b'{"id":"one"}\n', b"\n", b'{"id":"two"}\n'])

        result = asyncio.run(collect_response(response))

        self.assertEqual(result, ['{"id":"one"}', '{"id":"two"}'])

    def test_stream_response_formats_sse_and_done_marker(self):
        response = FakeResponse(
            lines=[b'data: {"token":"hello"}\n\n', b"data: [DONE]\n\n"]
        )

        result = asyncio.run(collect_response(response, is_stream=True))

        self.assertEqual(
            result,
            ['data: {"token": "hello"}\n\n', "data: [DONE]\n\n"],
        )

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


if __name__ == "__main__":
    unittest.main()
