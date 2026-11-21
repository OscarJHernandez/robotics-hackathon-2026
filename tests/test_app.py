"""Tests for the web app's command parser and chat stream handling (no network, no window)."""

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import llm  # noqa: E402
from robot import ARENA, parse_command  # noqa: E402


class ParseCommandTest(unittest.TestCase):
    def test_commands(self):
        self.assertEqual(parse_command("goto 1 0.5"), ("goto", 1.0, 0.5))
        self.assertEqual(parse_command("/goto 1, -0.5"), ("goto", 1.0, -0.5))
        self.assertEqual(parse_command("goto 9 -9"), ("goto", ARENA, -ARENA))
        self.assertEqual(parse_command("forward 0.5"), ("forward", 0.5))
        self.assertAlmostEqual(parse_command("turn 90")[1], math.pi / 2)
        self.assertEqual(parse_command("STOP"), ("stop",))
        self.assertEqual(parse_command("reset"), ("reset",))

    def test_rejects(self):
        for text in ("", "fly", "goto 1", "goto a b", "forward -1", "turn 720", "goto nan 0", "stop now"):
            with self.assertRaises(ValueError, msg=text):
                parse_command(text)


class ChatStreamTest(unittest.TestCase):
    def test_command_scanner_across_pieces(self):
        scanner = llm.CommandScanner()
        found = []
        for piece in ["Sure. [[", "forward", " 0.5]] then [[turn 9", "0]]", " done"]:
            found += scanner.feed(piece)
        self.assertEqual(found, ["forward 0.5", "turn 90"])

    def test_think_blocks_are_removed(self):
        pieces = ["<thi", "nk>planning", " steps</th", "ink>Hello", " <b>there</b>"]
        lines = [b'data: {"choices":[{"delta":{"content":%s}}]}\n' % __import__("json").dumps(p).encode()
                 for p in pieces] + [b"data: [DONE]\n"]

        class Response(list):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with mock.patch("urllib.request.urlopen", return_value=Response(lines)):
            text = "".join(llm.stream_chat([{"role": "user", "content": "hi"}], {}, "key"))
        self.assertEqual(text, "Hello <b>there</b>")

    def test_clean_history(self):
        history = [{"role": "system", "content": "x"}, {"role": "user", "content": " hi "},
                   {"role": "assistant", "content": ""}, "junk", {"role": "user", "content": "a" * 5000}]
        cleaned = llm.clean_history(history)
        self.assertEqual([m["role"] for m in cleaned], ["user", "user"])
        self.assertEqual(cleaned[0]["content"], "hi")
        self.assertEqual(len(cleaned[1]["content"]), llm.MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
