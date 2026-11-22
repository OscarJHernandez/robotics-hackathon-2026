"""Tests for the web app's static files: byte ranges (Safari needs them to play a video), 404 for missing media,
the single-page fallback, and no bytes from outside the built front end. The server runs on a free local port
and serves a temporary folder instead of app/web/dist."""

import http.client
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import server  # noqa: E402

CLIP = bytes(range(256)) * 4  # 1,024 bytes, every byte position recognisable
PAGE = b"<!doctype html><title>page</title>"


class StaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        dist = base / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_bytes(PAGE)
        (dist / "assets" / "clip-abc123.mp4").write_bytes(CLIP)
        (base / "secret.txt").write_bytes(b"outside the front end")
        cls.patch = mock.patch.object(server, "WEB", dist)
        cls.patch.start()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.httpd.daemon_threads = True
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.patch.stop()
        cls.tmp.cleanup()

    def get(self, path, range_=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
        try:
            conn.request("GET", path, headers={"Range": range_} if range_ else {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def test_first_two_bytes(self):
        status, headers, body = self.get("/assets/clip-abc123.mp4", "bytes=0-1")
        self.assertEqual(status, 206)
        self.assertEqual(body, CLIP[:2])
        self.assertEqual(headers["Content-Length"], "2")
        self.assertEqual(headers["Content-Range"], "bytes 0-1/1024")
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        self.assertEqual(headers["Content-Type"], "video/mp4")

    def test_bounded_range(self):
        status, headers, body = self.get("/assets/clip-abc123.mp4", "bytes=100-299")
        self.assertEqual(status, 206)
        self.assertEqual(body, CLIP[100:300])
        self.assertEqual(headers["Content-Range"], "bytes 100-299/1024")
        status, headers, body = self.get("/assets/clip-abc123.mp4", "bytes=1000-5000")  # end past the file
        self.assertEqual(status, 206)
        self.assertEqual(body, CLIP[1000:])
        self.assertEqual(headers["Content-Range"], "bytes 1000-1023/1024")

    def test_open_ended_range(self):
        status, headers, body = self.get("/assets/clip-abc123.mp4", "bytes=900-")
        self.assertEqual(status, 206)
        self.assertEqual(body, CLIP[900:])
        self.assertEqual(headers["Content-Length"], "124")
        self.assertEqual(headers["Content-Range"], "bytes 900-1023/1024")

    def test_suffix_range(self):
        status, headers, body = self.get("/assets/clip-abc123.mp4", "bytes=-100")
        self.assertEqual(status, 206)
        self.assertEqual(body, CLIP[-100:])
        self.assertEqual(headers["Content-Range"], "bytes 924-1023/1024")
        status, _, body = self.get("/assets/clip-abc123.mp4", "bytes=-5000")  # more than the file: all of it
        self.assertEqual(status, 206)
        self.assertEqual(body, CLIP)

    def test_unsatisfiable(self):
        status, headers, body = self.get("/assets/clip-abc123.mp4", "bytes=1024-")
        self.assertEqual(status, 416)
        self.assertEqual(headers["Content-Range"], "bytes */1024")
        self.assertEqual(body, b"")

    def test_multi_range_and_malformed_send_everything(self):
        for header in ("bytes=0-1,5-6", "bytes=abc", "items=0-1", "bytes=5-2", "bytes=-"):
            status, headers, body = self.get("/assets/clip-abc123.mp4", header)
            self.assertEqual(status, 200, header)
            self.assertEqual(body, CLIP, header)
            self.assertNotIn("Content-Range", headers, header)

    def test_full_get_unchanged(self):
        status, headers, body = self.get("/assets/clip-abc123.mp4")
        self.assertEqual(status, 200)
        self.assertEqual(body, CLIP)
        self.assertEqual(headers["Content-Length"], "1024")
        self.assertIn("immutable", headers["Cache-Control"])
        status, headers, body = self.get("/")
        self.assertEqual((status, body), (200, PAGE))
        self.assertEqual(headers["Cache-Control"], "no-cache")

    def test_missing_media_is_404(self):
        for path in ("/assets/clip-old999.mp4", "/assets/index-old.js", "/assets/poster.jpg"):
            status, headers, _ = self.get(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(headers["Content-Type"], "application/json", path)

    def test_spa_fallback(self):
        for path in ("/how-it-works", "/some/page", "/?phone=1"):
            status, _, body = self.get(path)
            self.assertEqual((status, body), (200, PAGE), path)

    def test_escaping_path_never_leaves_the_front_end(self):
        for path in ("/../secret.txt", "/assets/../../secret.txt", "//../secret.txt", "/..%2fsecret.txt"):
            status, _, body = self.get(path)
            self.assertNotIn(b"outside", body, path)
            self.assertIn(status, (200, 404), path)
        status, _, body = self.get("/../secret.txt", "bytes=0-5")
        self.assertNotIn(b"outsid", body)

    def test_byte_range_parser(self):
        self.assertIsNone(server.byte_range(None, 10))
        self.assertEqual(server.byte_range("bytes=0-0", 10), (0, 0))
        self.assertEqual(server.byte_range("bytes=-3", 10), (7, 9))
        self.assertEqual(server.byte_range("bytes=-0", 10), "unsatisfiable")
        self.assertEqual(server.byte_range("bytes=0-", 0), "unsatisfiable")


if __name__ == "__main__":
    unittest.main()
