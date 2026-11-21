"""Speech for the web app: transcription and the robot's spoken replies, from Oscar's Voice Control service.

Voice Control runs on Oscar's Mac (vendetta) and keeps all audio in the homelab. The server calls it; the
browser never does, so its key stays here. Settings:

    VOICE_CONTROL_URL      default http://127.0.0.1:8765 (on another machine: vendetta's Tailscale address)
    VOICE_CONTROL_API_KEY  needed when the service is not on this machine
    JETHEXA_VOICE          the robot's voice, by name or id (default "Glados-portal"; empty = the service default)
    JETHEXA_LANGUAGE       spoken language for transcription (default "en"; "auto" detects, about 0.2 s slower)

The service runs one job at a time (speech and transcription share one model), so every call here takes LOCK.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid

URL = os.environ.get("VOICE_CONTROL_URL", "http://127.0.0.1:8765").rstrip("/")
VOICE = os.environ.get("JETHEXA_VOICE", "Glados-portal")
LANGUAGE = os.environ.get("JETHEXA_LANGUAGE", "en")
MAX_SPEAK = 600  # characters of a reply that are spoken
TYPES = {"audio/webm": ".webm", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/aac": ".m4a",
         "audio/ogg": ".ogg", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mpeg": ".mp3"}

LOCK = threading.Lock()
_health = {"at": -1e9, "ok": False}
_voice = {"resolved": False, "id": None}


def api_key():
    return os.environ.get("VOICE_CONTROL_API_KEY", "")


def _request(path, data=None, headers=None, timeout=60):
    headers = dict(headers or {})
    key = api_key()
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(URL + path, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def check():
    """Ask /healthz now and remember the answer."""
    try:
        _health["ok"] = json.loads(_request("/healthz", timeout=3)).get("status") == "ok"
    except (OSError, ValueError):
        _health["ok"] = False
    _health["at"] = time.monotonic()
    return _health["ok"]


def watch(every=30.0):
    """Keep available() fresh from a background thread, so /api/health never waits for the voice service."""
    def loop():
        while True:
            check()
            time.sleep(every)
    threading.Thread(target=loop, name="voice-health", daemon=True).start()


def available():
    """The last /healthz answer (see watch)."""
    return _health["ok"]


def voice_id():
    """The id of the voice named in JETHEXA_VOICE, or None for the service's default voice."""
    if not _voice["resolved"] and VOICE:
        try:
            voices = json.loads(_request("/v1/voices", timeout=10))
            wanted = VOICE.lower()
            _voice["id"] = next((v["voice_id"] for v in voices
                                 if wanted in (str(v.get("voice_id", "")).lower(),
                                               str(v.get("display_name", "")).lower())), None)
            _voice["resolved"] = True
            if _voice["id"] is None:
                print("[voice] no voice named {!r}, using the service default".format(VOICE), flush=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print("[voice] cannot list voices ({}), using the service default".format(error), flush=True)
    return _voice["id"]


def extension(content_type):
    base = (content_type or "").split(";")[0].strip().lower()
    if base not in TYPES:
        raise ValueError("unsupported audio type {!r}".format(base or "none"))
    return TYPES[base]


def multipart(fields, file_name, file_type, audio):
    """A multipart/form-data body with text fields and one file part named "file"."""
    boundary = "jethexa" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append('--{}\r\nContent-Disposition: form-data; name="{}"\r\n\r\n{}\r\n'
                     .format(boundary, name, value).encode())
    parts.append('--{}\r\nContent-Disposition: form-data; name="file"; filename="{}"\r\nContent-Type: {}\r\n\r\n'
                 .format(boundary, file_name, file_type).encode() + audio + b"\r\n")
    parts.append("--{}--\r\n".format(boundary).encode())
    return b"".join(parts), "multipart/form-data; boundary=" + boundary


def transcribe(audio, content_type, wait=True):
    """Words in a recording. With wait=False it returns None at once if the service is busy."""
    ext = extension(content_type)
    body, ctype = multipart({"language": LANGUAGE}, "speech" + ext, content_type.split(";")[0].strip(), audio)
    if not LOCK.acquire(timeout=30 if wait else 0):
        if not wait:
            return None
        raise TimeoutError("voice service is busy")
    try:
        answer = json.loads(_request("/v1/audio/transcriptions", body, {"Content-Type": ctype}))
    finally:
        LOCK.release()
    return str(answer.get("text") or "").strip()


def speak(text):
    """MP3 of the robot's voice saying text (cut to MAX_SPEAK characters)."""
    payload = {"input": text.strip()[:MAX_SPEAK], "response_format": "mp3"}
    voice = voice_id()
    if voice:
        payload["voice"] = voice
    if not LOCK.acquire(timeout=30):
        raise TimeoutError("voice service is busy")
    try:
        return _request("/v1/audio/speech", json.dumps(payload).encode(), {"Content-Type": "application/json"},
                        timeout=90)
    finally:
        LOCK.release()


def describe_error(error):
    if isinstance(error, urllib.error.HTTPError):
        if error.code == 400:
            return "could not understand the audio"
        if error.code == 401:
            return "the voice service refused the key"
        if error.code == 503:
            return "the voice service is busy, try again"
        return "the voice service answered HTTP {}".format(error.code)
    if isinstance(error, TimeoutError):
        return "the voice service is busy, try again"
    if isinstance(error, urllib.error.URLError):
        return "cannot reach the voice service"
    return "voice failed: {}".format(error)
