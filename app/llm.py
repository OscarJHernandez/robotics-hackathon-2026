"""Chat with a metacon LLM (through the LiteLLM gateway) that can drive the simulated robot.

The model answers in plain text. To move the robot it writes a command on its own line in double
square brackets, for example [[goto 1.0 0.5]]. The server runs those commands and the chat shows
them. The API key comes from $METACON_LLM_KEY or the macOS Keychain (service "metacon-llm"); it is
never sent to the browser.
"""

import json
import os
import re
import subprocess
import urllib.error
import urllib.request

GATEWAY = os.environ.get("METACON_URL", "http://metacon:4000/v1")
MODEL = os.environ.get("METACON_MODEL", "qwen3:8b")
MAX_TURNS = 20  # earlier messages are dropped
MAX_CHARS = 2000  # per user message

COMMAND = re.compile(r"\[\[([^\[\]]{1,60})\]\]")

SYSTEM = """You are the onboard assistant of a simulated JetHexa, a six-legged robot (hexapod) walking in a \
MuJoCo physics simulation. The user watches it live. Answer briefly and plainly, like a terminal: no markdown \
headings, no tables, at most a few short lines.

You can move the robot. Put a command on its own line in double square brackets; it runs as soon as you write it:
[[goto X Y]]    walk to the point (X, Y) in metres. The floor runs from -2.6 to 2.6 on both axes. +x is east.
[[forward M]]   walk M metres straight ahead
[[turn DEG]]    turn on the spot, positive = left (counter-clockwise)
[[stop]]        stand still
[[reset]]       back to the start, at (0, 0) facing +x
[[random]]      move the red pillar (the end point) to a new random spot; the new spot appears in the state
Use only one movement command per answer: a new command replaces the one still running. Only send a command \
when the user asks for movement. Always write one short, friendly sentence first (what you are doing and why), \
then the command on its own line. The red pillar is the target marker at target_x, target_y. The robot walks \
about 0.13 m per second.

Current robot state: {state}
/no_think"""


def api_key():
    key = os.environ.get("METACON_LLM_KEY", "")
    if key:
        return key
    try:
        return subprocess.run(["security", "find-generic-password", "-s", "metacon-llm", "-a",
                               os.environ.get("USER", ""), "-w"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def clean_history(messages):
    """Keep only user and assistant turns with text, newest MAX_TURNS, each cut to MAX_CHARS."""
    out = []
    for m in messages if isinstance(messages, list) else []:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            text = m["content"].strip()[:MAX_CHARS]
            if text:
                out.append({"role": m["role"], "content": text})
    return out[-MAX_TURNS:]


def stream_chat(messages, state, key):
    """Yield the answer's text in pieces as the model writes it, with <think> blocks removed."""
    body = json.dumps({"model": MODEL, "stream": True, "temperature": 0.4,
                       "messages": [{"role": "system", "content": SYSTEM.format(state=json.dumps(state))}]
                       + messages}).encode()
    request = urllib.request.Request(GATEWAY + "/chat/completions", data=body, method="POST", headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
    thinking = False
    pending = ""
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0].get("delta") or {}
            except (ValueError, KeyError, IndexError):
                continue
            pending += delta.get("content") or ""
            # Drop <think>...</think>; hold back a possible partial tag at the end.
            while True:
                if thinking:
                    end = pending.find("</think>")
                    if end < 0:
                        pending = pending[-8:]
                        break
                    pending, thinking = pending[end + 8:], False
                else:
                    start = pending.find("<think>")
                    if start >= 0:
                        if start:
                            yield pending[:start]
                        pending, thinking = pending[start + 7:], True
                        continue
                    cut = pending.rfind("<")
                    if cut >= 0 and len(pending) - cut < 8:
                        if cut:
                            yield pending[:cut]
                        pending = pending[cut:]
                    else:
                        if pending:
                            yield pending
                        pending = ""
                    break
    if pending and not thinking:
        yield pending


class CommandScanner:
    """Feed it streamed text; it returns each complete [[command]] once, in order."""

    def __init__(self):
        self.text = ""
        self.done = 0

    def feed(self, piece):
        self.text += piece
        found = []
        for match in COMMAND.finditer(self.text, self.done):
            found.append(match.group(1).strip())
            self.done = match.end()
        return found


def describe_error(error):
    if isinstance(error, urllib.error.HTTPError):
        if error.code == 401:
            return "metacon refused the API key (401)"
        if error.code == 503:
            return "metacon is low on memory right now (503), try again in a minute"
        return "metacon answered HTTP {}".format(error.code)
    if isinstance(error, urllib.error.URLError):
        return "cannot reach metacon ({})".format(error.reason)
    return "chat failed: {}".format(error)
