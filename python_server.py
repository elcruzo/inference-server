#!/usr/bin/env python3
"""Fallback HTTP/1.1 server (stdlib). Primary server is the Rust binary."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue
from urllib.parse import urlparse

from engine import DEFAULT_MAX_TOKENS, MAX_TOKENS_CAP, Job, Lcg, LanguageModel, Scheduler

LM = LanguageModel()
SCHED = Scheduler(LM, max_batch=8)


def _error(status: int, message: str) -> bytes:
    ty = "invalid_request_error" if status == 400 else "timeout_error" if status == 408 else "server_error"
    return json.dumps({"error": {"message": message, "type": ty, "code": status}}).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, b'{"status":"ok"}')
        else:
            self._send(404, _error(404, "not found"))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw.decode() or "null")
        except Exception:
            self._send(400, _error(400, "invalid json"))
            return
        if not isinstance(body, dict):
            self._send(400, _error(400, "invalid json"))
            return
        if path == "/v1/completions":
            self._complete(body, chat=False)
        elif path == "/v1/chat/completions":
            self._complete(body, chat=True)
        else:
            self._send(404, _error(404, "not found"))

    def _complete(self, body: dict, chat: bool) -> None:
        if chat:
            msgs = body.get("messages")
            if not isinstance(msgs, list):
                self._send(400, _error(400, "messages required"))
                return
            prompt = ""
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                prompt += f"{m.get('role', 'user')}: {m.get('content', '')}\n"
            prompt += "assistant: "
        else:
            prompt = str(body.get("prompt", ""))
        max_tokens = int(body.get("max_tokens", DEFAULT_MAX_TOKENS))
        max_tokens = max(0, min(max_tokens, MAX_TOKENS_CAP))
        temperature = float(body.get("temperature", 1.0))
        top_p = float(body.get("top_p", 1.0))
        seed = int(body.get("seed", 1))
        stream = bool(body.get("stream", False))
        tokens = list(prompt.encode("utf-8", "replace")) or [ord("t")]
        tx: Queue = Queue()
        SCHED.enqueue(Job(tokens=tokens, max_new=max_tokens, temperature=temperature, top_p=top_p, rng=Lcg(seed), tx=tx))
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                try:
                    tok, done = tx.get(timeout=30)
                except Empty:
                    self.wfile.write(b'data: {"error":"timeout"}\n\n')
                    break
                if tok:
                    self.wfile.write(f"data: {json.dumps({'token': tok})}\n\n".encode())
                    self.wfile.flush()
                if done:
                    self.wfile.write(b"data: [DONE]\n\n")
                    break
            return
        text = []
        while True:
            try:
                tok, done = tx.get(timeout=30)
            except Empty:
                self._send(408, _error(408, "generation timed out"))
                return
            text.append(tok)
            if done:
                break
        out = "".join(text)
        if chat:
            payload = {
                "id": "chatcmpl-py",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": "length"}],
            }
        else:
            payload = {
                "id": "cmpl-py",
                "object": "text_completion",
                "choices": [{"index": 0, "text": out, "finish_reason": "length"}],
            }
        self._send(200, json.dumps(payload).encode())


def main() -> None:
    host = "127.0.0.1"
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3003
    import threading

    threading.Thread(target=SCHED.run_forever, daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"python fallback listening on http://{host}:{port}", file=sys.stderr)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
