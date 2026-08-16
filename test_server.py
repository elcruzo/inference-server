"""HTTP inference-server tests. Prefer the Rust binary; fall back to python_server.py."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_health(port: int, timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last = None
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=0.3)
            if r.status_code == 200:
                return
            last = r.status_code
        except Exception as e:
            last = e
        time.sleep(0.05)
    raise RuntimeError(f"server did not become healthy: {last}")


def _start_rust(port: int) -> subprocess.Popen | None:
    cargo = _which("cargo")
    if cargo is None:
        return None
    build = subprocess.run([cargo, "build", "--quiet"], cwd=ROOT, capture_output=True, text=True)
    if build.returncode != 0:
        print(build.stderr, file=sys.stderr)
        return None
    binary = ROOT / "target" / "debug" / "inference-server"
    if os.name == "nt":
        binary = binary.with_suffix(".exe")
    if not binary.exists():
        return None
    env = {**os.environ, "LISTEN": f"127.0.0.1:{port}"}
    return subprocess.Popen([str(binary)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def _start_python(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(ROOT / "python_server.py"), str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def base_url():
    port = _free_port()
    proc = _start_rust(port)
    backend = "rust"
    if proc is None:
        proc = _start_python(port)
        backend = "python"
    try:
        _wait_health(port)
    except Exception:
        if backend == "rust":
            proc.terminate()
            proc.wait(timeout=5)
            proc = _start_python(port)
            backend = "python"
            _wait_health(port)
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_health_200(base_url: str):
    r = httpx.get(f"{base_url}/health", timeout=5)
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_completion_generated_neq_prompt(base_url: str):
    prompt = "hello"
    r = httpx.post(
        f"{base_url}/v1/completions",
        json={"prompt": prompt, "max_tokens": 8, "temperature": 0.8, "top_p": 0.95, "seed": 3},
        timeout=10,
    )
    assert r.status_code == 200
    text = r.json()["choices"][0]["text"]
    assert text != prompt
    assert len(text) > 0


def test_max_tokens_respected(base_url: str):
    r = httpx.post(
        f"{base_url}/v1/completions",
        json={"prompt": "Hi", "max_tokens": 4, "temperature": 0, "seed": 1},
        timeout=10,
    )
    assert r.status_code == 200
    text = r.json()["choices"][0]["text"]
    assert len(text) == 4


def test_invalid_json_400(base_url: str):
    r = httpx.post(
        f"{base_url}/v1/completions",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
        timeout=5,
    )
    assert r.status_code == 400
    body = r.json()
    assert "error" in body


def test_concurrent_four_complete(base_url: str):
    def one(i: int):
        return httpx.post(
            f"{base_url}/v1/completions",
            json={"prompt": f"q{i} ", "max_tokens": 6, "temperature": 0.5, "seed": i + 1},
            timeout=15,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, range(4)))
    assert all(r.status_code == 200 for r in results)
    assert all(r.json()["choices"][0]["text"] for r in results)


def test_sse_stream(base_url: str):
    with httpx.stream(
        "POST",
        f"{base_url}/v1/completions",
        json={"prompt": "the ", "max_tokens": 3, "temperature": 0, "seed": 1, "stream": True},
        timeout=10,
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode()
    assert "data:" in body
    assert "token" in body


def test_chat_completions(base_url: str):
    r = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "temperature": 0, "seed": 2},
        timeout=10,
    )
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content != "hi"
    assert len(content) <= 5
