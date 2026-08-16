"""HTTP inference-server tests. Rust binary is the primary path; Python server is a separate named backend when rustc/cargo is unavailable."""

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


def _wait_health(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last = None
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=0.5)
            if r.status_code == 200:
                return
            last = r.status_code
        except Exception as e:
            last = e
        time.sleep(0.05)
    raise RuntimeError(f"server did not become healthy: {last}")


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def _start_rust(port: int) -> subprocess.Popen:
    cargo = _which("cargo")
    if cargo is None:
        raise RuntimeError("cargo not available")
    build = subprocess.run([cargo, "build", "--quiet"], cwd=ROOT, capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(f"cargo build failed: {build.stderr[-500:]}")
    binary = ROOT / "target" / "debug" / "inference-server"
    if os.name == "nt":
        binary = binary.with_suffix(".exe")
    if not binary.exists():
        raise RuntimeError("cargo build succeeded but binary missing")
    env = {**os.environ, "LISTEN": f"127.0.0.1:{port}"}
    return subprocess.Popen([str(binary)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
    if _which("cargo") is not None:
        backend = "rust"
        proc = _start_rust(port)
    else:
        backend = "python"
        proc = _start_python(port)
    try:
        _wait_health(port)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise RuntimeError(f"{backend} server failed health check; not switching backends") from None
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
        timeout=60,
    )
    assert r.status_code == 200
    text = r.json()["choices"][0]["text"]
    assert text != prompt
    assert len(text) == 8


def test_max_tokens_respected(base_url: str):
    r = httpx.post(
        f"{base_url}/v1/completions",
        json={"prompt": "Hi", "max_tokens": 4, "temperature": 0, "seed": 1},
        timeout=60,
    )
    assert r.status_code == 200
    text = r.json()["choices"][0]["text"]
    assert len(text) == 4
    assert r.json()["usage"]["completion_tokens"] == 4


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
            timeout=60,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, range(4)))
    assert all(r.status_code == 200 for r in results)
    assert all(len(r.json()["choices"][0]["text"]) == 6 for r in results)


def test_sse_stream(base_url: str):
    with httpx.stream(
        "POST",
        f"{base_url}/v1/completions",
        json={"prompt": "the ", "max_tokens": 3, "temperature": 0, "seed": 1, "stream": True},
        timeout=60,
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode()
    assert "data:" in body
    assert "token" in body
    assert "[DONE]" in body


def test_chat_completions(base_url: str):
    r = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "temperature": 0, "seed": 2},
        timeout=60,
    )
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content != "hi"
    assert len(content) == 5


def test_metrics_exposed(base_url: str):
    httpx.post(
        f"{base_url}/v1/completions",
        json={"prompt": "ab", "max_tokens": 2, "temperature": 0, "seed": 1},
        timeout=60,
    )
    r = httpx.get(f"{base_url}/metrics", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["tokens_total"] >= 2
    assert "mean_latency_ms" in body
    assert "scheduler_steps" in body
