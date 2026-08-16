//! HTTP/1.1 (std TcpListener) + OpenAI-ish routes + SSE + JSON errors + /metrics.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::json::{escape, Json};
use crate::lm::{LanguageModel, Lcg};
use crate::metrics::Metrics;
use crate::scheduler::{Job, Scheduler};
use crate::{DEFAULT_MAX_TOKENS, GEN_TIMEOUT_SECS, MAX_TOKENS_CAP};

pub struct Engine {
    pub lm: LanguageModel,
    pub sched: Scheduler,
}

pub struct Shared {
    pub engine: Mutex<Engine>,
    pub cv: Condvar,
    pub next_id: AtomicU64,
    pub metrics: Metrics,
}

impl Shared {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            engine: Mutex::new(Engine {
                lm: LanguageModel::default_model(),
                sched: Scheduler::new(8),
            }),
            cv: Condvar::new(),
            next_id: AtomicU64::new(1),
            metrics: Metrics::default(),
        })
    }
}

pub fn serve(addr: &str) {
    let listener = TcpListener::bind(addr).expect("bind");
    serve_listener(listener);
}

pub fn serve_listener(listener: TcpListener) {
    let shared = Shared::new();
    let worker = shared.clone();
    thread::spawn(move || scheduler_loop(worker));
    listener.set_nonblocking(false).ok();
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let shared = shared.clone();
                thread::spawn(move || {
                    if let Err(e) = handle_client(s, &shared) {
                        eprintln!("conn: {e}");
                    }
                });
            }
            Err(e) => eprintln!("accept: {e}"),
        }
    }
}

fn scheduler_loop(shared: Arc<Shared>) {
    loop {
        let mut guard = shared.engine.lock().expect("engine lock");
        while !guard.sched.has_work() {
            guard = shared.cv.wait(guard).expect("cv");
        }
        let Engine { lm, sched } = &mut *guard;
        sched.step(lm);
        shared.metrics.tick_sched();
    }
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    loop {
        let n = stream.read(&mut tmp).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_crlf2(&buf) {
            let header = std::str::from_utf8(&buf[..pos]).map_err(|e| e.to_string())?;
            let mut lines = header.split("\r\n");
            let reqline = lines.next().ok_or("empty request")?;
            let mut parts = reqline.split_whitespace();
            let method = parts.next().ok_or("no method")?.to_string();
            let raw_path = parts.next().ok_or("no path")?.to_string();
            let path = raw_path.split('?').next().unwrap_or(&raw_path).to_string();
            let mut content_len = 0usize;
            for line in lines {
                if let Some((k, v)) = line.split_once(':') {
                    if k.trim().eq_ignore_ascii_case("content-length") {
                        content_len = v.trim().parse().unwrap_or(0);
                    }
                }
            }
            if content_len > 1_000_000 {
                return Err("body too large".into());
            }
            let mut body = buf[pos + 4..].to_vec();
            while body.len() < content_len {
                let n = stream.read(&mut tmp).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&tmp[..n]);
            }
            body.truncate(content_len);
            return Ok(Request { method, path, body });
        }
        if buf.len() > 1_000_000 {
            return Err("headers too large".into());
        }
    }
    Err("incomplete request".into())
}

fn find_crlf2(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn write_http(stream: &mut TcpStream, status: u16, reason: &str, ctype: &str, body: &[u8]) -> Result<(), String> {
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    stream.write_all(body).map_err(|e| e.to_string())?;
    Ok(())
}

fn error_body(status: u16, message: &str) -> String {
    let ty = if status == 400 {
        "invalid_request_error"
    } else if status == 408 {
        "timeout_error"
    } else {
        "server_error"
    };
    format!(
        "{{\"error\":{{\"message\":\"{}\",\"type\":\"{ty}\",\"code\":{status}}}}}",
        escape(message)
    )
}

fn write_error(stream: &mut TcpStream, status: u16, reason: &str, message: &str) -> Result<(), String> {
    write_http(stream, status, reason, "application/json", error_body(status, message).as_bytes())
}

fn handle_client(mut stream: TcpStream, shared: &Arc<Shared>) -> Result<(), String> {
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| e.to_string())?;
    let req = match read_request(&mut stream) {
        Ok(r) => r,
        Err(e) => {
            shared.metrics.record_error();
            let _ = write_error(&mut stream, 400, "Bad Request", &e);
            return Ok(());
        }
    };
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            write_http(&mut stream, 200, "OK", "application/json", br#"{"status":"ok"}"#)
        }
        ("GET", "/metrics") => {
            let body = shared.metrics.snapshot_json();
            write_http(&mut stream, 200, "OK", "application/json", body.as_bytes())
        }
        ("POST", "/v1/completions") => handle_completion(&mut stream, shared, &req.body, false),
        ("POST", "/v1/chat/completions") => handle_completion(&mut stream, shared, &req.body, true),
        (m, _) if m != "GET" && m != "POST" => {
            write_error(&mut stream, 405, "Method Not Allowed", "method not allowed")
        }
        _ => write_error(&mut stream, 404, "Not Found", "not found"),
    }
}

struct GenParams {
    prompt: String,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    seed: u64,
    stream: bool,
}

fn parse_gen(body: &[u8], chat: bool) -> Result<GenParams, String> {
    let s = std::str::from_utf8(body).map_err(|_| "invalid utf-8".to_string())?;
    if s.is_empty() {
        return Err("empty body".into());
    }
    let j = Json::parse(s).map_err(|e| e.0)?;
    let prompt = if chat {
        let msgs = j
            .get("messages")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "messages required".to_string())?;
        let mut p = String::new();
        for m in msgs {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
            p.push_str(role);
            p.push_str(": ");
            p.push_str(content);
            p.push('\n');
        }
        p.push_str("assistant: ");
        p
    } else {
        j.get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    };
    let mut max_tokens = j
        .get("max_tokens")
        .and_then(|v| v.as_usize())
        .unwrap_or(DEFAULT_MAX_TOKENS);
    if max_tokens > MAX_TOKENS_CAP {
        max_tokens = MAX_TOKENS_CAP;
    }
    let temperature = j.get("temperature").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let top_p = j.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let seed = j.get("seed").and_then(|v| v.as_u64()).unwrap_or(1);
    let stream = j.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    Ok(GenParams {
        prompt,
        max_tokens,
        temperature,
        top_p,
        seed,
        stream,
    })
}

fn handle_completion(
    stream: &mut TcpStream,
    shared: &Arc<Shared>,
    body: &[u8],
    chat: bool,
) -> Result<(), String> {
    let started = Instant::now();
    let params = match parse_gen(body, chat) {
        Ok(p) => p,
        Err(e) => {
            shared.metrics.record_error();
            return write_error(stream, 400, "Bad Request", &e);
        }
    };
    let id = shared.next_id.fetch_add(1, Ordering::Relaxed);
    let (tx, rx) = mpsc::channel();
    let ids = {
        let g = shared.engine.lock().map_err(|e| e.to_string())?;
        g.lm.encode(&params.prompt)
    };
    {
        let mut g = shared.engine.lock().map_err(|e| e.to_string())?;
        g.sched.enqueue(Job {
            ids,
            max_new: params.max_tokens,
            n_new: 0,
            temperature: params.temperature,
            top_p: params.top_p,
            rng: Lcg::new(params.seed),
            caches: None,
            last_logits: None,
            tx,
        });
        shared.cv.notify_one();
    }

    if params.stream {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
        stream.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
        let deadline = Instant::now() + Duration::from_secs(GEN_TIMEOUT_SECS);
        let mut n_tok = 0u64;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(left) {
                Ok(ev) => {
                    if !ev.token.is_empty() {
                        n_tok += 1;
                        let line = format!("data: {{\"token\":\"{}\"}}\n\n", escape(&ev.token));
                        stream.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
                        stream.flush().ok();
                    }
                    if ev.done {
                        stream.write_all(b"data: [DONE]\n\n").map_err(|e| e.to_string())?;
                        shared.metrics.record_ok(n_tok, started);
                        break;
                    }
                }
                Err(_) => {
                    shared.metrics.record_error();
                    stream.write_all(b"data: {\"error\":\"timeout\"}\n\n").ok();
                    break;
                }
            }
        }
        return Ok(());
    }

    let mut text = String::new();
    let deadline = Instant::now() + Duration::from_secs(GEN_TIMEOUT_SECS);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(left) {
            Ok(ev) => {
                text.push_str(&ev.token);
                if ev.done {
                    break;
                }
            }
            Err(_) => {
                shared.metrics.record_error();
                return write_error(stream, 408, "Request Timeout", "generation timed out");
            }
        }
    }
    shared.metrics.record_ok(text.len() as u64, started);
    let body = if chat {
        format!(
            "{{\"id\":\"chatcmpl-{id}\",\"object\":\"chat.completion\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\"finish_reason\":\"length\"}}],\"usage\":{{\"completion_tokens\":{}}}}}",
            escape(&text),
            text.len()
        )
    } else {
        format!(
            "{{\"id\":\"cmpl-{id}\",\"object\":\"text_completion\",\"choices\":[{{\"index\":0,\"text\":\"{}\",\"finish_reason\":\"length\"}}],\"usage\":{{\"completion_tokens\":{}}}}}",
            escape(&text),
            text.len()
        )
    };
    write_http(stream, 200, "OK", "application/json", body.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::{Shutdown, TcpListener, TcpStream};
    use std::thread;
    use std::time::Duration;

    fn spawn_server() -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || serve_listener(listener));
        for _ in 0..80 {
            if let Ok(r) = try_exchange(addr, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n") {
                if r.contains("200") {
                    return addr;
                }
            }
            thread::sleep(Duration::from_millis(15));
        }
        panic!("server did not start on {addr}");
    }

    fn try_exchange(addr: std::net::SocketAddr, req: &str) -> std::io::Result<String> {
        let mut s = TcpStream::connect_timeout(&addr, Duration::from_millis(200))?;
        s.set_read_timeout(Some(Duration::from_secs(30)))?;
        s.write_all(req.as_bytes())?;
        s.shutdown(Shutdown::Write)?;
        let mut buf = Vec::new();
        s.read_to_end(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn exchange(addr: std::net::SocketAddr, req: &str) -> String {
        try_exchange(addr, req).unwrap()
    }

    #[test]
    fn health_ok() {
        let addr = spawn_server();
        let r = exchange(addr, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert!(r.contains("200"), "{r}");
        assert!(r.contains("ok"), "{r}");
    }

    #[test]
    fn metrics_endpoint() {
        let addr = spawn_server();
        let body = r#"{"prompt":"hi","max_tokens":2,"temperature":0,"seed":1}"#;
        let req = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let _ = exchange(addr, &req);
        let r = exchange(addr, "GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert!(r.contains("200"), "{r}");
        assert!(r.contains("tokens_total"), "{r}");
        assert!(r.contains("mean_latency_ms"), "{r}");
    }

    #[test]
    fn completion_not_echo_and_max_tokens() {
        let addr = spawn_server();
        let body = r#"{"prompt":"hello","max_tokens":5,"temperature":0,"seed":1}"#;
        let req = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let r = exchange(addr, &req);
        assert!(r.contains("200"), "{r}");
        assert!(r.contains("\"text\":"), "{r}");
        assert!(!r.contains("\"text\":\"hello\""), "{r}");
        assert!(r.contains("\"completion_tokens\":5"), "{r}");
    }

    #[test]
    fn invalid_json_is_400() {
        let addr = spawn_server();
        let body = "{not json";
        let req = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let r = exchange(addr, &req);
        assert!(r.contains("400"), "{r}");
        assert!(r.contains("invalid_request_error"), "{r}");
    }
}
