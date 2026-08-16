use std::net::TcpListener;

fn main() {
    let addr = std::env::var("LISTEN").unwrap_or_else(|_| "127.0.0.1:3003".to_string());
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("bind {addr}: {e}");
        std::process::exit(1);
    });
    let bound = listener.local_addr().expect("local_addr");
    eprintln!("inference-server listening on http://{bound}");
    inference_server::serve_listener(listener);
}
