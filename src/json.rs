//! Minimal JSON parser (objects, arrays, strings, numbers, bool, null).

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

#[derive(Debug)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ParseError {}

impl Json {
    pub fn parse(s: &str) -> Result<Json, ParseError> {
        let mut p = Parser { s: s.as_bytes(), i: 0 };
        p.skip_ws();
        let v = p.parse_value()?;
        p.skip_ws();
        if p.i != p.s.len() {
            return Err(ParseError("trailing junk after JSON value".into()));
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_f64().map(|n| n.max(0.0) as usize)
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_f64().map(|n| n.max(0.0) as u64)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn bump_byte(&mut self) -> Result<u8, ParseError> {
        let b = self.peek().ok_or_else(|| ParseError("unexpected end of JSON".into()))?;
        self.i += 1;
        Ok(b)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.i += 1;
        }
    }

    fn parse_value(&mut self) -> Result<Json, ParseError> {
        self.skip_ws();
        match self.peek() {
            Some(b'n') => self.parse_lit(b"null", Json::Null),
            Some(b't') => self.parse_lit(b"true", Json::Bool(true)),
            Some(b'f') => self.parse_lit(b"false", Json::Bool(false)),
            Some(b'"') => Ok(Json::String(self.parse_string()?)),
            Some(b'[') => Ok(Json::Array(self.parse_array()?)),
            Some(b'{') => Ok(Json::Object(self.parse_object()?)),
            Some(b'-') | Some(b'0'..=b'9') => Ok(Json::Number(self.parse_number()?)),
            Some(c) => Err(ParseError(format!("unexpected byte {c}"))),
            None => Err(ParseError("unexpected end of JSON".into())),
        }
    }

    fn parse_lit(&mut self, lit: &[u8], v: Json) -> Result<Json, ParseError> {
        if self.s[self.i..].starts_with(lit) {
            self.i += lit.len();
            Ok(v)
        } else {
            Err(ParseError(format!("expected {}", String::from_utf8_lossy(lit))))
        }
    }

    fn parse_number(&mut self) -> Result<f64, ParseError> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        let s = std::str::from_utf8(&self.s[start..self.i])
            .map_err(|_| ParseError("invalid number utf8".into()))?;
        s.parse().map_err(|_| ParseError(format!("invalid number {s}")))
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        if self.bump_byte()? != b'"' {
            return Err(ParseError("expected string".into()));
        }
        let mut out = String::new();
        loop {
            let c = self.bump_byte()?;
            match c {
                b'"' => return Ok(out),
                b'\\' => match self.bump_byte()? {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let mut hex = [0u8; 4];
                        for slot in &mut hex {
                            *slot = self.bump_byte()?;
                        }
                        let hex_s = std::str::from_utf8(&hex)
                            .map_err(|_| ParseError("bad unicode escape".into()))?;
                        let n = u32::from_str_radix(hex_s, 16)
                            .map_err(|_| ParseError("bad unicode escape".into()))?;
                        out.push(char::from_u32(n).unwrap_or('\u{FFFD}'));
                    }
                    other => return Err(ParseError(format!("bad escape {other}"))),
                },
                c if c < 0x20 => return Err(ParseError("unescaped control in string".into())),
                c if c < 0x80 => out.push(c as char),
                c => {
                    self.i -= 1;
                    let rest = &self.s[self.i..];
                    let s = std::str::from_utf8(rest).map_err(|_| ParseError("invalid utf-8".into()))?;
                    let ch = s.chars().next().ok_or_else(|| ParseError("invalid utf-8".into()))?;
                    self.i += ch.len_utf8();
                    out.push(ch);
                    let _ = c;
                }
            }
        }
    }

    fn parse_array(&mut self) -> Result<Vec<Json>, ParseError> {
        if self.bump_byte()? != b'[' {
            return Err(ParseError("expected array".into()));
        }
        self.skip_ws();
        let mut out = Vec::new();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(out);
        }
        loop {
            out.push(self.parse_value()?);
            self.skip_ws();
            match self.bump_byte()? {
                b']' => return Ok(out),
                b',' => self.skip_ws(),
                c => return Err(ParseError(format!("expected comma or ], got {c}"))),
            }
        }
    }

    fn parse_object(&mut self) -> Result<Vec<(String, Json)>, ParseError> {
        if self.bump_byte()? != b'{' {
            return Err(ParseError("expected object".into()));
        }
        self.skip_ws();
        let mut out = Vec::new();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(out);
        }
        loop {
            self.skip_ws();
            let k = self.parse_string()?;
            self.skip_ws();
            if self.bump_byte()? != b':' {
                return Err(ParseError("expected colon".into()));
            }
            let v = self.parse_value()?;
            out.push((k, v));
            self.skip_ws();
            match self.bump_byte()? {
                b'}' => return Ok(out),
                b',' => self.skip_ws(),
                c => return Err(ParseError(format!("expected comma or }}, got {c}"))),
            }
        }
    }
}

pub fn escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 32 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_chat_payload() {
        let j = Json::parse(
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0.5}"#,
        )
        .unwrap();
        assert_eq!(j.get("max_tokens").and_then(|v| v.as_usize()), Some(8));
        let msgs = j.get("messages").and_then(|v| v.as_array()).unwrap();
        assert_eq!(msgs[0].get("content").and_then(|v| v.as_str()), Some("hi"));
    }

    #[test]
    fn rejects_junk() {
        assert!(Json::parse("{not json").is_err());
        assert!(Json::parse("[] trailing").is_err());
    }
}
