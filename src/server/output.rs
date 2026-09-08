//! Incremental interpretation of committed LLaDA blocks. Nothing from a live
//! refinement is exposed as text or tool arguments.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const SECTION: &str = "<|tool_calls_section_begin|>";
const SECTION_END: &str = "<|tool_calls_section_end|>";
const CALL: &str = "<|tool_call_begin|>";
const ARG: &str = "<|tool_call_argument_begin|>";
const END: &str = "<|tool_call_end|>";
const MARKERS: &[&str] = &[
    SECTION,
    SECTION_END,
    CALL,
    ARG,
    END,
    "<|role_end|>",
    "<|endoftext|>",
];

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Function,
}
#[derive(Clone, Default, Debug, PartialEq)]
pub struct Assistant {
    pub content: String,
    pub calls: Vec<ToolCall>,
}

/// Hold back a suffix that could become a stop string or a protocol marker.
pub fn safe_end(text: &str, needles: &[&str]) -> usize {
    let hold = needles
        .iter()
        .flat_map(|s| {
            s.char_indices()
                .map(|(i, _)| i)
                .skip(1)
                .map(move |n| &s[..n])
        })
        .filter(|prefix| text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0);
    text.len() - hold
}

pub fn parse(
    raw: &str,
    id: &str,
    allowed: &[String],
    final_output: bool,
    complete_tools: bool,
) -> Result<Assistant> {
    let raw = if final_output {
        raw
    } else {
        raw.trim_end_matches('\u{fffd}')
    };
    let mut rest = raw;
    let mut out = Assistant::default();
    let mut section = false;
    while !rest.is_empty() {
        if let Some(s) = rest.strip_prefix(SECTION) {
            ensure!(!section, "nested tool call section");
            section = true;
            rest = s;
            continue;
        }
        if let Some(s) = rest.strip_prefix(SECTION_END) {
            ensure!(section, "unexpected tool section end");
            section = false;
            rest = s;
            continue;
        }
        if rest.starts_with("<|role_end|>") || rest.starts_with("<|endoftext|>") {
            rest = "";
            continue;
        }
        if let Some(s) = rest.strip_prefix(CALL) {
            ensure!(section, "tool call outside a tool section");
            let Some((header, args)) = s.split_once(ARG) else {
                ensure!(!complete_tools, "incomplete tool call header");
                break;
            };
            let name = header
                .strip_prefix("functions.")
                .and_then(|s| s.rsplit_once(':'))
                .map(|(name, _)| name)
                .ok_or_else(|| anyhow::anyhow!("invalid tool call header"))?;
            ensure!(
                allowed.iter().any(|n| n == name),
                "model called an undeclared function: {name}"
            );
            let (arguments, tail) = if let Some((args, tail)) = args.split_once(END) {
                let value: Value = serde_json::from_str(args)
                    .map_err(|e| anyhow::anyhow!("invalid tool arguments: {e}"))?;
                ensure!(value.is_object(), "tool arguments must be a JSON object");
                (args, Some(tail))
            } else {
                ensure!(!complete_tools, "incomplete tool arguments");
                (&args[..safe_end(args, &[END])], None)
            };
            out.calls.push(ToolCall {
                id: format!("call_{id}_{}", out.calls.len()),
                r#type: "function".into(),
                function: Function {
                    name: name.into(),
                    arguments: arguments.into(),
                },
            });
            if let Some(tail) = tail {
                rest = tail;
            } else {
                break;
            }
            continue;
        }
        let next = MARKERS
            .iter()
            .filter_map(|m| rest.find(m))
            .min()
            .unwrap_or(rest.len());
        if next == 0 {
            anyhow::bail!("unexpected tool control marker");
        }
        let end = if next == rest.len() && !final_output {
            safe_end(rest, MARKERS)
        } else {
            next
        };
        if section {
            ensure!(
                rest[..end].trim().is_empty(),
                "unexpected content in tool section"
            );
        } else {
            out.content.push_str(&rest[..end]);
        }
        if end < next || next == rest.len() {
            break;
        }
        rest = &rest[next..];
    }
    ensure!(!complete_tools || !section, "incomplete tool section");
    Ok(out)
}

pub fn delta(previous: &Assistant, current: &Assistant) -> Result<Value> {
    ensure!(
        current.content.starts_with(&previous.content),
        "committed text changed during streaming"
    );
    ensure!(
        current.calls.len() >= previous.calls.len(),
        "committed tool calls changed"
    );
    let mut d = json!({});
    let text = &current.content[previous.content.len()..];
    if !text.is_empty() {
        d["content"] = json!(text);
    }
    let mut calls = Vec::new();
    for (i, call) in current.calls.iter().enumerate() {
        if let Some(old) = previous.calls.get(i) {
            ensure!(
                old.id == call.id
                    && old.function.name == call.function.name
                    && call.function.arguments.starts_with(&old.function.arguments),
                "committed tool call changed"
            );
            let args = &call.function.arguments[old.function.arguments.len()..];
            if !args.is_empty() {
                calls.push(json!({"index":i,"function":{"arguments":args}}));
            }
        } else {
            calls.push(json!({"index":i,"id":call.id,"type":"function","function":call.function}));
        }
    }
    if !calls.is_empty() {
        d["tool_calls"] = json!(calls);
    }
    Ok(d)
}

impl Assistant {
    pub fn message(&self) -> Value {
        let mut message = json!({"role":"assistant","content":if self.content.is_empty() && !self.calls.is_empty() { Value::Null } else { json!(self.content) }});
        if !self.calls.is_empty() {
            message["tool_calls"] = json!(self.calls);
        }
        message
    }
    pub fn stop(&mut self, stops: &[String], final_output: bool) -> bool {
        if let Some(at) = stops.iter().filter_map(|s| self.content.find(s)).min() {
            self.content.truncate(at);
            return true;
        }
        if !final_output {
            let end = safe_end(
                &self.content,
                &stops.iter().map(String::as_str).collect::<Vec<_>>(),
            );
            self.content.truncate(end);
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tool_markers_and_json_survive_every_stream_boundary() {
        let raw = format!(
            "Checking. {SECTION}{CALL}functions.weather:0{ARG}{{\"city\":\"Montréal\"}}{END}{CALL}functions.weather:1{ARG}{{\"city\":\"東京\"}}{END}{SECTION_END}<|role_end|><|endoftext|>"
        );
        let mut previous = Assistant::default();
        let mut content = String::new();
        let mut args = vec![String::new(); 2];
        for (end, _) in raw
            .char_indices()
            .skip(1)
            .chain(std::iter::once((raw.len(), ' ')))
        {
            let current = parse(
                &raw[..end],
                "test",
                &["weather".into()],
                end == raw.len(),
                end == raw.len(),
            )
            .unwrap();
            let d = delta(&previous, &current).unwrap();
            content.push_str(d["content"].as_str().unwrap_or(""));
            if let Some(calls) = d["tool_calls"].as_array() {
                for c in calls {
                    args[c["index"].as_u64().unwrap() as usize]
                        .push_str(c["function"]["arguments"].as_str().unwrap_or(""));
                }
            }
            previous = current;
        }
        assert_eq!(content, "Checking. ");
        assert_eq!(args, ["{\"city\":\"Montréal\"}", "{\"city\":\"東京\"}"]);
    }
    #[test]
    fn stop_prefixes_are_held_until_they_resolve() {
        let stops = vec!["STOP".into()];
        let mut a = Assistant {
            content: "hello ST".into(),
            ..Assistant::default()
        };
        assert!(!a.stop(&stops, false));
        assert_eq!(a.content, "hello ");
        a.content = "hello STOP more".into();
        assert!(a.stop(&stops, false));
        assert_eq!(a.content, "hello ");
        a.content = "hello ST".into();
        assert!(!a.stop(&stops, true));
        assert_eq!(a.content, "hello ST");
    }
    #[test]
    fn length_cutoff_flushes_text_without_requiring_closed_tools() {
        let mut text = parse("hello ST", "t", &[], true, false).unwrap();
        assert!(!text.stop(&["STOP".into()], true));
        assert_eq!(text.content, "hello ST");
        let raw = format!("{SECTION}{CALL}functions.ok:0{ARG}{{");
        let partial = parse(&raw, "t", &["ok".into()], true, false).unwrap();
        assert_eq!(partial.calls[0].function.arguments, "{");
        assert!(parse(&raw, "t", &["ok".into()], true, true).is_err());
    }
    #[test]
    fn invalid_or_undeclared_calls_are_errors() {
        for raw in [
            format!("{SECTION}{CALL}functions.bad:0{ARG}{{}}{END}{SECTION_END}"),
            format!("{SECTION}{CALL}functions.ok:0{ARG}not json{END}{SECTION_END}"),
            format!("{SECTION}{CALL}functions.ok:0{ARG}{{"),
        ] {
            assert!(parse(&raw, "t", &["ok".into()], true, true).is_err());
        }
    }
}
