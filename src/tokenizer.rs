use anyhow::{Result, anyhow, ensure};
use minijinja::{Environment, context};
use serde_json::Value;
use std::{fs, path::Path};
use tokenizers::Tokenizer;

pub struct TextCodec {
    tokenizer: Tokenizer,
    template: Environment<'static>,
    pub chat_template: String,
}
impl TextCodec {
    #[cfg(test)]
    pub(crate) fn fixture() -> Self {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [("[UNK]".to_owned(), 0), ("hello".to_owned(), 1)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        let chat_template = "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}{{ tools_ts_str }} assistant".to_owned();
        let mut template = Environment::new();
        template
            .add_template_owned("chat", chat_template.clone())
            .unwrap();
        Self {
            tokenizer,
            template,
            chat_template,
        }
    }
    pub fn load(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let mut template = Environment::new();
        template.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        template.add_function(
            "raise_exception",
            |msg: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        let chat_template = fs::read_to_string(path.join("chat_template.jinja"))?;
        template.add_template_owned("chat", chat_template.clone())?;
        Ok(Self {
            tokenizer,
            template,
            chat_template,
        })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizing: {e}"))?
            .get_ids()
            .to_vec())
    }
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow!("decoding: {e}"))
    }
    pub fn decode_raw(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, false)
            .map_err(|e| anyhow!("decoding: {e}"))
    }
    /// Count generated token IDs that contribute text, excluding added special
    /// tokens such as role/end markers. This does not re-tokenize decoded text.
    pub fn count_text_tokens(&self, tokens: &[u32]) -> usize {
        let added = self
            .tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder();
        tokens
            .iter()
            .filter(|id| !added.get(id).is_some_and(|token| token.special))
            .count()
    }
    pub fn chat_prompt(&self, messages: &[Value]) -> Result<String> {
        self.chat_prompt_with_tools(messages, &[], true)
    }
    pub fn chat_prompt_with_tools(
        &self,
        messages: &[Value],
        tools: &[Value],
        add_generation_prompt: bool,
    ) -> Result<String> {
        ensure!(!messages.is_empty(), "messages must not be empty");
        for m in messages {
            ensure!(
                matches!(
                    m.get("role").and_then(Value::as_str),
                    Some("system" | "user" | "assistant" | "tool")
                ),
                "unsupported message role"
            );
        }
        // The checkpoint explicitly supports pre-rendered tool definitions. This
        // avoids Jinja's Python-specific tojson(ensure_ascii=False) argument.
        let tools_ts_str = tools
            .iter()
            .map(serde_json::to_string)
            .collect::<std::result::Result<Vec<_>, _>>()?
            .join("\n");
        Ok(self.template.get_template("chat")?.render(context! { messages => messages, add_generation_prompt => add_generation_prompt, tools => tools, tools_ts_str => tools_ts_str })?)
    }
}
