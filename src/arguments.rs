use std::{fmt, str::FromStr};

pub const DEFAULT_SEED: u64 = 42;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenizerKind {
    #[default]
    Word,
    Bpe,
}

impl fmt::Display for TokenizerKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Word => formatter.write_str("word"),
            Self::Bpe => formatter.write_str("bpe"),
        }
    }
}

impl FromStr for TokenizerKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "word" => Ok(Self::Word),
            "bpe" => Ok(Self::Bpe),
            _ => Err(format!(
                "Invalid tokenizer: {value}. Expected 'word' or 'bpe'"
            )),
        }
    }
}

pub struct Args {
    pub tokenizer: TokenizerKind,
    pub vocab_size: u32,
    pub epochs: usize,
    pub seq_len: usize,
    pub d_model: usize,
    pub learning_rate: f32,
    pub seed: u64,
    pub file_paths: Vec<String>,
    pub prompt: String,
}

/// Leser kommandolinjevalg og skiller treningsfiler fra prompten.
pub fn parse_args() -> Args {
    let mut args: Vec<String> = std::env::args().collect();
    args.remove(0);

    if args.is_empty() {
        eprintln!("Usage: simple-llm [options] <file1> [file2] ... [prompt]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  -tokenizer=TYPE  Tokenizer: word or bpe (default: word)");
        eprintln!("  -vocab=N        Target vocabulary size (default: 256)");
        eprintln!("  -epochs=N       Training epochs per repetition (default: 100)");
        eprintln!("  -seq-len=N      Context window size (default: 3)");
        eprintln!("  -d-model=N      Embedding dimension (default: 8)");
        eprintln!("  -learning-rate=N Learning rate (default: 0.001)");
        eprintln!("  -seed=N         Random seed (default: {DEFAULT_SEED})");
        eprintln!("                  Larger d_model needs smaller learning rate");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  simple-llm training.txt \"hello\"");
        eprintln!("  simple-llm -tokenizer=bpe -vocab=300 -seq-len=20 training.txt \"hello\"");
        eprintln!("  simple-llm -count=3 -vocab=32 file1.txt file2.txt \"a b\"");
        std::process::exit(1);
    }

    let mut tokens = args;

    let mut tokenizer = TokenizerKind::default();
    let mut vocab_size: u32 = 256;
    let mut epochs: usize = 100;
    let mut seq_len: usize = 3;
    let mut d_model: usize = 8;
    let mut learning_rate: f32 = 0.001;
    let mut seed: u64 = DEFAULT_SEED;

    let mut file_tokens = Vec::new();

    while !tokens.is_empty() {
        let token = tokens.remove(0);

        match token.as_str() {
            s if s.starts_with("-tokenizer=") => {
                tokenizer = s
                    .strip_prefix("-tokenizer=")
                    .expect("prefix was checked")
                    .parse()
                    .unwrap_or_else(|error| {
                        eprintln!("{error}");
                        std::process::exit(1);
                    });
            }
            s if s.starts_with("-vocab=") => {
                vocab_size = s
                    .strip_prefix("-vocab=")
                    .unwrap()
                    .parse::<u32>()
                    .unwrap_or(256);
            }
            s if s.starts_with("-epochs=") => {
                epochs = s
                    .strip_prefix("-epochs=")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap_or(100);
            }
            s if s.starts_with("-seq-len=") => {
                seq_len = s
                    .strip_prefix("-seq-len=")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap_or(3);
            }
            s if s.starts_with("-d-model=") => {
                d_model = s
                    .strip_prefix("-d-model=")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap_or(8);
            }
            s if s.starts_with("-learning-rate=") => {
                learning_rate = s
                    .strip_prefix("-learning-rate=")
                    .unwrap()
                    .parse::<f32>()
                    .unwrap_or(0.001);
            }
            s if s.starts_with("-seed=") => {
                seed = s
                    .strip_prefix("-seed=")
                    .unwrap()
                    .parse::<u64>()
                    .unwrap_or(DEFAULT_SEED);
            }
            s if s.starts_with('-') => {
                eprintln!("Unknown option: {}", token);
                std::process::exit(1);
            }
            _ => {
                file_tokens.push(token);
            }
        }
    }

    if file_tokens.is_empty() {
        eprintln!("Error: No training files specified");
        std::process::exit(1);
    }

    let prompt = file_tokens.pop().unwrap_or_else(|| String::from("a b"));

    let file_paths: Vec<String> = file_tokens;

    Args {
        tokenizer,
        vocab_size,
        epochs,
        seq_len,
        d_model,
        learning_rate,
        seed,
        file_paths,
        prompt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_kind_defaults_to_word() {
        assert_eq!(TokenizerKind::default(), TokenizerKind::Word);
    }

    #[test]
    fn default_seed_is_stable() {
        assert_eq!(DEFAULT_SEED, 42);
    }

    #[test]
    fn tokenizer_kind_parses_supported_cli_values() {
        assert_eq!("word".parse(), Ok(TokenizerKind::Word));
        assert_eq!("bpe".parse(), Ok(TokenizerKind::Bpe));
    }

    #[test]
    fn tokenizer_kind_rejects_unsupported_cli_values() {
        let error = "characters"
            .parse::<TokenizerKind>()
            .expect_err("unsupported tokenizer should fail");

        assert_eq!(
            error,
            "Invalid tokenizer: characters. Expected 'word' or 'bpe'"
        );
    }
}
