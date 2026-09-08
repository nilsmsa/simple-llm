use std::{
    fs::File,
    io::{self, BufReader, Read},
};

use crate::{
    arguments::{Args, TokenizerKind},
    tokenizer::Tokenizer,
    word_tokenizer::WordTokenizer,
};

pub mod arguments;
pub mod bpe_tokenizer;
pub mod embedding_layer;
pub mod linear_layer;
pub mod multi_train;
pub mod self_attatention_layer;
pub mod sliding_window;
pub mod tokenizer;
pub mod toy_llm;
pub mod word_tokenizer;

fn main() {
    multi_file_demo();
}

/// Leser treningsdata, velger tokenizer og starter hele demoen.
///
/// Dette er koblingen mellom CLI-et og modellkoden. Selve LLM-stegene ligger i
/// `multi_train`.
fn multi_file_demo() {
    let args = arguments::parse_args();

    println!("Training files: {}", args.file_paths.len());
    println!("Tokenizer: {}", args.tokenizer);
    println!("Seed: {}", args.seed);
    println!("Prompt: '{}'", args.prompt);
    println!();

    let mut training_text = String::new();
    files_to_reader(&args.file_paths)
        .read_to_string(&mut training_text)
        .expect("Failed to read files");

    match args.tokenizer {
        TokenizerKind::Word => {
            train_and_predict(WordTokenizer::build(&training_text), &args, &training_text)
        }
        TokenizerKind::Bpe => train_and_predict(
            multi_train::build_tokenizer_from_text(&training_text, args.vocab_size),
            &args,
            &training_text,
        ),
    }
}

/// Bygger modellen, trener den og genererer 50 nye tokens fra prompten.
fn train_and_predict<T: Tokenizer>(tokenizer: T, args: &Args, training_text: &str) {
    let mut model = multi_train::build_model(tokenizer, args.d_model, args.seq_len, args.seed);
    println!("Vocab size: {}", model.vocab_size);
    let parameters = model.parameter_counts();
    println!("Trainable parameters: {}", parameters.total());
    println!("  Embedding: {}", parameters.embedding);
    println!("  Attention (Q, K, V): {}", parameters.attention);
    println!("  Output: {}", parameters.output);
    multi_train::train_model(&mut model, training_text, args.learning_rate, args.epochs);
    println!("Prompt: '{}'", args.prompt);
    println!(
        "Predicted: '{}'",
        multi_train::predict_tokens(&mut model, &args.prompt, 50)
    );
}

/// Kobler flere treningsfiler sammen til én sammenhengende tekststrøm.
fn files_to_reader(training_texts: &[String]) -> Box<dyn Read + 'static> {
    let training_reader: Box<dyn Read> =
        training_texts
            .iter()
            .fold(Box::new(io::empty()), |acc, reader| {
                let file = File::open(reader)
                    .unwrap_or_else(|err| panic!("Failed to open file: {:?} => {:?}", reader, err));
                Box::new(acc.chain(BufReader::new(file)))
            });
    training_reader
}
