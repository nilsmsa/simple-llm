use std::{
    fs::File,
    io::{self, BufReader, Read},
};

use simple_llm::{
    arguments::{self, Args, TokenizerKind},
    multi_train,
    tokenizer::Tokenizer,
    word_tokenizer::WordTokenizer,
};

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
    if args.trace {
        let (predicted, trace) =
            multi_train::predict_tokens_with_trace(&mut model, &args.prompt, 50, 3, 3);
        print_prediction_trace(&model, &trace);
        println!("Predicted: '{predicted}'");
    } else {
        println!(
            "Predicted: '{}'",
            multi_train::predict_tokens(&mut model, &args.prompt, 50)
        );
    }
}

/// Skriver kandidatene uten å fremstille token-sannsynlighet som sannhet.
fn print_prediction_trace<T: Tokenizer>(
    model: &multi_train::Model<T>,
    trace: &[multi_train::PredictionStep],
) {
    println!("Prediction trace:");
    for (step_index, step) in trace.iter().enumerate() {
        println!(
            "Step {} context: '{}'",
            step_index + 1,
            model.tokenizer.decode(&step.context_tokens)
        );
        let shown_probability: f32 = step
            .candidates
            .iter()
            .map(|candidate| candidate.probability)
            .sum();
        for candidate in &step.candidates {
            println!(
                "  '{}': {:.6}",
                model.tokenizer.decode(&[candidate.token_id]),
                candidate.probability
            );
        }
        if step.candidates.len() < model.vocab_size {
            println!("  other tokens: {:.6}", (1.0 - shown_probability).max(0.0));
        }
        println!(
            "  selected with argmax: '{}'",
            model.tokenizer.decode(&[step.selected_token_id])
        );
    }
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
