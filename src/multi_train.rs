use crate::{
    bpe_tokenizer::{build_tokenizer, train},
    embedding_layer::EmbeddingLayer,
    linear_layer::{LinearLayer, cross_entropy_derivative, softmax},
    self_attatention_layer::SelfAttentionLayer,
    sliding_window::sliding_windows,
    tokenizer::Tokenizer,
};
use rand::{SeedableRng, rngs::StdRng};

/// Trener en BPE-tokenizer på teksten og bygger vocabulary fra resultatet.
pub fn build_tokenizer_from_text(
    training_text: &str,
    target_vocab_size: u32,
) -> crate::bpe_tokenizer::BpeTokenizer {
    let tokenizer_state = train(training_text.as_bytes(), target_vocab_size);
    build_tokenizer(tokenizer_state)
}

/// Setter sammen modellens tre trenbare deler.
///
/// Modellen består av embedding, ett self-attention-lag og ett output-lag.
/// Seed-en gjør de tilfeldige startvektene reproduserbare.
pub fn build_model<T: Tokenizer>(
    tokenizer: T,
    d_model: usize,
    seq_len: usize,
    seed: u64,
) -> Model<T> {
    let vocab_size = tokenizer.vocab_size();
    let mut rng = StdRng::seed_from_u64(seed);
    let embedding = EmbeddingLayer::new(vocab_size, d_model, &mut rng);
    let mut attention_rng = StdRng::seed_from_u64(seed.wrapping_add(1));
    let attention = SelfAttentionLayer::new(d_model, &mut attention_rng);
    let linear = LinearLayer::new(vocab_size, d_model, &mut rng);

    Model {
        tokenizer,
        embedding,
        attention,
        linear,
        d_model,
        seq_len,
        vocab_size,
    }
}

/// Trener modellen til å predikere neste token i treningsdataene.
///
/// Hvert vindu gir modellen `seq_len` tokens som kontekst og neste token som
/// fasit. Feilen sendes bakover gjennom output, attention og embedding før
/// SGD oppdaterer alle vektene.
pub fn train_model<T: Tokenizer>(
    model: &mut Model<T>,
    training_data: &str,
    learning_rate: f32,
    epochs: usize,
) {
    let tokens: Vec<u32> = model.tokenizer.encode(training_data);

    for _ in 0..epochs {
        for (sequence, target) in sliding_windows(&tokens, model.seq_len) {
            let embedded_sequence: Vec<f32> = sequence
                .iter()
                .flat_map(|&t| model.embedding.forward(t).to_vec())
                .collect();

            // 1. Forward Pass
            let context_sequence = model.attention.forward(&embedded_sequence, model.seq_len);
            let last_token_idx = (model.seq_len - 1) * model.d_model;
            let last_token_vector =
                &context_sequence[last_token_idx..(last_token_idx + model.d_model)];

            let predictions = model.linear.forward(last_token_vector);
            let targets = create_target(&predictions, target as usize);

            let gradients = cross_entropy_derivative(&predictions, &targets);

            // 2. Backward Pass (Linear -> Attention -> Embedding)
            let d_last_token = model.linear.backward(last_token_vector, &gradients);

            let mut d_context_sequence = vec![0.0; model.seq_len * model.d_model];
            d_context_sequence[last_token_idx..].copy_from_slice(&d_last_token);

            // Attention pulls its own internal cache now
            let d_embedded = model.attention.backward(&d_context_sequence);
            model.embedding.backward(sequence, &d_embedded);

            // 3. Update Weights
            model.linear.update_weights(learning_rate);
            model.attention.update_weights(learning_rate);
            model.embedding.update_weights(learning_rate);
        }
    }
}

/// Genererer tekst autoregressivt, ett token om gangen.
///
/// Etter hver prediksjon legges det valgte tokenet til konteksten. Modellen
/// bruker dermed sin egen output som input til neste runde.
pub fn predict_tokens<T: Tokenizer>(
    model: &mut Model<T>,
    prompt: &str,
    max_new_tokens: usize,
) -> String {
    generate_tokens(model, prompt, max_new_tokens, 0, 0).0
}

/// Genererer tekst og samler de høyest rangerte tokenene for de første stegene.
pub fn predict_tokens_with_trace<T: Tokenizer>(
    model: &mut Model<T>,
    prompt: &str,
    max_new_tokens: usize,
    trace_steps: usize,
    top_k: usize,
) -> (String, Vec<PredictionStep>) {
    generate_tokens(model, prompt, max_new_tokens, trace_steps, top_k)
}

fn generate_tokens<T: Tokenizer>(
    model: &mut Model<T>,
    prompt: &str,
    max_new_tokens: usize,
    trace_steps: usize,
    top_k: usize,
) -> (String, Vec<PredictionStep>) {
    let mut current_tokens = model.tokenizer.encode(prompt);
    let mut trace = Vec::with_capacity(trace_steps.min(max_new_tokens));

    if current_tokens.is_empty() {
        current_tokens.push(0);
    }

    for _ in 0..max_new_tokens {
        let logits = forward(model, &current_tokens);
        let next_token_id = argmax(&logits);

        if trace.len() < trace_steps {
            trace.push(PredictionStep {
                context_tokens: current_tokens.clone(),
                candidates: top_predictions(&logits, top_k),
                selected_token_id: next_token_id,
            });
        }
        current_tokens.push(next_token_id);
    }
    (model.tokenizer.decode(&current_tokens), trace)
}

/// Kjører ett forward pass og returnerer logits for neste token.
///
/// Bare outputen ved siste posisjon brukes, fordi oppgaven er å fortsette
/// teksten etter hele konteksten.
pub fn forward<T: Tokenizer>(model: &mut Model<T>, tokens: &[u32]) -> Vec<f32> {
    let seq_len = tokens.len();
    let mut embedded_sequence = Vec::with_capacity(seq_len * model.embedding.d_model);
    for &token_id in tokens {
        let token_vector = model.embedding.forward(token_id);
        embedded_sequence.extend_from_slice(token_vector);
    }
    let context_aware_sequence = model.attention.forward(&embedded_sequence, seq_len);
    let start_idx = (seq_len - 1) * model.attention.d_model;
    let last_token_vector = &context_aware_sequence[start_idx..];
    model.linear.forward(last_token_vector)
}

/// Lager en one-hot-vektor med fasit-tokenet markert som 1.
fn create_target(predictions: &[f32], target_index: usize) -> Vec<f32> {
    let mut target = vec![0.0; predictions.len()];
    target[target_index] = 1.0;
    target
}

/// Velger tokenet med høyest logit.
///
/// Dette gir deterministisk generering. Virkelige LLM-er sampler ofte fra
/// sannsynlighetsfordelingen for å få mer varierte svar.
fn argmax(logits: &[f32]) -> u32 {
    let mut best_index = 0;
    let mut best_score = f32::NEG_INFINITY;
    for (token_index, &score) in logits.iter().enumerate() {
        if score > best_score {
            best_score = score;
            best_index = token_index;
        }
    }
    best_index as u32
}

/// Ett observert genereringssteg med context, kandidater og valgt token.
#[derive(Debug, PartialEq)]
pub struct PredictionStep {
    pub context_tokens: Vec<u32>,
    pub candidates: Vec<TokenPrediction>,
    pub selected_token_id: u32,
}

/// Ett token og modellens softmax-sannsynlighet for neste posisjon.
#[derive(Debug, PartialEq)]
pub struct TokenPrediction {
    pub token_id: u32,
    pub probability: f32,
}

/// Rangerer de mest sannsynlige neste tokenene.
pub fn top_predictions(logits: &[f32], limit: usize) -> Vec<TokenPrediction> {
    let probabilities = softmax(logits);
    let mut token_ids: Vec<_> = (0..logits.len()).collect();
    token_ids.sort_by(|&left, &right| {
        logits[right]
            .total_cmp(&logits[left])
            .then_with(|| left.cmp(&right))
    });
    token_ids.truncate(limit.min(token_ids.len()));
    token_ids
        .into_iter()
        .map(|token_id| TokenPrediction {
            token_id: token_id as u32,
            probability: probabilities[token_id],
        })
        .collect()
}

pub struct Model<T: Tokenizer> {
    pub tokenizer: T,
    pub embedding: EmbeddingLayer,
    pub attention: SelfAttentionLayer,
    pub linear: LinearLayer,
    pub d_model: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParameterCounts {
    pub embedding: usize,
    pub attention: usize,
    pub output: usize,
}

impl ParameterCounts {
    /// Summerer antall trenbare parametere i alle modellagene.
    pub fn total(&self) -> usize {
        self.embedding + self.attention + self.output
    }
}

impl<T: Tokenizer> Model<T> {
    /// Teller de faktiske trenbare vektene i hvert lag.
    ///
    /// Gradienter og midlertidige verdier teller ikke som modellparametere.
    pub fn parameter_counts(&self) -> ParameterCounts {
        ParameterCounts {
            embedding: self.embedding.weights.len(),
            attention: self.attention.w_q.len()
                + self.attention.w_k.len()
                + self.attention.w_v.len(),
            output: self.linear.weights.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_layer::cross_entropy_loss;
    use crate::word_tokenizer::WordTokenizer;

    #[test]
    fn same_seed_reproduces_training_and_prediction() {
        let training_data = "one two three. one two three. one two three.";
        let mut first = build_model(WordTokenizer::build(training_data), 4, 2, 42);
        let mut second = build_model(WordTokenizer::build(training_data), 4, 2, 42);

        train_model(&mut first, training_data, 0.01, 10);
        train_model(&mut second, training_data, 0.01, 10);

        assert_eq!(first.embedding.weights, second.embedding.weights);
        assert_eq!(first.attention.w_q, second.attention.w_q);
        assert_eq!(first.attention.w_k, second.attention.w_k);
        assert_eq!(first.attention.w_v, second.attention.w_v);
        assert_eq!(first.linear.weights, second.linear.weights);
        assert_eq!(
            predict_tokens(&mut first, "one two", 3),
            predict_tokens(&mut second, "one two", 3)
        );
    }

    #[test]
    fn different_seeds_produce_different_initial_weights() {
        let first = build_model(WordTokenizer::build("one two three"), 4, 2, 1);
        let second = build_model(WordTokenizer::build("one two three"), 4, 2, 2);

        assert_ne!(first.embedding.weights, second.embedding.weights);
        assert_ne!(first.attention.w_q, second.attention.w_q);
        assert_ne!(first.attention.w_k, second.attention.w_k);
        assert_ne!(first.attention.w_v, second.attention.w_v);
        assert_ne!(first.attention.w_q, first.attention.w_k);
        assert_ne!(first.attention.w_q, first.attention.w_v);
        assert_ne!(first.attention.w_k, first.attention.w_v);
        assert_ne!(first.linear.weights, second.linear.weights);
    }

    #[test]
    fn counts_all_trainable_model_parameters() {
        let model = build_model(WordTokenizer::build("one two three"), 4, 2, 42);

        let counts = model.parameter_counts();

        assert_eq!(
            counts,
            ParameterCounts {
                embedding: 12,
                attention: 48,
                output: 12,
            }
        );
        assert_eq!(counts.total(), 72);
    }

    #[test]
    fn top_predictions_sorts_probabilities_and_preserves_total_mass() {
        let predictions = top_predictions(&[0.0, 3.0_f32.ln(), 0.0], 3);

        assert_eq!(
            predictions
                .iter()
                .map(|prediction| prediction.token_id)
                .collect::<Vec<_>>(),
            vec![1, 0, 2]
        );
        assert!((predictions.iter().map(|item| item.probability).sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn traced_prediction_records_context_candidates_and_argmax_choice() {
        let training_data = "one two three. one two three.";
        let mut model = build_model(WordTokenizer::build(training_data), 4, 2, 42);

        let (_, trace) = predict_tokens_with_trace(&mut model, "one two", 2, 1, 3);

        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].context_tokens, vec![0, 1]);
        assert_eq!(trace[0].selected_token_id, trace[0].candidates[0].token_id);
        let shown_probability = trace[0]
            .candidates
            .iter()
            .map(|candidate| candidate.probability)
            .sum::<f32>();
        assert!(shown_probability <= 1.0 + f32::EPSILON);
    }

    #[test]
    fn training_reduces_next_token_loss() {
        let training_data = "one two three. one two three. one two three.";
        let mut model = build_model(WordTokenizer::build(training_data), 4, 2, 42);
        let context = model.tokenizer.encode("one two");
        let target_id = model.tokenizer.encode("three")[0] as usize;

        let loss_before = target_loss(&mut model, &context, target_id);
        train_model(&mut model, training_data, 0.01, 100);
        let loss_after = target_loss(&mut model, &context, target_id);

        assert!(
            loss_after < loss_before,
            "expected training to reduce loss from {loss_before}, got {loss_after}"
        );
    }

    #[test]
    fn municipality_demo_completes_county_and_period() {
        let training_data = include_str!("../kommuner_demo.txt");
        let mut model = build_model(WordTokenizer::build(training_data), 8, 4, 42);

        train_model(&mut model, training_data, 0.05, 100);
        let prediction = predict_tokens(&mut model, "bergen ligger i", 3);

        assert_eq!(prediction, "bergen ligger i vestland fylke.");
    }

    fn target_loss<T: Tokenizer>(model: &mut Model<T>, context: &[u32], target_id: usize) -> f32 {
        let logits = forward(model, context);
        let target = create_target(&logits, target_id);
        cross_entropy_loss(&logits, &target)
    }
}
