use crate::{
    bpe_tokenizer::{build_tokenizer, train},
    embedding_layer::EmbeddingLayer,
    linear_layer::{LinearLayer, cross_entropy_derivative},
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
    let attention = SelfAttentionLayer::new(d_model);
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
    let mut current_tokens = model.tokenizer.encode(prompt);

    if current_tokens.is_empty() {
        current_tokens.push(0);
    }

    for _ in 0..max_new_tokens {
        // Get embedded sequences
        let mut embedded_sequence =
            Vec::with_capacity(current_tokens.len() * model.embedding.d_model);
        for &token_id in &current_tokens {
            let token_vector = model.embedding.forward(token_id);
            embedded_sequence.extend_from_slice(token_vector);
        }

        let context_aware = model
            .attention
            .forward(&embedded_sequence, current_tokens.len());
        let start_idx = (current_tokens.len() - 1) * model.d_model;
        let last_token_vector = &context_aware[start_idx..start_idx + model.d_model];

        let logits = model.linear.forward(last_token_vector);
        let next_token_id = argmax(&logits);

        current_tokens.push(next_token_id);
    }
    model.tokenizer.decode(&current_tokens)
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
}
