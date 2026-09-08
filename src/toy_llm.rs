use crate::{
    embedding_layer::EmbeddingLayer, linear_layer::LinearLayer,
    self_attatention_layer::SelfAttentionLayer, tokenizer::Tokenizer,
};
use rand::{SeedableRng, rngs::StdRng};

pub struct ToyLLM<T: Tokenizer> {
    pub tokenizer: T,
    pub embedding: EmbeddingLayer,
    pub attention: SelfAttentionLayer,
    pub linear: LinearLayer,
}

impl<T: Tokenizer> ToyLLM<T> {
    /// Bygger en liten modell med reproduserbare, tilfeldige startvekter.
    pub fn new(tokenizer: T, d_model: usize, seed: u64) -> Self {
        let vocab_size = tokenizer.vocab_size();
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            tokenizer,
            embedding: EmbeddingLayer::new(vocab_size, d_model, &mut rng),
            attention: SelfAttentionLayer::new(d_model),
            linear: LinearLayer::new(vocab_size, d_model, &mut rng),
        }
    }

    /// Gjør tokens om til logits for tokenet som skal komme etter konteksten.
    ///
    /// Dataflyten er embedding → self-attention → output-lag.
    pub fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len();
        let mut embedded_sequence = Vec::with_capacity(seq_len * self.embedding.d_model);
        for &token_id in tokens {
            let token_vector = self.embedding.forward(token_id);
            embedded_sequence.extend_from_slice(token_vector);
        }
        let context_aware_sequence = self.attention.forward(&embedded_sequence, seq_len);
        let start_idx = (seq_len - 1) * self.attention.d_model;
        let last_token_vector = &context_aware_sequence[start_idx..];
        self.linear.forward(last_token_vector)
    }

    /// Fortsetter prompten ved å legge til modellens beste token gjentatte ganger.
    ///
    /// Hvert generert token blir del av konteksten for neste prediksjon.
    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> String {
        let mut current_tokens = self.tokenizer.encode(prompt);
        for _ in 0..max_new_tokens {
            let logits = self.forward(&current_tokens);
            let next_token_id = argmax(&logits);
            current_tokens.push(next_token_id);
        }
        self.tokenizer.decode(&current_tokens)
    }
}

/// Finner token-ID-en med høyest logit.
pub fn argmax(logits: &[f32]) -> u32 {
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
