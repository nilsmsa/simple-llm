use rand::Rng;
use rand_distr::{Distribution, Normal};

pub struct EmbeddingLayer {
    #[allow(dead_code)]
    pub vocab_size: usize,
    pub d_model: usize,
    pub weights: Vec<f32>,
    gradients: Vec<f32>,
}

impl EmbeddingLayer {
    /// Oppretter én trenbar embedding-vektor for hvert token i vocabulary.
    /// Vektorene starter tilfeldig og justeres under trening slik at modellen
    /// blir bedre til å forutsi neste token ut fra konteksten.
    pub fn new<R: Rng + ?Sized>(vocab_size: usize, d_model: usize, rng: &mut R) -> Self {
        let normal = Normal::new(0.0, 0.1).expect("Should be ok?");
        let total_weights = vocab_size * d_model;
        let mut weights = Vec::with_capacity(total_weights);
        let mut gradients = Vec::with_capacity(total_weights);
        for _ in 0..total_weights {
            weights.push(normal.sample(rng) as f32);
            gradients.push(0.0);
        }
        Self {
            vocab_size,
            d_model,
            weights,
            gradients,
        }
    }

    /// Slår opp den trenbare vektoren som representerer ett token.
    pub fn forward(&self, token_id: u32) -> &[f32] {
        let start = (token_id as usize) * self.d_model;
        let end = start + self.d_model;
        &self.weights[start..end]
    }

    /// Samler gradientene som viser hvordan hver brukt embedding bør endres.
    /// Hvis et token forekommer flere ganger, summeres bidragene før vektene
    /// oppdateres.
    pub fn backward(&mut self, sequence: &[u32], grad_output: &[f32]) {
        for (seq_idx, &token_id) in sequence.iter().enumerate() {
            let token_id = token_id as usize;
            let grad_start_idx = seq_idx * self.d_model;
            let weight_start_idx = token_id * self.d_model;
            for feature_index in 0..self.d_model {
                let gradient = grad_output[grad_start_idx + feature_index];
                self.gradients[weight_start_idx + feature_index] += gradient;
            }
        }
    }

    /// Flytter embedding-vektene mot lavere loss og nullstiller gradientene.
    pub fn update_weights(&mut self, learning_rate: f32) {
        for weight_index in 0..self.weights.len() {
            self.weights[weight_index] -= learning_rate * self.gradients[weight_index];
            self.gradients[weight_index] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn new_creates_a_usable_embedding_table_with_the_expected_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut layer = EmbeddingLayer::new(4, 3, &mut rng);

        assert_eq!(layer.vocab_size, 4);
        assert_eq!(layer.d_model, 3);
        assert_eq!(layer.weights.len(), 12);
        assert!(layer.weights.iter().all(|weight| weight.is_finite()));

        let initial_weights = layer.weights.clone();
        layer.update_weights(0.1);
        assert_eq!(layer.weights, initial_weights);
    }

    #[test]
    fn forward_returns_the_embedding_for_the_requested_token() {
        let layer = layer_with_weights(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(layer.forward(0), &[1.0, 2.0]);
        assert_eq!(layer.forward(1), &[3.0, 4.0]);
        assert_eq!(layer.forward(2), &[5.0, 6.0]);
    }

    #[test]
    fn backward_places_each_gradient_in_the_corresponding_token_embedding() {
        let mut layer = layer_with_weights(3, 2, vec![0.0; 6]);
        let sequence = [2, 0, 1];
        let grad_output = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        layer.backward(&sequence, &grad_output);
        layer.update_weights(1.0);

        assert_eq!(layer.weights, vec![-3.0, -4.0, -5.0, -6.0, -1.0, -2.0]);
    }

    #[test]
    fn backward_accumulates_gradients_for_repeated_tokens_and_calls() {
        let mut layer = layer_with_weights(3, 2, vec![0.0; 6]);

        layer.backward(&[2, 0, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        layer.backward(&[2], &[7.0, 8.0]);
        layer.update_weights(1.0);

        assert_eq!(layer.weights, vec![-3.0, -4.0, 0.0, 0.0, -13.0, -16.0]);
    }

    #[test]
    fn update_weights_applies_accumulated_gradients_and_resets_them() {
        let mut layer = layer_with_weights(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        layer.backward(&[0, 1], &[0.5, -1.0, 0.0, 2.0]);

        layer.update_weights(0.2);

        assert_float_slices_eq(&layer.weights, &[0.9, 2.2, 3.0, 3.6]);

        let weights_after_first_update = layer.weights.clone();
        layer.update_weights(0.2);
        assert_eq!(layer.weights, weights_after_first_update);
    }

    fn layer_with_weights(vocab_size: usize, d_model: usize, weights: Vec<f32>) -> EmbeddingLayer {
        assert_eq!(weights.len(), vocab_size * d_model);
        EmbeddingLayer {
            vocab_size,
            d_model,
            gradients: vec![0.0; weights.len()],
            weights,
        }
    }

    fn assert_float_slices_eq(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < f32::EPSILON,
                "value at index {index}: expected {expected}, got {actual}"
            );
        }
    }
}
