use rand::Rng;
use rand_distr::{Distribution, Normal};

pub struct LinearLayer {
    pub d_model: usize,
    pub vocab_size: usize,
    pub weights: Vec<f32>,
    pub weight_gradients: Vec<f32>,
}

impl LinearLayer {
    /// Oppretter output-laget med én rad tilfeldige vekter per token.
    ///
    /// Laget lærer å gjøre den siste kontekstvektoren om til én score, en
    /// `logit`, for hvert mulig neste token.
    pub fn new<R: Rng + ?Sized>(vocab_size: usize, d_model: usize, rng: &mut R) -> Self {
        let normal = Normal::new(0.0, 1.0).expect("Should be ok?");
        let total_weights = vocab_size * d_model;
        let mut weights = Vec::with_capacity(total_weights);
        for _ in 0..total_weights {
            weights.push(normal.sample(rng) as f32);
        }
        Self {
            d_model,
            vocab_size,
            weights,
            weight_gradients: vec![0.0; total_weights],
        }
    }

    /// Beregner én logit per token i vocabulary.
    ///
    /// Høyere logit betyr at modellen vurderer tokenet som en bedre kandidat
    /// for neste plass i teksten.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0; self.vocab_size];
        for (token_index, logit) in logits.iter_mut().enumerate().take(self.vocab_size) {
            let mut sum = 0.0;
            let row_start = token_index * self.d_model;
            for (feature_index, &input_value) in input.iter().enumerate().take(self.d_model) {
                let weight_index = row_start + feature_index;
                sum += input_value * self.weights[weight_index];
            }
            *logit = sum;
        }
        logits
    }

    /// Sender feilsignalet bakover og samler gradienter for output-vektene.
    ///
    /// Returverdien forteller attention-laget hvordan kontekstvektoren bidro
    /// til loss.
    pub fn backward(&mut self, input: &[f32], grad_output: &[f32]) -> Vec<f32> {
        let d_model = input.len();
        let vocab_size = grad_output.len();

        let mut grad_input = vec![0.0; d_model];

        for (token_index, &error_signal) in grad_output.iter().enumerate().take(vocab_size) {
            let row_start = token_index * d_model;

            for (feature_index, &input_value) in input.iter().enumerate().take(d_model) {
                self.weight_gradients[row_start + feature_index] += error_signal * input_value;
            }
        }

        for (token_index, &error_signal) in grad_output.iter().enumerate().take(vocab_size) {
            let row_start = token_index * d_model;

            for (feature_index, input_gradient) in grad_input.iter_mut().enumerate().take(d_model) {
                *input_gradient += self.weights[row_start + feature_index] * error_signal;
            }
        }

        grad_input
    }

    /// Oppdaterer output-vektene med SGD og nullstiller gradientene.
    pub fn update_weights(&mut self, learning_rate: f32) {
        for weight_index in 0..self.weights.len() {
            self.weights[weight_index] -= learning_rate * self.weight_gradients[weight_index];
            self.weight_gradients[weight_index] = 0.0;
        }
    }
}

/// Måler gjennomsnittlig kvadratisk avstand mellom prediksjon og fasit.
///
/// MSE er med som et enkelt loss-eksempel, men språkmodellen trener med
/// cross-entropy fordi målet er å velge ett token blant mange.
pub fn mse_loss(predictions: &[f32], targets: &[f32]) -> f32 {
    let mut sum = 0.0;
    let len = predictions.len();
    for value_index in 0..len {
        let diff = predictions[value_index] - targets[value_index];
        sum += diff * diff;
    }
    sum / len as f32
}

/// Beregner gradienten til MSE for hver predikert verdi.
pub fn mse_loss_derivative(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    let len = predictions.len();
    let mut gradients = Vec::with_capacity(len);
    for value_index in 0..len {
        let diff = predictions[value_index] - targets[value_index];
        gradients.push(2.0 * diff / len as f32);
    }
    gradients
}

/// Gjør vilkårlige logits om til sannsynligheter som summerer til 1.
///
/// Den største logiten trekkes fra først for å unngå numerisk overflow.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut max_val = f32::NEG_INFINITY;
    for &logit in logits {
        if logit > max_val {
            max_val = logit;
        }
    }

    let mut sum_exp = 0.0;
    let mut exps = vec![0.0; logits.len()];
    for (token_index, &logit) in logits.iter().enumerate() {
        exps[token_index] = (logit - max_val).exp();
        sum_exp += exps[token_index];
    }

    let mut probs = vec![0.0; logits.len()];
    for (token_index, &exp_value) in exps.iter().enumerate() {
        probs[token_index] = exp_value / sum_exp;
    }
    probs
}

/// Måler hvor lite sannsynlighet modellen ga tokenet som faktisk var riktig.
///
/// Riktig token med høy sannsynlighet gir lav loss. Et sikkert, men feil svar
/// gir høy loss.
pub fn cross_entropy_loss(predictions: &[f32], targets: &[f32]) -> f32 {
    let probs = softmax(predictions);
    let mut loss = 0.0;
    for (token_index, &target) in targets.iter().enumerate() {
        if target > 0.0 {
            loss -= (probs[token_index] + 1e-7).ln();
        }
    }
    loss
}

/// Beregner feilsignalet som starter backpropagation fra output-laget.
///
/// For softmax og cross-entropy blir gradienten enkelt
/// `sannsynlighet - fasit` for hvert token.
pub fn cross_entropy_derivative(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    let probs = softmax(predictions);
    let mut gradients = vec![0.0; predictions.len()];
    for token_index in 0..gradients.len() {
        gradients[token_index] = probs[token_index] - targets[token_index];
    }
    gradients
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    const TOLERANCE: f32 = 1e-6;

    #[test]
    fn new_creates_a_usable_weight_matrix_with_the_expected_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut layer = LinearLayer::new(4, 3, &mut rng);

        assert_eq!(layer.vocab_size, 4);
        assert_eq!(layer.d_model, 3);
        assert_eq!(layer.weights.len(), 12);
        assert!(layer.weights.iter().all(|weight| weight.is_finite()));

        let initial_weights = layer.weights.clone();
        layer.update_weights(0.1);
        assert_eq!(layer.weights, initial_weights);
    }

    #[test]
    fn forward_projects_the_input_to_one_logit_per_vocabulary_item() {
        let layer = layer_with_weights(3, 2, vec![1.0, 3.0, -2.0, 4.0, 0.5, -1.0]);

        let logits = layer.forward(&[2.0, -1.0]);

        assert_float_slices_eq(&logits, &[-1.0, -8.0, 2.0], TOLERANCE);
    }

    #[test]
    fn backward_returns_input_gradient_and_update_applies_weight_gradient() {
        let initial_weights = vec![1.0, 3.0, -2.0, 4.0, 0.5, -1.0];
        let mut layer = layer_with_weights(3, 2, initial_weights);

        let input_gradient = layer.backward(&[2.0, -1.0], &[0.2, -0.3, 0.4]);
        layer.update_weights(0.5);

        assert_float_slices_eq(&input_gradient, &[1.0, -1.0], TOLERANCE);
        assert_float_slices_eq(
            &layer.weights,
            &[0.8, 3.1, -1.7, 3.85, 0.1, -0.8],
            TOLERANCE,
        );
    }

    #[test]
    fn backward_accumulates_gradients_until_weights_are_updated() {
        let mut layer = layer_with_weights(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

        layer.backward(&[2.0, -1.0], &[0.5, -0.25]);
        layer.backward(&[2.0, -1.0], &[0.5, -0.25]);
        layer.update_weights(0.1);

        assert_float_slices_eq(&layer.weights, &[0.8, 2.1, 3.1, 3.95], TOLERANCE);

        let weights_after_first_update = layer.weights.clone();
        layer.update_weights(0.1);
        assert_eq!(layer.weights, weights_after_first_update);
    }

    #[test]
    fn mse_loss_is_the_mean_of_squared_errors() {
        let loss = mse_loss(&[1.0, 3.0], &[0.0, 1.0]);

        assert!((loss - 2.5).abs() < TOLERANCE);
    }

    #[test]
    fn mse_loss_derivative_matches_the_numerical_loss_gradient() {
        let predictions = [1.0, 3.0];
        let targets = [0.0, 1.0];

        let derivative = mse_loss_derivative(&predictions, &targets);
        let numerical_derivative =
            numerical_gradients(&predictions, |candidate| mse_loss(candidate, &targets));

        assert_float_slices_eq(&derivative, &numerical_derivative, 1e-3);
    }

    #[test]
    fn softmax_is_stable_shift_invariant_and_normalized() {
        let probabilities = softmax(&[1_000.0, 1_001.0, 1_002.0]);
        let shifted_probabilities = softmax(&[-2.0, -1.0, 0.0]);

        assert_float_slices_eq(
            &probabilities,
            &[0.090_030_57, 0.244_728_48, 0.665_240_94],
            TOLERANCE,
        );
        assert_float_slices_eq(&probabilities, &shifted_probabilities, TOLERANCE);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < TOLERANCE);
        assert!(probabilities.iter().all(|&probability| probability > 0.0));
    }

    #[test]
    fn cross_entropy_loss_is_negative_log_likelihood_of_the_target() {
        let logits = [0.0, 3.0_f32.ln()];
        let targets = [0.0, 1.0];

        let loss = cross_entropy_loss(&logits, &targets);

        assert!((loss - -(0.75_f32).ln()).abs() < TOLERANCE);
    }

    #[test]
    fn cross_entropy_derivative_matches_softmax_minus_target() {
        let logits = [0.0, 3.0_f32.ln()];
        let targets = [0.0, 1.0];

        let derivative = cross_entropy_derivative(&logits, &targets);

        assert_float_slices_eq(&derivative, &[0.25, -0.25], TOLERANCE);
        assert!(derivative.iter().sum::<f32>().abs() < TOLERANCE);
    }

    fn layer_with_weights(vocab_size: usize, d_model: usize, weights: Vec<f32>) -> LinearLayer {
        assert_eq!(weights.len(), vocab_size * d_model);
        LinearLayer {
            d_model,
            vocab_size,
            weight_gradients: vec![0.0; weights.len()],
            weights,
        }
    }

    fn numerical_gradients(values: &[f32], loss: impl Fn(&[f32]) -> f32) -> Vec<f32> {
        const EPSILON: f32 = 1e-3;

        (0..values.len())
            .map(|index| {
                let mut above = values.to_vec();
                let mut below = values.to_vec();
                above[index] += EPSILON;
                below[index] -= EPSILON;
                (loss(&above) - loss(&below)) / (2.0 * EPSILON)
            })
            .collect()
    }

    fn assert_float_slices_eq(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value at index {index}: expected {expected}, got {actual}"
            );
        }
    }
}
