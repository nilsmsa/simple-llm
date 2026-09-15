use rand::Rng;
use rand_distr::{Distribution, Normal};

pub struct AttentionCache {
    pub input: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub probs: Vec<f32>,
}

pub struct SelfAttentionLayer {
    pub d_model: usize,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub wq_gradients: Vec<f32>,
    pub wk_gradients: Vec<f32>,
    pub wv_gradients: Vec<f32>,
    pub cache: Option<AttentionCache>,
    /// `dLoss/dProbability` for hver (query, key)-attention-vekt fra siste
    /// `backward`-kall, radvis som `cache.probs`. Positiv verdi betyr at en
    /// høyere attention-vekt der ville økt loss; negativ verdi betyr at en
    /// høyere vekt ville redusert loss. Kun til forklaring/trasering — brukes
    /// ikke videre i selve gradientberegningen (den bruker `d_scores`, som
    /// går gjennom softmax-jacobianen fra denne verdien).
    pub last_d_probs: Option<Vec<f32>>,
}

impl SelfAttentionLayer {
    /// Oppretter projeksjonene som lager query, key og value for hvert token.
    ///
    /// De tre rollene lar modellen lære hva et token leter etter, hva det
    /// tilbyr som kontekst, og hvilken informasjon som skal sendes videre.
    pub fn new<R: Rng + ?Sized>(d_model: usize, rng: &mut R) -> Self {
        let size = d_model * d_model;
        let normal = Normal::new(0.0, 0.01).expect("standard deviation is positive");
        let mut random_weights = || {
            (0..size)
                .map(|_| normal.sample(rng) as f32)
                .collect::<Vec<_>>()
        };
        Self {
            d_model,
            w_q: random_weights(),
            w_k: random_weights(),
            w_v: random_weights(),
            wq_gradients: vec![0.0; size],
            wk_gradients: vec![0.0; size],
            wv_gradients: vec![0.0; size],
            cache: None,
            last_d_probs: None,
        }
    }

    /// Lar hvert token hente relevant informasjon fra tidligere tokens.
    ///
    /// Query og key bestemmer hvor mye oppmerksomhet hvert tidligere token
    /// får. Disse vektene brukes til å blande value-vektorene til ny kontekst.
    pub fn forward(&mut self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let model_width = self.d_model;
        // Regner ut "q" for hvert token: input multiplisert med vektmatrisen "w_q".
        let queries = project_role(input, &self.w_q, seq_len, model_width);
        // Samme som over, men med vektmatrisen "w_k" gir dette "k".
        let keys = project_role(input, &self.w_k, seq_len, model_width);
        // Samme som over, men med vektmatrisen "w_v" gir dette "v".
        let values = project_role(input, &self.w_v, seq_len, model_width);

        let mut probs = compute_scores(&queries, &keys, seq_len, model_width);
        apply_causal_mask(&mut probs, seq_len);
        softmax_rows(&mut probs, seq_len);

        let mut context_sequence = mix_values(&probs, &values, seq_len, model_width);

        // Bevar tokenets egen embedding ved å legge den til attention-resultatet.
        for (context_value, input_value) in context_sequence.iter_mut().zip(input) {
            *context_value += input_value;
        }
        self.cache = Some(AttentionCache {
            input: input.to_vec(),
            q: queries,
            k: keys,
            v: values,
            probs,
        });

        context_sequence
    }

    /// Sender gradientene bakover gjennom hele attention-beregningen.
    ///
    /// Funksjonen finner både hvordan query-, key- og value-vektene bør
    /// endres, og hvilket feilsignal embedding-laget skal få.
    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        let cache = self
            .cache
            .as_ref()
            .expect("Forward pass required before backward");
        let sequence_length = cache.input.len() / self.d_model;
        let model_width = self.d_model;

        let mut d_v = vec![0.0; sequence_length * model_width];
        // Regner ut hvor mye hver rad i "v" må justeres for å redusere feilen,
        // basert på hvor mye vekt raden fikk i "probs" for hver posisjon i sekvensen.
        matmul(
            &cache.probs,
            grad_output,
            &mut d_v,
            sequence_length,
            model_width,
            sequence_length,
            true,
            false,
        );

        let mut d_probs = vec![0.0; sequence_length * sequence_length];
        // Regner ut hvor mye feilen ville endret seg om vektingen mellom
        // posisjonene i "probs" var litt annerledes.
        matmul(
            grad_output,
            &cache.v,
            &mut d_probs,
            sequence_length,
            sequence_length,
            model_width,
            false,
            true,
        );
        self.last_d_probs = Some(d_probs.clone());

        let mut d_scores = vec![0.0; sequence_length * sequence_length];
        // Softmax-gradienten kobler alle sannsynlighetene i samme attention-rad.
        for query_index in 0..sequence_length {
            let row_start = query_index * sequence_length;
            let mut probability_weighted_gradient = 0.0;
            for key_index in 0..sequence_length {
                probability_weighted_gradient +=
                    d_probs[row_start + key_index] * cache.probs[row_start + key_index];
            }
            for key_index in 0..sequence_length {
                let probability = cache.probs[row_start + key_index];
                let probability_gradient = d_probs[row_start + key_index];
                d_scores[row_start + key_index] = probability
                    * (probability_gradient - probability_weighted_gradient)
                    / (model_width as f32).sqrt();
            }
        }
        // Maskerte framtidsposisjoner skal heller ikke motta gradienter.
        for query_index in 0..sequence_length {
            for future_key_index in query_index + 1..sequence_length {
                let score_index = query_index * sequence_length + future_key_index;
                d_scores[score_index] = 0.0;
            }
        }

        let mut d_q = vec![0.0; sequence_length * model_width];
        let mut d_k = vec![0.0; sequence_length * model_width];
        // Regner ut hvor mye "q" må justeres, ved å kombinere feilen per
        // posisjonspar ("d_scores") med de tilhørende radene i "k".
        matmul(
            &d_scores,
            &cache.k,
            &mut d_q,
            sequence_length,
            model_width,
            sequence_length,
            false,
            false,
        );
        // Regner ut hvor mye "k" må justeres, samme feil som over, men nå
        // koblet mot de tilhørende radene i "q" i stedet.
        matmul(
            &d_scores,
            &cache.q,
            &mut d_k,
            sequence_length,
            model_width,
            sequence_length,
            true,
            false,
        );

        // Regner ut hvor mye vektmatrisen "w_q" må justeres, ved å kombinere
        // input-verdiene med feilen som ble funnet for "q" over.
        matmul(
            &cache.input,
            &d_q,
            &mut self.wq_gradients,
            model_width,
            model_width,
            sequence_length,
            true,
            false,
        );
        // Samme som over, men for vektmatrisen "w_k".
        matmul(
            &cache.input,
            &d_k,
            &mut self.wk_gradients,
            model_width,
            model_width,
            sequence_length,
            true,
            false,
        );
        // Samme som over, men for vektmatrisen "w_v".
        matmul(
            &cache.input,
            &d_v,
            &mut self.wv_gradients,
            model_width,
            model_width,
            sequence_length,
            true,
            false,
        );

        let mut d_x = vec![0.0; sequence_length * model_width];
        // Fører feilen fra "q" tilbake til selve input-teksten (embeddingen),
        // slik at laget under også kan justeres riktig vei.
        matmul(
            &d_q,
            &self.w_q,
            &mut d_x,
            sequence_length,
            model_width,
            model_width,
            false,
            true,
        );
        // Samme som over, men feilen fra "k".
        matmul(
            &d_k,
            &self.w_k,
            &mut d_x,
            sequence_length,
            model_width,
            model_width,
            false,
            true,
        );
        // Samme som over, men feilen fra "v".
        matmul(
            &d_v,
            &self.w_v,
            &mut d_x,
            sequence_length,
            model_width,
            model_width,
            false,
            true,
        );
        // Før gradienten gjennom den direkte residualveien tilbake til input.
        for i in 0..d_x.len() {
            d_x[i] += grad_output[i];
        }
        d_x
    }

    /// Oppdaterer query-, key- og value-vektene med SGD.
    pub fn update_weights(&mut self, learning_rate: f32) {
        for weight_index in 0..self.w_q.len() {
            self.w_q[weight_index] -= learning_rate * self.wq_gradients[weight_index];
            self.wq_gradients[weight_index] = 0.0;
            self.w_k[weight_index] -= learning_rate * self.wk_gradients[weight_index];
            self.wk_gradients[weight_index] = 0.0;
            self.w_v[weight_index] -= learning_rate * self.wv_gradients[weight_index];
            self.wv_gradients[weight_index] = 0.0;
        }
    }
}

/// Projiserer hver embedding til en ny rolle, for eksempel query eller key.
///
/// Samme projeksjonsmatrise brukes på alle tokens, og vektene læres under
/// trening.
pub fn project_role(input: &[f32], weights: &[f32], seq_len: usize, d_model: usize) -> Vec<f32> {
    let mut output = vec![0.0; seq_len * d_model];
    for row in 0..seq_len {
        for col in 0..d_model {
            let mut sum = 0.0;
            for input_feature in 0..d_model {
                let input_idx = row * d_model + input_feature;
                let weight_idx = input_feature * d_model + col;
                sum += input[input_idx] * weights[weight_idx];
            }
            let out_idx = row * d_model + col;
            output[out_idx] = sum;
        }
    }
    output
}

/// Måler hvor relevant hvert key-token er for hvert query-token.
///
/// Dot product gir høy score når vektorene både peker i samme retning og har
/// stor størrelse. Skalering med kvadratroten av `d_model` holder tallene i et
/// stabilt område.
pub fn compute_scores(queries: &[f32], keys: &[f32], seq_len: usize, d_model: usize) -> Vec<f32> {
    let mut scores = vec![0.0; seq_len * seq_len];
    let scale = (d_model as f32).sqrt();
    for q_row in 0..seq_len {
        for k_row in 0..seq_len {
            let mut sum = 0.0;
            for feature_index in 0..d_model {
                let q_idx = q_row * d_model + feature_index;
                let k_idx = k_row * d_model + feature_index;
                sum += queries[q_idx] * keys[k_idx];
            }
            let score_idx = q_row * seq_len + k_row;
            scores[score_idx] = sum / scale;
        }
    }
    scores
}

/// Lager kontekst ved å ta et vektet gjennomsnitt av value-vektorene.
///
/// Attention-sannsynlighetene bestemmer hvilke tidligere tokens som bidrar
/// mest til representasjonen ved hver posisjon.
pub fn mix_values(
    attention_probs: &[f32],
    values: &[f32],
    seq_len: usize,
    d_model: usize,
) -> Vec<f32> {
    let mut mixed = vec![0.0; seq_len * d_model];
    for row in 0..seq_len {
        for col in 0..d_model {
            let mut sum = 0.0;
            for value_row in 0..seq_len {
                let prob_idx = row * seq_len + value_row;
                let val_idx = value_row * d_model + col;
                sum += attention_probs[prob_idx] * values[val_idx];
            }
            let out_idx = row * d_model + col;
            mixed[out_idx] = sum;
        }
    }
    mixed
}

/// Skjuler framtidige tokens, slik at modellen ikke kan se fasiten.
///
/// Dette gjør attention causal og er nødvendig for ærlig neste-token-trening.
pub fn apply_causal_mask(scores: &mut [f32], seq_len: usize) {
    for row in 0..seq_len {
        for col in 0..seq_len {
            if col > row {
                let idx = row * seq_len + col;
                scores[idx] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Gjør attention-scorene i hver rad om til sannsynligheter.
///
/// Maskerte posisjoner får 0, mens de synlige posisjonene summerer til 1.
pub fn softmax_rows(scores: &mut [f32], seq_len: usize) {
    for row in 0..seq_len {
        let row_start = row * seq_len;

        let mut max_val = f32::NEG_INFINITY;
        for col in 0..seq_len {
            let val = scores[row_start + col];
            if val > max_val {
                max_val = val
            }
        }

        let mut sum_exp = 0.0;
        for col in 0..seq_len {
            let val = scores[row_start + col];
            if val != f32::NEG_INFINITY {
                sum_exp += (val - max_val).exp();
            }
        }

        for col in 0..seq_len {
            let idx = row_start + col;
            if scores[idx] == f32::NEG_INFINITY {
                scores[idx] = 0.0;
            } else {
                scores[idx] = (scores[idx] - max_val).exp() / sum_exp;
            }
        }
    }
}

/// Multipliserer to matriser som ligger flatt i minnet.
///
/// Transpose-flaggene gjør at samme hjelpefunksjon kan brukes i både forward
/// pass og backpropagation. Resultatet legges til eksisterende verdier i `c`.
#[allow(clippy::too_many_arguments)]
pub fn matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
) {
    for output_row in 0..m {
        for output_column in 0..n {
            let mut sum = 0.0;
            for shared_index in 0..k {
                let a_val = if trans_a {
                    a[shared_index * m + output_row]
                } else {
                    a[output_row * k + shared_index]
                };
                let b_val = if trans_b {
                    b[output_column * k + shared_index]
                } else {
                    b[shared_index * n + output_column]
                };
                sum += a_val * b_val;
            }
            c[output_row * n + output_column] += sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    const TOLERANCE: f32 = 1e-5;

    #[test]
    fn new_creates_usable_projection_matrices_with_the_expected_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut layer = SelfAttentionLayer::new(3, &mut rng);

        assert_eq!(layer.d_model, 3);
        assert_eq!(layer.w_q.len(), 9);
        assert_eq!(layer.w_k.len(), 9);
        assert_eq!(layer.w_v.len(), 9);
        assert!(layer.w_q.iter().all(|weight| weight.is_finite()));
        assert!(layer.w_k.iter().all(|weight| weight.is_finite()));
        assert!(layer.w_v.iter().all(|weight| weight.is_finite()));

        let initial_weights = (layer.w_q.clone(), layer.w_k.clone(), layer.w_v.clone());
        layer.update_weights(0.1);
        assert_eq!((layer.w_q, layer.w_k, layer.w_v), initial_weights);
    }

    #[test]
    fn project_role_multiplies_each_token_by_the_projection_matrix() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights = [1.0, 2.0, 3.0, 4.0];

        let projected = project_role(&input, &weights, 2, 2);

        assert_float_slices_eq(&projected, &[7.0, 10.0, 15.0, 22.0], TOLERANCE);
    }

    #[test]
    fn compute_scores_calculates_scaled_query_key_dot_products() {
        let queries = [1.0, 0.0, 0.0, 2.0];
        let keys = [3.0, 0.0, 0.0, 4.0];
        let scale = 2.0_f32.sqrt();

        let scores = compute_scores(&queries, &keys, 2, 2);

        assert_float_slices_eq(&scores, &[3.0 / scale, 0.0, 0.0, 8.0 / scale], TOLERANCE);
    }

    #[test]
    fn apply_causal_mask_blocks_attention_to_future_tokens() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        apply_causal_mask(&mut scores, 3);

        assert_eq!(
            scores,
            vec![
                1.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                4.0,
                5.0,
                f32::NEG_INFINITY,
                7.0,
                8.0,
                9.0
            ]
        );
    }

    #[test]
    fn softmax_rows_normalizes_each_row_and_keeps_masked_values_at_zero() {
        let mut scores = [1_000.0, f32::NEG_INFINITY, 1_000.0, 1_000.0];

        softmax_rows(&mut scores, 2);

        assert_float_slices_eq(&scores, &[1.0, 0.0, 0.5, 0.5], TOLERANCE);
        for row in scores.chunks(2) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < TOLERANCE);
        }
    }

    #[test]
    fn mix_values_calculates_weighted_value_sums_for_each_token() {
        let attention_probs = [1.0, 0.0, 0.25, 0.75];
        let values = [2.0, 4.0, 6.0, 8.0];

        let mixed = mix_values(&attention_probs, &values, 2, 2);

        assert_float_slices_eq(&mixed, &[2.0, 4.0, 5.0, 7.0], TOLERANCE);
    }

    #[test]
    fn matmul_multiplies_matrices_and_accumulates_into_output() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut output = [1.0; 4];

        matmul(&a, &b, &mut output, 2, 2, 3, false, false);

        assert_eq!(output, [59.0, 65.0, 140.0, 155.0]);
    }

    #[test]
    fn matmul_supports_transposed_inputs() {
        let a_transposed = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b_transposed = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
        let mut with_transposed_a = [0.0; 4];
        let mut with_transposed_b = [0.0; 4];

        matmul(
            &a_transposed,
            &b,
            &mut with_transposed_a,
            2,
            2,
            3,
            true,
            false,
        );
        matmul(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &b_transposed,
            &mut with_transposed_b,
            2,
            2,
            3,
            false,
            true,
        );

        assert_eq!(with_transposed_a, [58.0, 64.0, 139.0, 154.0]);
        assert_eq!(with_transposed_b, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn forward_applies_scaled_causal_self_attention_with_residual_connection() {
        let mut layer = layer_with_weights(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
        );
        let input = [1.0, 0.0, 0.0, 1.0];
        let second_token_self_attention = (1.0 / 2.0_f32.sqrt()).exp();
        let second_token_first_probability = 1.0 / (1.0 + second_token_self_attention);
        let second_token_self_probability =
            second_token_self_attention / (1.0 + second_token_self_attention);

        let output = layer.forward(&input, 2);

        assert_float_slices_eq(
            &output,
            &[
                2.0,
                0.0,
                second_token_first_probability,
                second_token_self_probability + 1.0,
            ],
            TOLERANCE,
        );
    }

    #[test]
    fn backward_and_weight_updates_match_numerical_gradients() {
        let input = vec![0.2, -0.4, 0.7, 0.1];
        let grad_output = [0.5, -0.7, 0.9, 0.2];
        let w_q = vec![0.3, -0.2, 0.4, 0.1];
        let w_k = vec![-0.1, 0.5, 0.2, 0.3];
        let w_v = vec![0.6, -0.3, 0.2, 0.8];
        let mut layer = layer_with_weights(w_q.clone(), w_k.clone(), w_v.clone());
        layer.forward(&input, 2);

        let input_gradients = layer.backward(&grad_output);

        let numerical_input_gradients =
            numerical_input_gradients(&input, &grad_output, &w_q, &w_k, &w_v);
        let numerical_wq_gradients =
            numerical_weight_gradients(&input, &grad_output, &w_q, &w_k, &w_v, WeightKind::Query);
        let numerical_wk_gradients =
            numerical_weight_gradients(&input, &grad_output, &w_q, &w_k, &w_v, WeightKind::Key);
        let numerical_wv_gradients =
            numerical_weight_gradients(&input, &grad_output, &w_q, &w_k, &w_v, WeightKind::Value);

        assert_float_slices_eq(&input_gradients, &numerical_input_gradients, 1e-3);

        let learning_rate = 0.25;
        layer.update_weights(learning_rate);
        assert_weights_updated_from_gradient(
            &layer.w_q,
            &w_q,
            &numerical_wq_gradients,
            learning_rate,
        );
        assert_weights_updated_from_gradient(
            &layer.w_k,
            &w_k,
            &numerical_wk_gradients,
            learning_rate,
        );
        assert_weights_updated_from_gradient(
            &layer.w_v,
            &w_v,
            &numerical_wv_gradients,
            learning_rate,
        );

        let weights_after_first_update = (layer.w_q.clone(), layer.w_k.clone(), layer.w_v.clone());
        layer.update_weights(learning_rate);
        assert_eq!(
            (layer.w_q, layer.w_k, layer.w_v),
            weights_after_first_update
        );
    }

    #[test]
    #[should_panic]
    fn backward_requires_a_forward_pass() {
        let mut rng = StdRng::seed_from_u64(42);
        SelfAttentionLayer::new(2, &mut rng).backward(&[0.0; 4]);
    }

    #[derive(Clone, Copy)]
    enum WeightKind {
        Query,
        Key,
        Value,
    }

    fn layer_with_weights(w_q: Vec<f32>, w_k: Vec<f32>, w_v: Vec<f32>) -> SelfAttentionLayer {
        assert_eq!(w_q.len(), w_k.len());
        assert_eq!(w_q.len(), w_v.len());
        let d_model = (w_q.len() as f32).sqrt() as usize;
        assert_eq!(d_model * d_model, w_q.len());
        SelfAttentionLayer {
            d_model,
            wq_gradients: vec![0.0; w_q.len()],
            wk_gradients: vec![0.0; w_k.len()],
            wv_gradients: vec![0.0; w_v.len()],
            w_q,
            w_k,
            w_v,
            cache: None,
            last_d_probs: None,
        }
    }

    fn numerical_input_gradients(
        input: &[f32],
        grad_output: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
    ) -> Vec<f32> {
        numerical_gradients(input, |candidate| {
            attention_loss(candidate, grad_output, w_q, w_k, w_v)
        })
    }

    fn numerical_weight_gradients(
        input: &[f32],
        grad_output: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        weight_kind: WeightKind,
    ) -> Vec<f32> {
        let weights = match weight_kind {
            WeightKind::Query => w_q,
            WeightKind::Key => w_k,
            WeightKind::Value => w_v,
        };
        numerical_gradients(weights, |candidate| {
            let (candidate_w_q, candidate_w_k, candidate_w_v) = match weight_kind {
                WeightKind::Query => (candidate, w_k, w_v),
                WeightKind::Key => (w_q, candidate, w_v),
                WeightKind::Value => (w_q, w_k, candidate),
            };
            attention_loss(
                input,
                grad_output,
                candidate_w_q,
                candidate_w_k,
                candidate_w_v,
            )
        })
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

    fn attention_loss(
        input: &[f32],
        grad_output: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
    ) -> f32 {
        let mut layer = layer_with_weights(w_q.to_vec(), w_k.to_vec(), w_v.to_vec());
        layer
            .forward(input, input.len() / layer.d_model)
            .iter()
            .zip(grad_output)
            .map(|(output, gradient)| output * gradient)
            .sum()
    }

    fn assert_weights_updated_from_gradient(
        actual: &[f32],
        before: &[f32],
        gradient: &[f32],
        learning_rate: f32,
    ) {
        let expected: Vec<_> = before
            .iter()
            .zip(gradient)
            .map(|(weight, gradient)| weight - learning_rate * gradient)
            .collect();
        assert_float_slices_eq(actual, &expected, 1e-3);
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
