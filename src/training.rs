use crate::{
    bpe_tokenizer::{build_tokenizer, train},
    embedding_layer::EmbeddingLayer,
    linear_layer::{LinearLayer, cross_entropy_derivative, softmax},
    matrix::Matrix,
    self_attatention_layer::SelfAttentionLayer,
    sliding_window::sliding_windows,
    tokenizer::Tokenizer,
};
use rand::{SeedableRng, rngs::StdRng};

/// Trener en BPE-tokenizer på teksten og bygger vocabulary fra resultatet.
pub fn build_bpe_tokenizer_from_text(
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
            train_on_window(model, sequence, target, learning_rate);
        }
    }
}

/// Som `train_model`, men tar stikkprøver av attention og toppkandidat for en
/// fast prompt (`probe_tokens`) mens treningen pågår.
///
/// Stikkprøvene tas mellom epokene, aldri midt i et vindus forward/backward,
/// så de forstyrrer ikke gradientene. De viser hvordan attention-vektene
/// beveger seg fra tilfeldige startverdier til det mønsteret `-trace` viser
/// ved prediksjon.
pub fn train_model_with_snapshots<T: Tokenizer>(
    model: &mut Model<T>,
    training_data: &str,
    learning_rate: f32,
    epochs: usize,
    probe_tokens: &[u32],
    snapshot_count: usize,
) -> (Vec<TrainingSnapshot>, BackwardStepTrace) {
    let tokens: Vec<u32> = model.tokenizer.encode(training_data);
    let snapshot_epochs = snapshot_schedule(epochs, snapshot_count);
    let mut snapshots = Vec::with_capacity(snapshot_epochs.len() + 1);
    let mut backward_trace = None;

    for epoch in 0..epochs {
        if snapshot_epochs.contains(&epoch) {
            snapshots.push(capture_training_snapshot(model, probe_tokens, epoch));
        }
        let mut windows = sliding_windows(&tokens, model.seq_len);
        if epoch == 0 {
            let (sequence, target) = windows
                .next()
                .expect("training data must contain at least one window");
            backward_trace = Some(train_on_window_with_trace(
                model,
                sequence,
                target,
                learning_rate,
            ));
        }
        for (sequence, target) in windows {
            train_on_window(model, &sequence, target, learning_rate);
        }
    }
    snapshots.push(capture_training_snapshot(model, probe_tokens, epochs));
    (
        snapshots,
        backward_trace.expect("epochs must be greater than zero to train at all"),
    )
}

/// Trener modellen på ett enkelt (kontekst, fasit)-vindu med SGD.
fn train_on_window<T: Tokenizer>(
    model: &mut Model<T>,
    sequence: &[u32],
    target: u32,
    learning_rate: f32,
) {
    train_on_window_core(model, sequence, target, learning_rate, false);
}

/// Som `train_on_window`, men samler forward- og backward-verdiene som
/// forklarer hvorfor attention-vektene beveger seg slik de gjør for akkurat
/// dette vinduet.
fn train_on_window_with_trace<T: Tokenizer>(
    model: &mut Model<T>,
    sequence: &[u32],
    target: u32,
    learning_rate: f32,
) -> BackwardStepTrace {
    train_on_window_core(model, sequence, target, learning_rate, true).expect("trace was requested")
}

/// Kjerneimplementasjonen for ett treningsvindu. `with_trace` slår av og på
/// den ekstra bokføringen som `train_on_window_with_trace` trenger, slik at
/// selve SGD-oppdateringen er identisk uansett.
fn train_on_window_core<T: Tokenizer>(
    model: &mut Model<T>,
    sequence: &[u32],
    target: u32,
    learning_rate: f32,
    with_trace: bool,
) -> Option<BackwardStepTrace> {
    let embedded_sequence = Matrix::from_vec(
        model.seq_len,
        model.d_model,
        sequence
            .iter()
            .flat_map(|&t| model.embedding.forward(t).to_vec())
            .collect(),
    );

    // 1. Forward Pass
    let context_sequence = model.attention.forward(&embedded_sequence);
    let last_token_idx = (model.seq_len - 1) * model.d_model;
    let last_token_vector = &context_sequence[last_token_idx..(last_token_idx + model.d_model)];

    let attention_weights_before = with_trace.then(|| last_position_attention_weights(model));

    let predictions = model.linear.forward(last_token_vector);
    let targets = create_target(&predictions, target as usize);

    let gradients = cross_entropy_derivative(&predictions, &targets);
    let output_gradients = with_trace.then(|| {
        gradients
            .iter()
            .enumerate()
            .map(|(token_id, &gradient)| TokenGradient {
                token_id: token_id as u32,
                gradient,
            })
            .collect::<Vec<_>>()
    });

    // 2. Backward Pass (Linear -> Attention -> Embedding)
    let d_last_token = model.linear.backward(last_token_vector, &gradients);

    let mut d_context_sequence = Matrix::zeros(model.seq_len, model.d_model);
    d_context_sequence[last_token_idx..].copy_from_slice(&d_last_token);

    // Attention pulls its own internal cache now
    let d_embedded = model.attention.backward(&d_context_sequence);
    let attention_probability_gradients = with_trace.then(|| {
        let sequence_length = model.seq_len;
        let d_probs = model
            .attention
            .last_d_probs
            .as_ref()
            .expect("backward just ran and set this");
        let row_start = (sequence_length - 1) * sequence_length;
        d_probs[row_start..row_start + sequence_length].to_vec()
    });
    model.embedding.backward(sequence, &d_embedded);

    // 3. Update Weights
    model.linear.update_weights(learning_rate);
    model.attention.update_weights(learning_rate);
    model.embedding.update_weights(learning_rate);

    if !with_trace {
        return None;
    }

    // De oppdaterte embedding- og attention-vektene brukes til å vise hva
    // attention faktisk ville gjort med dette vinduet neste gang det dukker
    // opp, nå som vektene er justert.
    let updated_embedded_sequence = Matrix::from_vec(
        model.seq_len,
        model.d_model,
        sequence
            .iter()
            .flat_map(|&t| model.embedding.forward(t).to_vec())
            .collect(),
    );
    model.attention.forward(&updated_embedded_sequence);
    let attention_weights_after = last_position_attention_weights(model);

    Some(BackwardStepTrace {
        context_tokens: sequence.to_vec(),
        target_token: target,
        attention_weights_before: attention_weights_before.expect("with_trace is true"),
        output_gradients: output_gradients.expect("with_trace is true"),
        attention_probability_gradients: attention_probability_gradients
            .expect("with_trace is true"),
        attention_weights_after,
    })
}

/// Velger hvilke epoker en snapshot skal tas ved, jevnt fordelt fra epoke 0.
///
/// Selve sluttilstanden (etter siste epoke) legges alltid til separat i
/// `train_model_with_snapshots`, så den trenger ikke være med her.
fn snapshot_schedule(epochs: usize, snapshot_count: usize) -> Vec<usize> {
    if snapshot_count == 0 || epochs == 0 {
        return Vec::new();
    }
    let mut epochs_list: Vec<usize> = (0..snapshot_count)
        .map(|step| step * epochs / snapshot_count)
        .collect();
    epochs_list.dedup();
    epochs_list
}

/// Kjører et forward-pass med gjeldende vekter og fanger opp
/// attention-vektene og toppkandidaten for `probe_tokens`.
///
/// Dette gjenbruker `forward`, som ikke muterer noe treningstilstand utover
/// attention-cachen, så det er trygt å kalle mellom epoker.
fn capture_training_snapshot<T: Tokenizer>(
    model: &mut Model<T>,
    probe_tokens: &[u32],
    epoch: usize,
) -> TrainingSnapshot {
    let logits = forward(model, probe_tokens);
    let top_prediction = top_predictions(&logits, 1)
        .into_iter()
        .next()
        .expect("vocabulary is non-empty");

    TrainingSnapshot {
        epoch,
        attention_weights: last_position_attention_weights(model),
        top_prediction,
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
                attention_weights: last_position_attention_weights(model),
            });
        }
        current_tokens.push(next_token_id);
    }
    (model.tokenizer.decode(&current_tokens), trace)
}

/// Henter attention-vekten fra siste posisjon mot hvert tidligere token.
///
/// `forward` kaller `attention.forward`, som legger igjen en `probs`-matrise
/// (én softmax-rad per posisjon) i cachen. Raden for siste posisjon viser
/// nøyaktig hvor mye oppmerksomhet det neste tokenet baserer seg på fra hvert
/// token i konteksten, inkludert dem lenger tilbake enn treningens `seq_len`.
fn last_position_attention_weights<T: Tokenizer>(model: &Model<T>) -> Vec<f32> {
    let cache = model
        .attention
        .cache
        .as_ref()
        .expect("forward pass sets the attention cache before this is called");
    let sequence_length = cache.input.rows;
    let row_start = (sequence_length - 1) * sequence_length;
    cache.probs[row_start..row_start + sequence_length].to_vec()
}

/// Kjører ett forward pass og returnerer logits for neste token.
///
/// Bare outputen ved siste posisjon brukes, fordi oppgaven er å fortsette
/// teksten etter hele konteksten.
pub fn forward<T: Tokenizer>(model: &mut Model<T>, tokens: &[u32]) -> Vec<f32> {
    let seq_len = tokens.len();
    let mut embedded_data = Vec::with_capacity(seq_len * model.embedding.d_model);
    for &token_id in tokens {
        let token_vector = model.embedding.forward(token_id);
        embedded_data.extend_from_slice(token_vector);
    }
    let embedded_sequence = Matrix::from_vec(seq_len, model.embedding.d_model, embedded_data);
    let context_aware_sequence = model.attention.forward(&embedded_sequence);
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
    /// Én vekt per token i `context_tokens`, i samme rekkefølge. Viser hvor
    /// mye det siste tokenet "ser på" hvert tidligere token når det bestemmer
    /// neste token.
    pub attention_weights: Vec<f32>,
}

/// Én stikkprøve av attention og toppkandidat for en fast prompt, tatt et
/// gitt sted i treningen (`epoch` er antall fullførte epoker på det
/// tidspunktet).
#[derive(Debug, PartialEq)]
pub struct TrainingSnapshot {
    pub epoch: usize,
    /// Én vekt per token i probe-konteksten, i samme rekkefølge.
    pub attention_weights: Vec<f32>,
    pub top_prediction: TokenPrediction,
}

/// Ett token og modellens softmax-sannsynlighet for neste posisjon.
#[derive(Debug, PartialEq)]
pub struct TokenPrediction {
    pub token_id: u32,
    pub probability: f32,
}

/// Cross-entropy-gradienten (`sannsynlighet - fasit`) for ett token i
/// vocabulary, fra ett enkelt treningssteg. Negativ verdi betyr at loss
/// reduseres hvis logiten til dette tokenet øker; positiv verdi betyr at
/// logiten bør ned.
#[derive(Debug, PartialEq)]
pub struct TokenGradient {
    pub token_id: u32,
    pub gradient: f32,
}

/// Forward- og backward-verdiene fra ett enkelt treningsvindu, samlet for å
/// vise konkret hvorfor attention-vektene endrer seg.
///
/// `attention_weights_before`/`attention_weights_after` viser den samme
/// softmax-fordelingen som `-trace` viser ved prediksjon, men målt rett før
/// og rett etter denne ene SGD-oppdateringen. `attention_probability_gradients`
/// er `dLoss/dProbability` for hver attention-vekt: et negativt tall betyr at
/// gradienten "vil" øke akkurat den vekten, fordi det ville redusert loss.
#[derive(Debug, PartialEq)]
pub struct BackwardStepTrace {
    pub context_tokens: Vec<u32>,
    pub target_token: u32,
    pub attention_weights_before: Vec<f32>,
    pub output_gradients: Vec<TokenGradient>,
    pub attention_probability_gradients: Vec<f32>,
    pub attention_weights_after: Vec<f32>,
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
