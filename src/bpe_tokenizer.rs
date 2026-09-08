use std::{
    collections::HashMap,
    io::{BufReader, Read},
};

use crate::tokenizer::Tokenizer;

/// Bygger token-bytene ved å spille av de lærte BPE-sammenslåingene i ID-rekkefølge.
pub fn build_tokenizer(state: TokenizerState) -> BpeTokenizer {
    let next_id: usize = state.next_id as usize;
    let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(next_id);
    let merges = state.merge;
    let mut inverse_lookup: Vec<(u32, u32)> = vec![(0, 0); next_id - 256];
    for (key, value) in &merges {
        let token_id: usize = *value as usize;
        inverse_lookup[token_id - 256] = *key;
    }
    for token_id in 0..next_id {
        if token_id < 256 {
            vocab.push(vec![token_id as u8]);
        } else {
            let (left_token_id, right_token_id) = inverse_lookup[token_id - 256];
            let mut new_bytes = vocab[left_token_id as usize].clone();
            new_bytes.extend_from_slice(&vocab[right_token_id as usize]);
            vocab.push(new_bytes);
        }
    }
    BpeTokenizer { merges, vocab }
}

pub struct BpeTokenizer {
    pub merges: HashMap<(u32, u32), u32>,
    pub vocab: Vec<Vec<u8>>,
}

impl Tokenizer for BpeTokenizer {
    /// Bruker lærte sammenslåinger i prioritetsrekkefølge til ingen flere passer.
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = tokenize_text(text);
        loop {
            if tokens.len() < 2 {
                break;
            }
            let mut best_pair = None;
            let mut best_id = u32::MAX;
            for pair_start_index in 0..(tokens.len() - 1) {
                let pair = (tokens[pair_start_index], tokens[pair_start_index + 1]);
                if let Some(&new_id) = self.merges.get(&pair)
                    && new_id < best_id
                {
                    best_id = new_id;
                    best_pair = Some(pair);
                }
            }

            let target_pair = match best_pair {
                Some(pair) => pair,
                None => break,
            };

            let mut new_tokens = Vec::with_capacity(tokens.len());
            let mut token_index = 0;
            while token_index < tokens.len() {
                if token_index < tokens.len() - 1
                    && tokens[token_index] == target_pair.0
                    && tokens[token_index + 1] == target_pair.1
                {
                    new_tokens.push(best_id);
                    token_index += 2;
                } else {
                    new_tokens.push(tokens[token_index]);
                    token_index += 1;
                }
            }
            tokens = new_tokens;
        }
        tokens
    }

    /// Slår opp råbytene for hver token-ID og setter dem sammen til tekst.
    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token_bytes) = self.vocab.get(id as usize) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Returnerer antall byte- og BPE-tokens i vocabulary.
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Lærer de vanligste naboparene frem til ønsket vocabulary-størrelse.
pub fn train(data: impl Read, target_vocab_size: u32) -> TokenizerState {
    let mut state = TokenizerState {
        merge: HashMap::with_capacity(target_vocab_size as usize),
        data_as_tokens: init(data),
        next_id: 256,
    };
    let mut was_merged = true;
    while was_merged && state.next_id < target_vocab_size {
        (was_merged, state) = iteration(state);
    }
    state
}

/// Leser treningsdata og representerer dem som UTF-8-byte-ID-er.
pub fn init(data: impl Read) -> Vec<u32> {
    let mut data = BufReader::new(data);
    let mut buf = String::new();
    data.read_to_string(&mut buf).expect("Failed to read data");
    tokenize_text(&buf)
}

fn tokenize_text(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

/// Utfører én BPE-runde ved å slå sammen det vanligste naboparet.
pub fn iteration(state: TokenizerState) -> (bool, TokenizerState) {
    let mut counter: HashMap<(u32, u32), u32> =
        HashMap::with_capacity((state.data_as_tokens.len() / 2) + 1);
    for pair_start_index in 0..(state.data_as_tokens.len() - 1) {
        let left = state.data_as_tokens[pair_start_index];
        let right = state.data_as_tokens[pair_start_index + 1];
        counter
            .entry((left, right))
            .and_modify(|value| *value += 1)
            .or_insert(1);
    }
    let winner = counter
        .iter()
        .max_by_key(|e| e.1)
        .expect("Map should be none emtpy");
    if *winner.1 < 2 {
        return (false, state);
    }
    let winner_left_id = winner.0.0;
    let winner_right_id = winner.0.1;
    let mut state = state;
    let new_id = state.next_id;
    state.next_id += 1;
    state
        .merge
        .insert((winner_left_id, winner_right_id), new_id);
    merge_ids(
        &mut state.data_as_tokens,
        &winner_left_id,
        &winner_right_id,
        new_id,
    );
    (true, state)
}

/// Erstatter ikke-overlappende forekomster av et tokenpar direkte i bufferen.
pub fn merge_ids(data: &mut Vec<u32>, left_id: &u32, right_id: &u32, new_id: u32) {
    if data.is_empty() {
        return;
    }
    let mut write_index = 0;
    let mut merged_with_prev = false;
    // Les fremover og skriv det komprimerte resultatet tilbake i samme buffer.
    for read_index in 0..data.len() {
        let mut merged_token = None;
        if !merged_with_prev && read_index + 1 < data.len() {
            let left = &data[read_index];
            let right = &data[read_index + 1];
            if left == left_id && right == right_id {
                merged_token = Some(new_id);
            }
        }
        let merged_with_next = merged_token.is_some();
        if !merged_with_prev && !merged_with_next {
            if write_index != read_index {
                data.swap(write_index, read_index);
            }
            write_index += 1;
        } else if let Some(merged_token) = merged_token {
            data[write_index] = merged_token;
            write_index += 1;
        }
        merged_with_prev = merged_with_next;
    }
    data.truncate(write_index);
}

#[derive(Debug)]
pub struct TokenizerState {
    pub merge: HashMap<(u32, u32), u32>,
    pub data_as_tokens: Vec<u32>,
    pub next_id: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;

    #[test]
    fn base_encoder_uses_utf8_bytes() {
        let tokenizer = base_tokenizer();
        let text = "blå";
        let expected: Vec<u32> = text.bytes().map(u32::from).collect();

        let ids = tokenizer.encode(text);

        assert_eq!(ids, expected);
        assert_eq!(tokenizer.decode(&ids), text);
        assert_eq!(tokenizer.vocab_size(), 256);
    }

    #[test]
    fn angle_bracketed_text_is_encoded_as_ordinary_bytes() {
        let tokenizer = base_tokenizer();
        let text = "a<MARKER>b";

        assert_eq!(
            tokenizer.encode(text),
            text.bytes().map(u32::from).collect::<Vec<_>>()
        );
        assert_eq!(tokenizer.decode(&tokenizer.encode(text)), text);
    }

    #[test]
    fn merges_are_applied_by_learned_priority_until_no_pair_remains() {
        let tokenizer = tokenizer_with_merges([
            ((u32::from(b'a'), u32::from(b'b')), 256),
            ((256, u32::from(b'c')), 257),
            ((u32::from(b'b'), u32::from(b'c')), 258),
        ]);

        assert_eq!(tokenizer.encode("abc"), vec![257]);
        assert_eq!(tokenizer.decode(&[257]), "abc");
    }

    #[test]
    fn a_merge_replaces_non_overlapping_occurrences() {
        let tokenizer = tokenizer_with_merges([((u32::from(b'a'), u32::from(b'a')), 256)]);

        assert_eq!(tokenizer.encode("aaaaa"), vec![256, 256, u32::from(b'a')]);
    }

    #[test]
    fn angle_bracketed_text_can_be_merged() {
        let tokenizer = tokenizer_with_merges([((u32::from(b'<'), u32::from(b'E')), 256)]);

        assert_eq!(
            tokenizer.encode("<EX>"),
            vec![256, u32::from(b'X'), u32::from(b'>')]
        );
    }

    #[test]
    fn decode_reconstructs_base_and_merged_tokens_and_skips_unknown_ids() {
        let tokenizer = tokenizer_with_merges([((u32::from(b'H'), u32::from(b'i')), 256)]);

        assert_eq!(tokenizer.decode(&[256, u32::from(b'!'), 999]), "Hi!");
    }

    #[test]
    fn init_encodes_input_as_utf8_bytes() {
        let text = "a<MARKER><MARKER>b";
        let tokens = init(text.as_bytes());

        assert_eq!(tokens, text.bytes().map(u32::from).collect::<Vec<_>>());
    }

    #[test]
    fn merge_ids_replaces_pairs_without_overlapping() {
        let mut tokens = vec![1, 1, 1, 1, 1];

        merge_ids(&mut tokens, &1, &1, 2);

        assert_eq!(tokens, vec![2, 2, 1]);
    }

    #[test]
    fn training_iteration_selects_the_most_frequent_pair() {
        let state = TokenizerState {
            merge: HashMap::new(),
            data_as_tokens: vec![
                u32::from(b'a'),
                u32::from(b'b'),
                u32::from(b'a'),
                u32::from(b'b'),
            ],
            next_id: 256,
        };

        let (was_merged, state) = iteration(state);

        assert!(was_merged);
        assert_eq!(
            state.merge.get(&(u32::from(b'a'), u32::from(b'b'))),
            Some(&256)
        );
        assert_eq!(state.data_as_tokens, vec![256, 256]);
        assert_eq!(state.next_id, 257);
    }

    #[test]
    fn trained_tokenizer_roundtrips_text_and_respects_target_vocab_size() {
        let text = "abababab";
        let state = train(text.as_bytes(), 258);

        assert!(state.next_id <= 258);

        let tokenizer = build_tokenizer(state);
        let ids = tokenizer.encode(text);
        assert_eq!(tokenizer.decode(&ids), text);
    }

    fn base_tokenizer() -> BpeTokenizer {
        build_tokenizer(TokenizerState {
            merge: HashMap::new(),
            data_as_tokens: Vec::new(),
            next_id: 256,
        })
    }

    fn tokenizer_with_merges<const N: usize>(merges: [((u32, u32), u32); N]) -> BpeTokenizer {
        let next_id = merges
            .iter()
            .map(|(_, id)| *id)
            .max()
            .map_or(256, |id| id + 1);
        build_tokenizer(TokenizerState {
            merge: HashMap::from(merges),
            data_as_tokens: Vec::new(),
            next_id,
        })
    }
}
