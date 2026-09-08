use std::collections::HashMap;

use crate::tokenizer::Tokenizer;

pub struct WordTokenizer {
    vocab: Vec<String>,
    index: HashMap<String, u32>,
}

/// Gjenkjenner tegn som avslutter et ord i den enkle tokenizeren.
pub fn is_delimiter(character: char) -> bool {
    character.is_whitespace()
        || character == ','
        || character == '.'
        || character == '!'
        || character == '?'
}

/// Deler tekst i ord og egne tokens for enkel tegnsetting.
///
/// Denne enkle tokenizeren gjør dataflyten lett å følge, men kjenner bare ord
/// som finnes nøyaktig slik i treningsdataene.
pub fn split_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_whitespace() {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
        } else if character == ',' || character == '.' || character == '!' || character == '?' {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
            words.push(character.to_string());
        } else {
            current.push(character);
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

impl WordTokenizer {
    /// Bygger vocabulary og gir hvert unike token en numerisk ID.
    ///
    /// ID-ene tildeles i samme rekkefølge som tokenene først forekommer i
    /// treningsdataene.
    pub fn build(text: &str) -> Self {
        let mut vocab: Vec<String> = Vec::new();
        let mut index: HashMap<String, u32> = HashMap::new();

        let mut next_id: u32 = 0;
        for word in split_words(text) {
            if !index.contains_key(&word) {
                index.insert(word.clone(), next_id);
                next_id += 1;
                vocab.push(word);
            }
        }
        WordTokenizer { vocab, index }
    }

    /// Slår opp teksten til en token-ID hvis den finnes i vocabulary.
    pub fn word(&self, id: usize) -> Option<&str> {
        self.vocab.get(id).map(String::as_str)
    }
}

impl Tokenizer for WordTokenizer {
    /// Slår opp hvert kjent ord som en token-ID.
    ///
    /// Ukjente ord hoppes over fordi denne pedagogiske modellen ikke har et
    /// eget `unknown`-token.
    fn encode(&self, text: &str) -> Vec<u32> {
        split_words(text)
            .into_iter()
            .filter_map(|word| self.index.get(&word).copied())
            .collect()
    }

    /// Gjør token-ID-er om til tekst og fester tegnsetting til forrige ord.
    fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for &id in ids {
            if let Some(token) = self.vocab.get(id as usize) {
                if !text.is_empty() && !matches!(token.as_str(), "," | "." | "!" | "?") {
                    text.push(' ');
                }
                text.push_str(token);
            }
        }
        text
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delimiter_recognizes_whitespace_and_supported_punctuation() {
        for delimiter in [' ', '\t', '\n', ',', '.', '!', '?'] {
            assert!(
                is_delimiter(delimiter),
                "{delimiter:?} should be a delimiter"
            );
        }
        for non_delimiter in ['a', '7', '-', '\'', 'å'] {
            assert!(
                !is_delimiter(non_delimiter),
                "{non_delimiter:?} should not be a delimiter"
            );
        }
    }

    #[test]
    fn split_words_separates_words_and_each_punctuation_mark() {
        assert_eq!(
            split_words("Hei,\tverden! Hva? Ja..."),
            vec!["Hei", ",", "verden", "!", "Hva", "?", "Ja", ".", ".", "."]
        );
    }

    #[test]
    fn split_words_preserves_unicode_inside_words_and_ignores_empty_segments() {
        assert_eq!(
            split_words("  blåbær   og\n crème brûlée  "),
            vec!["blåbær", "og", "crème", "brûlée"]
        );
        assert!(split_words(" \t\n").is_empty());
    }

    #[test]
    fn build_assigns_unique_tokens_in_first_seen_order() {
        let tokenizer = WordTokenizer::build("hei verden hei !");

        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(tokenizer.word(0), Some("hei"));
        assert_eq!(tokenizer.word(1), Some("verden"));
        assert_eq!(tokenizer.word(2), Some("!"));
        assert_eq!(tokenizer.word(3), None);
    }

    #[test]
    fn encode_uses_vocabulary_ids_for_words_and_punctuation() {
        let tokenizer = WordTokenizer::build("Hei, verden! Hei?");

        assert_eq!(tokenizer.encode("Hei,\nverden?"), vec![0, 1, 2, 4]);
    }

    #[test]
    fn encode_skips_unknown_tokens_and_is_case_sensitive() {
        let tokenizer = WordTokenizer::build("kjent !");

        assert_eq!(tokenizer.encode("ukjent kjent KJENT !"), vec![0, 1]);
    }

    #[test]
    fn angle_bracketed_text_is_treated_as_an_ordinary_word() {
        let tokenizer = WordTokenizer::build("før <MARKER> etter");

        let ids = tokenizer.encode("før <MARKER> etter");

        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(tokenizer.decode(&ids), "før <MARKER> etter");
    }

    #[test]
    fn decode_joins_words_and_attaches_punctuation() {
        let tokenizer = WordTokenizer::build("Hei, verden!");

        assert_eq!(tokenizer.decode(&[0, 1, 2, 3]), "Hei, verden!");
        assert_eq!(tokenizer.decode(&[0, 1, 99, 2, 3]), "Hei, verden!");
    }

    #[test]
    fn encode_decode_roundtrip_normalizes_whitespace_without_changing_tokens() {
        let tokenizer = WordTokenizer::build("Hei, verden! Hvordan går det?");
        let input = "Hei,\n\nverden!  Hvordan\tgår det?";

        let decoded = tokenizer.decode(&tokenizer.encode(input));

        assert_eq!(decoded, "Hei, verden! Hvordan går det?");
        assert_eq!(split_words(&decoded), split_words(input));
    }
}
