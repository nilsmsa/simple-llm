pub trait Tokenizer {
    /// Gjør tekst om til numeriske token-ID-er som modellen kan behandle.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Gjør token-ID-er om til lesbar tekst igjen.
    fn decode(&self, ids: &[u32]) -> String;

    /// Returnerer hvor mange forskjellige tokens modellen kan predikere.
    fn vocab_size(&self) -> usize;
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Token {
    id: u32,
    from_inclusive: usize,
    to_exclusive: usize,
}

impl Token {
    /// Oppretter et token med ID og halvt åpent spenn i originalteksten.
    pub fn new(id: u32, from_inclusive: usize, to_exclusive: usize) -> Self {
        Token {
            id,
            from_inclusive,
            to_exclusive,
        }
    }
}
