# Verifikasjon av simple-llm

Se [`BEGREPER.md`](./BEGREPER.md) for en trinnvis forklaring av dataflyten og LLM-begrepene som brukes her og i kildekoden.

Koden implementerer en enkel autoregressiv språkmodell:

1. Tokenizer lager tokens av teksten.
2. Forward pass beregner logits for neste token.
3. Cross-entropy-gradienten beskriver avviket fra riktig token.
4. Backpropagation sender gradienten gjennom lagene.
5. SGD oppdaterer vektene.
6. Generering legger til ett token om gangen.

## Enhetstester

```bash
cargo test --quiet
```

Testene for `linear_layer` og `self_attatention_layer` sammenligner `backward()` med numeriske gradienter beregnet med finite differences. De gjentar derfor ikke bare formlene i implementasjonen.

Testene bekrefter også at samme `seed` gir identiske vekter og prediksjoner, mens ulike seeds gir ulike startvekter.

## Loss synker under trening

Testen `training_reduces_next_token_loss` måler cross-entropy-loss for den samme contexten før og etter trening. Den bekrefter at loss blir lavere med fast seed og treningsdata. `train_model` beregner gradienten direkte og trenger ikke selve loss-tallet for å oppdatere vektene.

## Attention bruker tidligere kontekst

Testdataene inneholder dette mønsteret:

```text
the cat sat on the mat. the dog sat on the rug.
```

Et kort context-vindu ser bare den tvetydige frasen `sat on the`:

```bash
cargo run --release -- \
  -seed=42 \
  -seq-len=3 \
  -d-model=8 \
  -epochs=2000 \
  -learning-rate=0.05 \
  verifikasjon_test_data.txt \
  "the cat sat on the"
```

Modellen kan da ikke vite om neste token skal være `mat` eller `rug`.

Med `seq-len=5` ser den også `cat`:

```bash
cargo run --release -- \
  -seed=42 \
  -seq-len=5 \
  -d-model=8 \
  -epochs=3000 \
  -learning-rate=0.05 \
  verifikasjon_test_data.txt \
  "the cat sat on the"
```

Det første nye tokenet blir `mat`. Modellen bruker dermed informasjon fra tidligere i contexten, ikke bare siste token.

Med vocabulary på 8 tokens og `d-model=8` har modellen 320 trenbare parametere:

- embedding: 64
- attention: 192
- output: 64

## Vurdering

Verifikasjonen viser at:

- gradientene stemmer med finite differences
- loss synker under trening
- modellen lærer å predikere neste token
- attention kan bruke tidligere tokens i contexten
- samme seed gir reproduserbare resultater
- trace rangerer kandidater med softmax og velger høyeste verdi med `argmax`

Modellen er med vilje begrenset. Den mangler blant annet positional encoding, layer normalization, multi-head attention, feed-forward-lag, stablede transformerblokker, batching og stopptoken.
