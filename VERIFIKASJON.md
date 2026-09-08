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

Kommandoen rapporterer 106 beståtte testkjøringer. Testene for `linear_layer` og `self_attatention_layer` sammenligner `backward()` med numeriske gradienter beregnet med finite differences. De gjentar derfor ikke bare formlene i implementasjonen.

Testene bekrefter også at samme `seed` gir identiske vekter og prediksjoner, mens ulike seeds gir ulike startvekter.

## Loss synker under trening

Modellen ble trent på [`verifikasjon_test_data.txt`](./verifikasjon_test_data.txt) med `seed=42`, `d-model=8`, `seq-len=3` og `learning-rate=0.05`.

| Epoch | Gjennomsnittlig cross-entropy-loss |
|------:|-----------------------------------:|
| 0 | 1.6059653 |
| 5 | 0.76554537 |
| 10 | 0.4641654 |
| 15 | 0.20894644 |
| 19 | 0.16499466 |

Denne separate målingen viser at loss synker når gradient descent forbedrer vektene. `train_model` beregner gradienten direkte og trenger ikke selve loss-tallet for å oppdatere vektene.

## Attention bruker tidligere kontekst

Testdataene inneholder dette mønsteret:

```text
the cat sat on the mat. the dog sat on the rug.
```

Et kort kontekstvindu ser bare den tvetydige frasen `sat on the`:

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

Det første nye tokenet blir `mat`. Modellen bruker dermed informasjon fra tidligere i konteksten, ikke bare siste token.

Med vocabulary på 9 tokens og `d-model=8` har modellen 336 trenbare parametere:

- embedding: 72
- attention: 192
- output: 72

## Vurdering

Verifikasjonen viser at:

- gradientene stemmer med finite differences
- loss synker under trening
- modellen lærer å predikere neste token
- attention kan bruke tidligere tokens i konteksten
- samme seed gir reproduserbare resultater

Modellen er med vilje begrenset. Den mangler blant annet positional encoding, multi-head attention, feed-forward-lag, batching og stopptoken.
