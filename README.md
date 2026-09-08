# simple-llm

En liten, pedagogisk språkmodell skrevet i Rust uten et maskinlæringsrammeverk. Prosjektet viser hvordan en autoregressiv modell kan trenes til å forutsi neste token.

Modellen inneholder:

- ord- og BPE-tokenisering
- trenbare embedding-vektorer
- causal self-attention
- et lineært output-lag
- cross-entropy og backpropagation
- oppdatering av vekter med SGD

Dette er først og fremst et læreprosjekt, ikke en modell beregnet for praktisk bruk.

## Kom i gang

Du trenger en nyere stabil versjon av Rust.

```bash
cargo test
cargo run --release -- veldig_enkel_tekst.txt "a b"
```

Eksempel med BPE-tokenisering:

```bash
cargo run --release -- \
  -tokenizer=bpe \
  -vocab=300 \
  -seq-len=20 \
  veldig_enkel_tekst.txt \
  "a b"
```

Se [BEGREPER.md](./BEGREPER.md) for en forklaring av dataflyten og [VERIFIKASJON.md](./VERIFIKASJON.md) for tester og målte resultater.

## Hvordan prosjektet ble til

Det meste av koden ble skrevet for hånd, med Google Gemini 3.1 Pro Advanced Thinking i nettleseren som sparringspartner og instruktør.

Rydding, refaktorering og den første testrunden ble gjort med Qwen3-Coder-Next. Det er en kodeorientert mixture-of-experts-modell med 80 milliarder parametere totalt, hvor omtrent 3 milliarder er aktive per token. Modellen har en hybridarkitektur med Gated DeltaNet, attention og MoE-lag, og støtter en kontekst på opptil 256 000 tokens. Under arbeidet med dette prosjektet ble en kontekstlengde på 216 000 tokens brukt.

Word-tokenizeren ble i sin helhet skrevet av Qwen3-Coder-Next. Den ble lagt til etter at BPE-tokenizeren viste seg å fungere dårlig for denne svært lille demo-modellen. Med lite treningsdata lærer BPE få nyttige sammenslåinger og lager ofte lengre token-sekvenser. Det gir modellen flere steg å lære og gjør resultatene vanskeligere å forstå. Hele ord gir kortere, mer lesbare sekvenser og passer derfor bedre til å demonstrere dataflyten, selv om en slik tokenizer ikke håndterer ukjente ord like godt.

Siste finish, en ny runde med validering og testing samt dokumentasjonen ble gjort med Nav Pilot, GPT-5.6 Sol og Sonnet 5.

## Forslag til presentasjon

Presenter først hvordan modellen lager en prediksjon, og følg deretter feilsignalet bakover. Da er hvert lag kjent før backpropagation forklares.

1. **Målet:** Modellen får noen tokens og skal forutsi det neste.
2. **Tokenisering:** Vis hvordan tekst blir token-ID-er. Bruk word-tokenizeren først, og presenter BPE som et mer realistisk, men mindre oversiktlig alternativ.
3. **Treningsdata:** Vis hvordan et sliding window deler teksten i context og target.
4. **Forward pass:** Følg dataene gjennom embedding → self-attention → lineært output-lag → logits → sannsynligheter.
5. **Loss:** Sammenlign sannsynlighetene med riktig token og forklar cross-entropy som mål på hvor feil prediksjonen var.
6. **Backpropagation:** Gå motsatt vei, fra linear → attention → embedding. Forklar at hvert lag både beregner gradienter for egne vekter og sender et feilsignal videre bakover.
7. **Oppdatering:** Vis hvordan SGD justerer vektene litt før neste treningseksempel.
8. **Generering:** Avslutt med at modellen gjentar forward pass og bruker hvert predikerte token som del av neste context.
9. **Verifikasjon:** Kjør en kort demo og vis at testene sammenligner backpropagation med numeriske gradienter.
