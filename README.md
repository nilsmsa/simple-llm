# simple-llm

En liten, pedagogisk språkmodell skrevet i Rust uten et maskinlæringsrammeverk. Prosjektet viser den sentrale treningsløkken bak autoregressive språkmodeller, men er ikke en representativ moderne LLM-arkitektur.

Modellen inneholder:

- ord- og BPE-tokenisering
- trenbare embedding-vektorer
- causal self-attention
- residualforbindelse rundt attention-laget
- et lineært output-lag
- cross-entropy og backpropagation
- oppdatering av vekter med SGD

## Hva modellen viser

Modellen gjør den grunnleggende prosessen konkret:

```text
bygg vocabulary -> initialiser vekter -> gjett neste token
-> mål feilen -> juster vektene -> gjenta
```

Token-ID-en er bare en indeks. Det er embeddingene og de øvrige modellvektene som inneholder tallene treningen endrer. Under generering lager modellen en logit for hvert mulig neste token, gjør logitene om til en sannsynlighetsfordeling og velger tokenet med høyest verdi.

Modellen har ingen egen sannhetskontroll eller «det vet jeg ikke»-mekanisme. Den velger alltid et token med `argmax`, selv når alle alternativene bygger på et svakt grunnlag. Informasjon kan være kodet i vektene, men neste-token-mekanismen avgjør ikke om en påstand er sann.

Dette er først og fremst et læreprosjekt, ikke en modell beregnet for praktisk bruk. Den mangler blant annet positional encoding, layer normalization, multi-head attention, feed-forward-lag, stablede transformerblokker og stopptoken. Etter en fornuftig begynnelse fortsetter den derfor ofte med usammenhengende tekst til grensen på antall tokens.

## Kom i gang

Du trenger en nyere stabil versjon av Rust.

```bash
cargo test
cargo run --release -- -epochs=6000 -d-model=8 -seq-len=4 -learning-rate=0.001 -seed=42 kommuner_demo.txt "bergen ligger i"
```

Legg til `-trace` for å vise de tre høyest rangerte neste-token-kandidatene i de tre første genereringsstegene:

```bash
cargo run --release -- -trace -epochs=6000 -d-model=8 -seq-len=4 -learning-rate=0.001 -seed=42 kommuner_demo.txt "bergen ligger i"
```

Trace-verdiene er token-sannsynligheter, ikke sannsynligheten for at teksten er sann.

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

Rydding, refaktorering og den første testrunden ble gjort med Qwen3-Coder-Next kjørende lokalt sammen med OpenCode. Det er en kodeorientert mixture-of-experts-modell med 80 milliarder parametere totalt, hvor omtrent 3 milliarder er aktive per token. Modellen har en hybridarkitektur med Gated DeltaNet, attention og MoE-lag, og støtter en kontekst på opptil 256 000 tokens. Under arbeidet med dette prosjektet ble en kontekstlengde på 216 000 tokens brukt.

Word-tokenizeren ble i sin helhet skrevet av Qwen3-Coder-Next. Den ble lagt til etter at BPE-tokenizeren viste seg å fungere dårlig for denne svært lille demo-modellen. Med lite treningsdata lærer BPE få nyttige sammenslåinger. Hele ord gir mer lesbare sekvenser og passer derfor bedre til å demonstrere dataflyten, selv om en slik tokenizer ikke håndterer ukjente ord like godt.

Den siste finpussen – en ny runde med validering og testing – samt dokumentasjonen ble laget med Nav Pilot, GPT-5.6 Sol og Sonnet 5.

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
9. **Tokenvalg:** Bruk `-trace` til å vise at modellen alltid rangerer og velger et neste token.
10. **Verifikasjon:** Kjør en kort demo og vis at testene sammenligner backpropagation med numeriske gradienter.
