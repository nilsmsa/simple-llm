# Slik lærer modellen

Dette dokumentet følger dataene gjennom modellen i samme rekkefølge som
koden. Et begrep forklares før det brukes til å forklare neste steg.

Se [`KOMMUNER_STEG_FOR_STEG.md`](./KOMMUNER_STEG_FOR_STEG.md) for et konkret regneeksempel som følger én feil prediksjon gjennom backpropagation.

Hele training-flyten kan oppsummeres slik:

```text
tekst
  -> tokens
  -> embedding
  -> attention
  -> logits
  -> sannsynligheter
  -> cross-entropy-gradient
  -> gradients bakover gjennom de samme lagene
  -> oppdaterte vekter
```

## 1. Tekst blir training-eksempler

### Token og vocabulary

Et token er enheten modellen leser og predikerer. Med word-tokenizeren er et
token vanligvis et ord eller et tegn. Vocabulary er alle tokenene modellen
kjenner.

Tokenizerens `encode` erstatter hvert token med en numerisk token-ID:

```text
bergen kommune ligger i vestland
   0       1       2   3    4
```

Modellen regner bare med ID-er og tallvektorer. `decode` gjør token-ID-er om
til tekst igjen.

### BPE

Byte Pair Encoding starter med én token-ID per byte. BPE-treningen finner
bytepar som forekommer ofte og erstatter hvert par med ett nytt token.

Dette endrer hvordan teksten deles opp, men ikke oppgaven til modellen:
Den skal fortsatt predikere neste token.

### Context og target

Context er tokenene modellen får se. Target er tokenet den skal lære å predikere.

Med `seq_len=4` lager `sliding_windows` blant annet dette eksempelet:

```text
context                              target
[bergen, kommune, ligger, i]    ->   vestland
```

Vinduet flyttes ett token om gangen. Én tekst gir derfor mange training-eksempler.

## 2. Embedding gjør tokens om til vektorer

En token-ID inneholder ingen mening i seg selv. ID 4 er ikke mer lik ID 5 enn ID 100.

Embedding-laget gir derfor hvert token en "trenbar" vektor. Med `d_model=3` kan et token for eksempel starte med:

```text
bergen -> [0.7, -0.2, 1.1]
```

Tallene starter tilfeldig. Treningen endrer dem slik at vektorene blir nyttige for neste-token-prediksjon.

`d_model` er antall tall i hver slik representasjon. En større verdi gir modellen mer kapasitet, men også flere vekter å trene.

## 3. Attention lager en kontekstrepresentasjon

Self-attention lar hvert token hente informasjon fra andre tokens i samme context. Hver embedding projiseres til tre roller.

### Query

Query beskriver hva tokenet leter etter.

### Key

Key beskriver hva tokenet kan matches på. En query sammenlignes med alle keys
for å finne relevante tokens.

### Value

Value er informasjonen tokenet sender videre hvis det blir vurdert som
relevant.

En huskeregel:

```text
query = hva leter jeg etter?
key   = hva kan jeg matches på?
value = hva sender jeg videre?
```

### Attention-score

`compute_scores` sammenligner hver query med hver key ved hjelp av dot product.
Vektorer som peker i samme retning, får høy score.

Scoren deles på kvadratroten av `d_model`. Denne skaleringen hindrer at større
vektorer gir svært store tall og ustabil softmax.

### Causal mask

Modellen skal predikere framtidige tokens uten å se dem. En causal mask setter
scoren til alle framtidige posisjoner til minus uendelig.

Et token kan dermed bare bruke seg selv og tokens som står tidligere i
teksten. Uten masken kunne modellen ha sett fasiten under training.

### Attention-softmax

`softmax_rows` gjør scorene om til attention-sannsynligheter som summerer til
1. En sannsynlighet på `0.7` betyr at det aktuelle tokenet får 70 prosent av
oppmerksomheten i den raden.

### Context-vektor

`mix_values` multipliserer hver value med attention-sannsynligheten sin og
summerer resultatene.

Resultatet er en context-vektor som inneholder en vektet blanding av
informasjon fra de synlige tokenene.

## 4. Output-laget lager logits

Modellen bruker context-vektoren ved siste posisjon til å vurdere neste token.
Output-laget beregner én logit for hvert token i vocabulary:

```text
vestland:   2.8
trøndelag:  0.4
troms:     -0.7
```

En logit er en rå score, ikke en sannsynlighet. Høyere logit betyr at modellen
foretrekker tokenet.

Softmax gjør logitene om til sannsynligheter som summerer til 1:

```text
vestland:   0.86
trøndelag:  0.10
troms:      0.04
```

Under generering velger `argmax` tokenet med høyest logit. Virkelige LLM-er
kan i stedet sample fra sannsynlighetsfordelingen for å få mer variasjon.

## 5. Loss måler hvor feil prediksjonen var

Target representeres som one-hot. Riktig token får verdien 1, og alle andre
får 0:

```text
target = vestland

vestland:   1
trøndelag:  0
troms:      0
```

Cross-entropy-loss ser på sannsynligheten modellen ga riktig token:

- høy sannsynlighet for riktig token gir lav loss
- lav sannsynlighet for riktig token gir høy loss

Loss er ett tall for hele prediksjonen. Tallet forteller hvor dårlig svaret
var, men ikke direkte hvilke vekter som må endres. Det er jobben til
gradientene.

`cross_entropy_loss` kan brukes til å måle fremgangen. `train_model` trenger
ikke selve loss-tallet i hver runde og går derfor direkte fra logits og target
til gradienten med `cross_entropy_derivative`. Det gir samme vektoppdatering.

## 6. En gradient beskriver hvordan loss kan reduseres

En gradient forteller hvordan en liten endring i en verdi påvirker loss:

- positiv gradient: høyere verdi gir høyere loss
- negativ gradient: høyere verdi gir lavere loss
- stor absoluttverdi: verdien påvirker loss mye
- verdi nær 0: verdien påvirker loss lite

SGD flytter vekten i motsatt retning av gradienten:

```text
ny vekt = gammel vekt - learning_rate × gradient
```

`learning_rate` bestemmer hvor stort steget er.

## 7. Den første gradienten kommer fra cross-entropy

Backpropagation må starte med en gradient for modellens siste output:
logitene.

`cross_entropy_derivative` lager denne gradienten ved å trekke target fra
sannsynligheten for hvert token:

```text
sannsynligheter: [0.10, 0.70, 0.20]
target:           [0.00, 1.00, 0.00]
gradient:         [0.10, -0.30, 0.20]
```

De positive verdiene sier at logitene til feil tokens bør ned. Den negative
verdien sier at logiten til riktig token bør opp.

I `train_model` heter denne verdien først `gradients`:

```rust
let gradients = cross_entropy_derivative(&predictions, &targets);
```

Den sendes så til output-lagets `backward` som argumentet `grad_output`:

```rust
let d_last_token = model.linear.backward(last_token_vector, &gradients);
```

Dette er altså opprinnelsen til `grad_output` i `LinearLayer::backward`.

## 8. `grad_output` og `grad_input` er relative navn

Se på ett vilkårlig lag:

```text
input -> [ lag ] -> output -> resten av modellen -> loss
```

Under backpropagation går feilsignalet motsatt vei:

```text
grad_input <- [ lag ] <- grad_output <- loss
```

### `grad_output`

`grad_output` beskriver hvordan lagets output påvirket loss. Verdien kommer
fra beregningen eller laget etter.

### `grad_input`

`grad_input` beskriver hvordan lagets input påvirket loss. Laget beregner
denne verdien fra `grad_output` og sine egne vekter, og sender den til laget
før.

Navnene avhenger derfor av laget:

| Lag | `grad_output` gjelder | Returnert `grad_input` gjelder |
|-----|-----------------------|-------------------------------|
| Output-lag | logits | siste context-vektor |
| Attention | context-vektorene | embedding-vektorene |
| Embedding | embedding-vektorene | Ingenting tidligere lag |

## 9. Backpropagation gjennom output-laget

`LinearLayer::backward` mottar:

```rust
input: &[f32]
grad_output: &[f32]
```

`input` er context-vektoren som laget brukte i forward pass.
`grad_output` er gradienten for logitene fra cross-entropy.

Funksjonen beregner to resultater:

1. `weight_gradients` for output-lagets egne vekter
2. gradienten for context-vektoren som kom inn

Gradienten for context-vektoren returneres som `d_last_token`. Navn med
prefikset `d_` betyr her «gradienten med hensyn til denne verdien».

## 10. Gradient for hele token-sekvensen

Output-laget brukte bare context-vektoren ved siste posisjon. Derfor gjelder
`d_last_token` bare denne posisjonen.

`train_model` lager `d_context_sequence`, som har plass til alle posisjonene.
Alle verdier starter på 0, og `d_last_token` kopieres inn på siste posisjon:

```text
tidligere posisjoner: 0
siste posisjon:       d_last_token
```

Denne listen blir `grad_output` til `SelfAttentionLayer::backward`.

## 11. Backpropagation gjennom attention

Attention-laget lagret `input`, query, key, value og
attention-sannsynlighetene fra forward pass i en cache. Backward pass trenger
disse verdiene for å følge regnestykkene i motsatt rekkefølge.

`SelfAttentionLayer::backward` finner:

1. hvordan value-vektorene påvirket context-vektorene
2. hvordan attention-sannsynlighetene påvirket blandingen av values
3. hvordan attention-scorene påvirket sannsynlighetene
4. hvordan query og key påvirket scorene
5. hvordan embedding-vektorene påvirket query, key og value

Underveis lagres gradients for `w_q`, `w_k` og `w_v`. Returverdien
`d_embedded` beskriver hvordan alle embedding-vektorene påvirket loss.

`d_embedded` blir dermed `grad_output` til embedding-lagets `backward`.

## 12. Backpropagation gjennom embedding

`EmbeddingLayer::backward` kobler hver del av `d_embedded` til token-ID-en som
opprinnelig hentet embedding-vektoren.

Hvis samme token forekommer flere ganger, summeres gradientene. Laget
returnerer ikke en ny gradient fordi token-ID-er er heltall og det ikke finnes
noe tidligere trenbart lag.

Nå finnes det gradients for alle modellens trenbare vekter:

- embedding-tabellen
- query-, key- og value-matrisene
- output-matrisen

## 13. Vektene oppdateres

Hvert lag kaller `update_weights`:

```text
vekt = vekt - learning_rate × gradient
```

Gradientene nullstilles etter oppdateringen, slik at neste training-eksempel
starter uten rester fra det forrige.

Én epoch er én full gjennomgang av alle training-vinduene. Flere epochs betyr
at modellen får flere muligheter til å justere vektene.

## 14. Generering bruker bare forward pass

Under generering finnes ingen target og dermed ingen loss eller
backpropagation.

`predict_tokens` gjentar denne prosessen:

1. Encode prompten.
2. Kjør embedding, attention og output-laget.
3. Velg tokenet med høyest logit.
4. Legg tokenet til context.
5. Bruk den utvidede contexten til neste prediksjon.

Dette kalles autoregressiv generering: Modellen bruker sin egen output som
input i neste runde.

## 15. Seed og reproduserbarhet

Embedding- og output-vektene starter tilfeldig. En seed bestemmer hvilke
startverdier som brukes.

Samme seed, programversjon, plattform, treningsdata og parametere gir samme
resultat. Seed gjør forsøket reproduserbart, men gjør ikke modellen bedre.
