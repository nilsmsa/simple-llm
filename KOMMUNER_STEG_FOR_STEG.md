# Kommunedemoen steg for steg

Dette dokumentet følger én kjøring av [`kommuner_demo.txt`](./kommuner_demo.txt) fra tekst til trening og prediksjon.

Kommandoen er:

```bash
cargo run --release -- \
  -tokenizer=word \
  -seed=42 \
  -epochs=3000 \
  -seq-len=4 \
  -d-model=8 \
  -learning-rate=0.05 \
  kommuner_demo.txt \
  "bergen ligger i"
```

## 1. Tokenizeren bygger vocabulary

Word-tokenizeren går gjennom treningsfilen fra venstre mot høyre. Hvert nytt ord eller tegn får neste ledige ID:

| ID | Token |
|---:|-------|
| 0 | `bergen` |
| 1 | `kommune` |
| 2 | `ligger` |
| 3 | `i` |
| 4 | `vestland` |
| 5 | `fylke` |
| 6 | `.` |
| 7 | `trondheim` |
| 8 | `trøndelag` |
| 9 | `tromsø` |
| 10 | `troms` |
| 11 | `bodø` |
| 12 | `nordland` |

Vocabulary har dermed 13 tokens.

Den første setningen blir:

```text
bergen kommune ligger i vestland fylke .
   0       1       2   3     4      5  6
```

## 2. Modellen bygges

`d_model=8` betyr at hvert token representeres med åtte tall.

Modellen får 400 trenbare parametere:

| Lag | Form | Parametere |
|-----|------|-----------:|
| Embedding | `13 × 8` | 104 |
| Query | `8 × 8` | 64 |
| Key | `8 × 8` | 64 |
| Value | `8 × 8` | 64 |
| Output | `13 × 8` | 104 |
| **Totalt** | | **400** |

Embedding- og output-vektene initialiseres tilfeldig med `seed=42`. Query-, key- og value-vektene starter på `0.01`.

## 3. Sliding windows lager treningseksempler

`seq-len=4` betyr fire context-tokens og ett target-token per vindu.

De første vinduene er:

```text
[bergen, kommune, ligger, i]       -> vestland
[kommune, ligger, i, vestland]     -> fylke
[ligger, i, vestland, fylke]       -> .
[i, vestland, fylke, .]            -> trondheim
```

Treningsfilen inneholder 224 tokens. Det gir 220 overlappende vinduer per epoch. Med 3000 epochs utfører modellen:

```text
220 × 3000 = 660 000 vektoppdateringer
```

Resten av dette treningseksempelet følger det første vinduet:

```text
context = [bergen, kommune, ligger, i]
target  = vestland
```

## 4. Forward pass

### 4.1 Embedding

Embedding-laget slår opp åtte tall for hver token-ID:

```text
bergen  -> 8 tall
kommune -> 8 tall
ligger  -> 8 tall
i       -> 8 tall
```

De fire vektorene legges etter hverandre. Attention mottar derfor:

```text
4 tokens × 8 tall = 32 tall
```

### 4.2 Query, key og value

Hver embedding multipliseres med tre matriser:

```text
embedding × Wq -> query
embedding × Wk -> key
embedding × Wv -> value
```

Resultatet er fire query-vektorer, fire key-vektorer og fire value-vektorer. Hver vektor har åtte tall.

### 4.3 Attention-score

Hver query sammenlignes med hver key. Det gir en `4 × 4`-matrise med 16 scorer.

Causal mask skjuler tokens som ligger framover i teksten:

```text
           bergen  kommune  ligger  i
bergen       synlig   skjult  skjult  skjult
kommune      synlig   synlig  skjult  skjult
ligger       synlig   synlig  synlig  skjult
i            synlig   synlig  synlig  synlig
```

Softmax gjør de synlige scorene om til attention-sannsynligheter. Den siste posisjonen, `i`, kan hente informasjon fra alle fire tokens.

### 4.4 Context-vektor

Attention-sannsynlighetene brukes til å blande value-vektorene. Resultatet er én ny vektor per token.

Bare den siste context-vektoren brukes videre:

```text
context-vektor for i -> 8 tall
```

Denne vektoren skal inneholde informasjonen modellen trenger for å svare på:

```text
bergen kommune ligger i -> ?
```

### 4.5 Output og feil prediksjon

Output-laget gjør de åtte tallene om til 13 logits, én per token i vocabulary.

Tidlig i treningen er vektene nesten tilfeldige. Modellen kan derfor for eksempel gi høyest logit til `trøndelag`:

```text
vestland:   lavere logit
trøndelag:  høyeste logit  <- feil prediksjon
troms:      lavere logit
```

Target er `vestland`. Treningen må øke preferansen for `vestland` og redusere preferansen for de andre tokenene.

## 5. Ett konkret regnestykke med små vektorer

Den virkelige modellen bruker åtte tall per vektor og 13 mulige output-tokens. For å gjøre regnestykket lesbart bruker dette avsnittet:

- to tall per vektor
- tre mulige fylkestokens
- samme operasjoner og samme `learning_rate=0.05` som koden

Tallene er pedagogiske eksempelverdier, ikke en utskrift av vektene fra `seed=42`.

### 5.1 Attention blander values

Anta at siste token gir disse attention-sannsynlighetene:

```text
bergen:   0.4
kommune:  0.2
ligger:   0.1
i:        0.3
```

Og at value-vektorene er:

```text
bergen:   [ 0.7,  0.1]
kommune:  [ 0.2,  0.4]
ligger:   [-0.1,  0.3]
i:        [ 0.5, -0.2]
```

Context-vektoren blir det vektede gjennomsnittet:

```text
første tall:
0.4×0.7 + 0.2×0.2 + 0.1×(-0.1) + 0.3×0.5 = 0.46

andre tall:
0.4×0.1 + 0.2×0.4 + 0.1×0.3 + 0.3×(-0.2) = 0.09

context = [0.46, 0.09]
```

### 5.2 Output-laget velger feil fylke

Anta at output-vektene er:

```text
vestland:   [ 1.0, 0.5]
trøndelag:  [ 1.2, 0.1]
troms:      [-0.2, 0.3]
```

Hver logit er dot product mellom context og tokenets output-vekt:

```text
vestland:
0.46×1.0 + 0.09×0.5 = 0.505

trøndelag:
0.46×1.2 + 0.09×0.1 = 0.561  <- høyest, men feil

troms:
0.46×(-0.2) + 0.09×0.3 = -0.065
```

Softmax gir omtrent:

```text
vestland:   0.381
trøndelag:  0.403
troms:      0.216
```

Target er one-hot:

```text
vestland:   1
trøndelag:  0
troms:      0
```

Cross-entropy-loss for dette svaret ville vært omtrent:

```text
-ln(0.381) = 0.965
```

`train_model` beregner ikke dette tallet i hver runde. Tallet er nyttig for å måle fremgang, men vektoppdateringen trenger bare gradienten som beregnes i neste steg.

## 6. Backpropagation starter ved output

For softmax og cross-entropy er gradienten:

```text
grad_logits = sannsynlighet - target
```

Dermed får vi:

```text
vestland:   0.381 - 1 = -0.619
trøndelag:  0.403 - 0 =  0.403
troms:      0.216 - 0 =  0.216
```

Dette er `grad_output` til `LinearLayer::backward`.

Fortegnet forteller hva som bør skje:

- `vestland` har negativ gradient: logiten bør opp
- `trøndelag` har positiv gradient: logiten bør ned
- `troms` har positiv gradient: logiten bør ned

## 7. Output-vektene endres

Gradienten for én output-vekt er:

```text
grad_logit × context-verdi
```

For `vestland`:

```text
[-0.619×0.46, -0.619×0.09] = [-0.285, -0.056]
```

SGD-oppdateringen er:

```text
ny vekt = gammel vekt - learning_rate × gradient
```

Med `learning_rate=0.05` blir `vestland`-vektene:

```text
[1.0, 0.5] - 0.05×[-0.285, -0.056]
= [1.014, 0.503]
```

Begge vektene øker. Den samme context-vektoren vil derfor gi `vestland` en høyere logit neste gang.

For `trøndelag`:

```text
gradient:
[0.403×0.46, 0.403×0.09] = [0.185, 0.036]

nye vekter:
[1.2, 0.1] - 0.05×[0.185, 0.036]
= [1.191, 0.098]
```

Vektene reduseres, slik at `trøndelag` får en lavere logit i en lignende context.

## 8. Feilsignalet sendes til attention

Output-laget må også forklare hvordan context-vektoren bidro til feilen. Det beregner:

```text
grad_context = output-vekter transponert × grad_logits
```

Med vektene fra før oppdateringen:

```text
første tall:
1.0×(-0.619) + 1.2×0.403 + (-0.2)×0.216 = -0.178

andre tall:
0.5×(-0.619) + 0.1×0.403 + 0.3×0.216 = -0.204

grad_context = [-0.178, -0.204]
```

Dette er returverdien `d_last_token` fra `LinearLayer::backward`.

`train_model` legger verdiene på siste plass i `d_context_sequence`. Denne listen blir `grad_output` til `SelfAttentionLayer::backward`.

## 9. Attention-vektene endres

Context-vektoren var et vektet gjennomsnitt av values. Derfor får hver value-vektor sin andel av `grad_context`:

```text
bergen:
0.4×[-0.178, -0.204] = [-0.071, -0.082]

kommune:
0.2×[-0.178, -0.204] = [-0.036, -0.041]

ligger:
0.1×[-0.178, -0.204] = [-0.018, -0.020]

i:
0.3×[-0.178, -0.204] = [-0.053, -0.061]
```

`bergen` får størst gradient fordi attention ga tokenet størst betydning i forward pass.

Backward pass følger også attention-sannsynlighetene tilbake gjennom softmax og scorene. Det gir gradients for query og key:

- query-gradienten endrer hva siste token leter etter
- key-gradienten endrer hvilke tokens som ser relevante ut
- value-gradienten endrer informasjonen tokenene sender videre

Anta at én value-vekt var `0.25`, og at den samlede gradienten for vekten ble `-0.087`. Oppdateringen blir:

```text
0.25 - 0.05×(-0.087) = 0.254
```

Vekten øker fordi den bidro i en retning som bør forsterkes.

## 10. Embedding-vektene endres

Query, key og value ble alle beregnet fra embedding-vektorene. Gradientene fra de tre veiene summeres derfor til `d_embedded`.

Anta at de to første tallene i embedding for `bergen` var:

```text
[0.70, 0.10]
```

Og at samlet gradient fra query, key og value ble:

```text
[-0.03, 0.08]
```

Oppdateringen blir:

```text
[0.70, 0.10] - 0.05×[-0.03, 0.08]
= [0.702, 0.096]
```

Embedding for `bergen` er nå litt bedre tilpasset prediksjonen av `vestland` i denne contexten.

Etter oppdateringen nullstilles alle gradients. Neste sliding window starter et nytt forward pass med de nye vektene.

## 11. Hva 660 000 oppdateringer lærer

Ett eksempel flytter vektene svært lite. De gjentatte setningene og 3000 epochs gir samme mønster mange muligheter til å påvirke vektene:

```text
bergen    ... -> vestland
trondheim ... -> trøndelag
tromsø    ... -> troms
bodø      ... -> nordland
```

Modellen lærer ikke en regel om norsk geografi. Den justerer 400 tall slik at riktig fylke får høyest logit etter de observerte contextene.

## 12. Prediksjonsloopen

Treningen er nå ferdig. Under prediksjon finnes verken target, loss, backpropagation eller vektoppdatering.

Prompten er:

```text
bergen ligger i
```

Denne teksten finnes ikke ordrett i treningsfilen, fordi `kommune` mangler.

### Runde 1

Tokenizeren lager:

```text
[bergen, ligger, i]
[0,      2,      3]
```

`predict_tokens` gjør:

1. Slår opp tre embeddings: `3 × 8 = 24` tall.
2. Kjører self-attention med tre tokens.
3. Tar context-vektoren ved `i`.
4. Beregner 13 logits.
5. Velger høyeste logit med `argmax`.

Med `seed=42` og de dokumenterte parameterne er høyeste logit token 4:

```text
4 -> vestland
```

Tokenet legges til context:

```text
[bergen, ligger, i, vestland]
```

### Runde 2

Hele den nye sekvensen kjøres gjennom modellen igjen:

```text
[bergen, ligger, i, vestland] -> neste token
```

Modellen beregner nye embeddings, ny attention og nye logits. Det valgte tokenet legges til listen.

### Runde 3 til 50

Prosessen gjentas:

```text
context -> forward pass -> argmax -> legg til token
```

Contexten vokser med ett token per runde. Koden begrenser ikke prediksjon til `seq_len=4`; den bruker hele den voksende sekvensen.

Modellen har heller ikke et stopptoken. Den utfører derfor alle 50 rundene selv om første nye token allerede besvarte spørsmålet.

## 13. Forskjellen på trening og prediksjon

| Trening | Prediksjon |
|----------|------------|
| Har et kjent target | Har ikke target |
| Beregner cross-entropy-gradient fra target | Beregner ingen gradient |
| Kjører backward pass | Kjører bare forward pass |
| Endrer vektene | Holder vektene faste |
| Bruker vinduer med `seq_len=4` | Bruker hele den voksende contexten |
| Lærer fra riktig neste token | Bruker eget forrige svar som input |

Det viktigste skillet er at trening spør «hvor feil var svaret, og hvilke vekter bidro til feilen?». Prediksjon spør bare «hvilket token har høyest logit nå?».
