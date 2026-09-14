# Demo uten eksakt tekstmatch

Demoen viser at modellen kan predikere riktig selv om prompten ikke finnes ordrett i treningsdataene.

Treningsfilen inneholder:

```text
bergen kommune ligger i vestland fylke.
```

Prompten utelater `kommune`:

```text
bergen ligger i
```

Den korte teksten forekommer ikke i treningsfilen.

## Kjør demoen

```bash
cargo run --release -- \
  -trace \
  -tokenizer=word \
  -seed=42 \
  -epochs=3000 \
  -seq-len=4 \
  -d-model=8 \
  -learning-rate=0.05 \
  kommuner_demo.txt \
  "bergen ligger i"
```

Relevante, forkortede linjer i output:

```text
Vocab size: 13
Trainable parameters: 400
  Embedding: 104
  Attention (Q, K, V): 192
  Output: 104
Prediction trace:
Step 1 context: 'bergen ligger i'
  'vestland': 0.999992
  'troms': 0.000005
  'nordland': 0.000002
  other tokens: 0.000000
  selected with argmax: 'vestland'
Step 2 context: 'bergen ligger i vestland'
  'fylke': 0.999955
  ...
  selected with argmax: 'fylke'
Step 3 context: 'bergen ligger i vestland fylke'
  '.': 0.999982
  ...
  selected with argmax: '.'
Predicted: 'bergen ligger i vestland fylke. trondheim kommune ...'
```

Verdiene er avrundet til seks desimaler. De beskriver sannsynligheten for neste token innenfor modellens vocabulary, ikke sannsynligheten for at en geografisk påstand er sann. `argmax` velger alltid kandidaten med høyest verdi; det finnes ingen egen «ukjent»-handling.

Den relevante fullføringen har dermed endret seg slik:

```text
før:   bergen ligger i -> vestland + videre, usammenhengende tekst
etter: bergen ligger i -> vestland fylke. + videre, usammenhengende tekst
```

Modellen har fortsatt ikke et stopptoken og genererer derfor videre etter
punktum. Forbedringen er at den nå beholder nok informasjon om forrige token
til å fullføre den trente frasen `vestland fylke.` før den fortsetter.

Flere prompter:

| Prompt | Første nye token |
|--------|------------------|
| `trondheim ligger i` | `trøndelag` |
| `tromsø ligger i` | `troms` |
| `bodø ligger i` | `nordland` |

`seed=42` gir samme resultat med samme programversjon, plattform, treningsdata og parametere.

## Hvorfor det virker

Under trening ser modellen fire tokens før fylket:

```text
bergen kommune ligger i -> vestland
trondheim kommune ligger i -> trøndelag
tromsø kommune ligger i -> troms
bodø kommune ligger i -> nordland
```

Kommunen er det eneste som skiller eksemplene. Modellen lærer derfor en kobling mellom kommunen og fylket. Attention-laget kan gi kommunen betydning i contexten selv om `kommune` mangler under prediksjon.

Dette er begrenset generalisering over kjente tokens, ikke et eksakt tekstoppslag.

Attention-outputen har nå en residualforbindelse:

```text
ny representasjon = attention(input) + input
```

Attention-grenen kan hente kommunen fra tidligere i contexten, mens den direkte
residualveien bevarer embedding-representasjonen av tokenet på den aktuelle
posisjonen. Etter at modellen har generert `vestland`, er det derfor enklere for
output-laget å kjenne igjen at neste token skal være `fylke`, og deretter `.`.
Uten residualforbindelsen måtte all informasjon om det siste tokenet passere
gjennom value-projeksjonen og attention-blandingen. I denne lille modellen ble
signalet for svakt eller forvrengt, slik at bare `vestland` ble riktig før
genereringen sporet av.

Backward pass har den tilsvarende direkte gradientveien:

```text
grad_input = grad_attention + grad_output
```

Dermed lærer embeddingene både gjennom attention-beregningen og direkte fra
feilen i output-laget. Forward- og backward-pass beskriver med andre ord den
samme residualforbindelsen.

## Begrensninger

Modellen kan ikke svare om kommuner eller ord som mangler i vocabulary. Word-tokenizeren skiller mellom store og små bokstaver og hopper over ukjente ord. Bruk derfor `bergen`, ikke `Bergen`.

Modellen mangler positional encoding. Resultatet viser at den kjenner igjen nyttige tokens, ikke at den forstår grammatikk eller ordstilling.

Bare det første fylkesnavnet vurderes. Uten stopptoken fortsetter modellen til grensen på 50 nye tokens, og resten blir ofte gjentakende.

Modellen har 400 trenbare parametere: 104 i embedding, 192 i attention og 104 i output.

Se [`KOMMUNER_STEG_FOR_STEG.md`](./KOMMUNER_STEG_FOR_STEG.md) for en konkret gjennomgang av forward pass, en feil prediksjon, backpropagation og predict-loopen.
