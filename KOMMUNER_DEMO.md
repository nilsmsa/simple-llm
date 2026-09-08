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
  -tokenizer=word \
  -seed=42 \
  -epochs=3000 \
  -seq-len=4 \
  -d-model=8 \
  -learning-rate=0.05 \
  kommuner_demo.txt \
  "bergen ligger i"
```

Relevante linjer i output:

```text
Vocab size: 13
Trainable parameters: 400
  Embedding: 104
  Attention (Q, K, V): 192
  Output: 104
Predicted: 'bergen ligger i vestland ...'
```

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

## Begrensninger

Modellen kan ikke svare om kommuner eller ord som mangler i vocabulary. Word-tokenizeren skiller mellom store og små bokstaver og hopper over ukjente ord. Bruk derfor `bergen`, ikke `Bergen`.

Modellen mangler positional encoding. Resultatet viser at den kjenner igjen nyttige tokens, ikke at den forstår grammatikk eller ordstilling.

Bare det første fylkesnavnet vurderes. Uten stopptoken fortsetter modellen til grensen på 50 nye tokens, og resten blir ofte gjentakende.

Modellen har 400 trenbare parametere: 104 i embedding, 192 i attention og 104 i output.

Se [`KOMMUNER_STEG_FOR_STEG.md`](./KOMMUNER_STEG_FOR_STEG.md) for en konkret gjennomgang av forward pass, en feil prediksjon, backpropagation og predict-loopen.
