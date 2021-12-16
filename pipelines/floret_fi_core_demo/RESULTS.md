## Results

With 1M (2.6G) tokenized training texts and 50K 300-dim vectors, ~300K keys for
the standard vectors:

| Vectors                                      |  TAG |  POS | DEP UAS | DEP LAS | NER F |
| -------------------------------------------- | ---: | ---: | ------: | ------: | ----: |
| none                                         | 93.5 | 92.4 |    80.1 |    73.0 |  61.6 |
| standard (pruned: 50K vectors for 300K keys) | 95.9 | 95.0 |    83.1 |    77.4 |  68.1 |
| standard (unpruned: 300K vectors/keys)       | 96.4 | 95.0 |    82.8 |    78.4 |  70.4 |
| floret (minn 4, maxn 5; 50K vectors, no OOV) | 96.9 | 95.9 |    84.5 |    79.9 |  70.1 |
