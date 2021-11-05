## Results

With 1M (2.6G) tokenized training texts and 50K 300-dim vectors, ~300K keys for
the standard vectors:

| Vectors                                      |  TAG |  POS | DEP UAS | DEP LAS | NER F |
| -------------------------------------------- | ---: | ---: | ------: | ------: | ----: |
| none                                         | 93.3 | 92.3 |    79.7 |    72.8 |  61.0 |
| standard (pruned: 50K vectors for 300K keys) | 95.9 | 94.7 |    83.3 |    77.9 |  68.5 |
| standard (unpruned: 300K vectors/keys)       | 96.0 | 95.0 |    83.8 |    78.4 |  69.1 |
| floret (minn 4, maxn 5; 50K vectors, no OOV) | 96.6 | 95.5 |    83.5 |    78.5 |  70.9 |
