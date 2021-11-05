## Results

With 1M (3.3G) tokenized training texts and 50K 300-dim vectors, ~800K keys for
the standard vectors:

| Vectors                                      |  TAG |  POS | DEP UAS | DEP LAS |
| -------------------------------------------- | ---: | ---: | ------: | ------: |
| none                                         | 72.5 | 85.0 |    73.2 |    64.3 |
| standard (pruned: 50K vectors for 800K keys) | 77.9 | 89.4 |    78.8 |    72.8 |
| standard (unpruned: 800K vectors/keys)       | 79.0 | 90.2 |    79.2 |    73.9 |
| floret (minn 2, maxn 3; 50K vectors, no OOV) | 82.5 | 93.8 |    83.0 |    80.1 |
