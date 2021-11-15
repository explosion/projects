## Results

With 1M (3.3G) tokenized training texts and 50K 100-dim vectors, ~800K keys for
the standard vectors:

| Vectors                                      |  TAG |  POS | DEP UAS | DEP LAS |
| -------------------------------------------- | ---: | ---: | ------: | ------: |
| none                                         | 72.5 | 85.3 |    74.0 |    65.0 |
| standard (pruned: 50K vectors for 800K keys) | 77.3 | 89.1 |    78.2 |    72.2 |
| standard (unpruned: 800K vectors/keys)       | 79.0 | 90.3 |    79.4 |    73.9 |
| floret (minn 2, maxn 3; 50K vectors, no OOV) | 82.8 | 94.1 |    83.5 |    80.5 |
