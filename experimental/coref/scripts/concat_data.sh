#!/usr/bin/env bash
# Concatenate training data. This is in a script to deal with Windows quoting issues.

set -eou pipefail

rm -f assets/*.gold.conll
cat assets/conll-2012/v4/data/development/data/english/annotations/*/*/*/*.v4_gold_conll >> assets/dev.gold.conll
cat assets/conll-2012/v4/data/train/data/english/annotations/*/*/*/*.v4_gold_conll >> assets/train.gold.conll
cat assets/conll-2012/v4/data/test/data/english/annotations/*/*/*/*.v4_gold_conll >> assets/test.gold.conll
