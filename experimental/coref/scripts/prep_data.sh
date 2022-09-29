#!/bin/bash
set -eou pipefail
ontonotes="$1"

tar xzf assets/conll-2012-development.v4.tar.gz -C assets/
tar xzf assets/conll-2012-test-key.tar.gz -C assets/
tar xzf assets/conll-2012-test-official.v9.tar.gz -C assets/
tar xzf assets/conll-2012-train.v4.tar.gz -C assets/

# We only need the English data
sed -i 's/arabic.english.chinese/english/' assets/conll-2012/v3/scripts/skeleton2conll.sh

echo "Rehydrating data... This will take a few minutes"
bash assets/conll-2012/v3/scripts/skeleton2conll.sh -D "$ontonotes" assets/conll-2012

# Concatenate the data into single files to make it easier to work with
bash scripts/concat_data.sh
