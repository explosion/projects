#!/bin/bash

set -euo pipefail

# Downloads and sets up Mewsli-9 dataset.
cd assets
SUBDIR=dense_representations_for_entity_retrieval
svn export https://github.com/google-research/google-research/trunk/$SUBDIR
# Fix outdated requirement.
sed -i -e 's/absl_py>=0.8.1/absl-py/g' $SUBDIR/requirements.txt
# Only download and process English data.
LANGS="ar de en es fa ja sr ta tr"
LANG_LIST=($(echo $LANGS | tr ' ' "\n"))
sed -i -e "s/LANG_LIST=(${LANGS})/LANG_LIST=(en)/g" $SUBDIR/mel/mewsli-9/run_parse_wikinews_i18n.sh
sed -i -e "s/LANG_LIST=(${LANGS})/LANG_LIST=(en)/g" $SUBDIR/mel/mewsli-9/get_wikinews_dumps.sh
sed -i -e "s/LANG_LIST=(${LANGS})/LANG_LIST=(en)/g" $SUBDIR/mel/mewsli-9/run_wikiextractor.sh
# Only keep 3rd line (english corpus) in checksum file, otherwise errors are raised due to the absence of the other
# languages corpora.
echo $(sed '3q;d' $SUBDIR/mel/mewsli-9/dump_checksums.txt) > $SUBDIR/mel/mewsli-9/dump_checksums.txt
cd $SUBDIR/mel
chmod +x get-mewsli-9.sh
PYTHONPATH=. bash get-mewsli-9.sh

# Move output & log data to folder, clean up everything else.
mkdir -p ../../mewsli-9
cp -r mewsli-9/output/* ../../mewsli-9
cd ../..
rm -rf $SUBDIR
for lang in "${LANG_LIST[@]}"
do
  rm -rf mewsli-9/dataset/${lang}
done
