#!/bin/bash

gpu_id="${6:--1}"

# Runs training. Expects as arguments:
#   (1) dataset ID,
#   (2) run name,
#   (3) language,
#   (4) config file name,
#   (5) max. steps.
#   (6) GPU information if GPU is to be used.
PYTHONPATH='src' python -m spacy train configs/$4 \
          --paths.dataset_name $1 \
          --output training/$1/$2 \
          --paths.train corpora/$1/train.spacy \
          --paths.dev corpora/$1/dev.spacy \
          --paths.kb wikid/output/$3/kb \
          --paths.db wikid/output/$3/wiki.sqlite3 \
          --paths.base_nlp training/base-nlp/$3 \
          --paths.mentions_candidates corpora/$1/mentions_candidates.pkl \
          --paths.language $3 \
          --training.max_steps $5 \
          -c src/custom_functions.py \
          --gpu-id $gpu_id