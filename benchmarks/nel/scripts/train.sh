#!/bin/bash

gpu_id="${5:--1}"

# Runs training. Expects as arguments:
#   (1) dataset ID,
#   (2) run name,
#   (3) config file name,
#   (4) max. steps.
#   (5) GPU information if GPU is to be used.
PYTHONPATH=scripts python -m spacy train configs/$3 \
          --paths.dataset_name $1 \
          --output training/$1/$2 \
          --paths.train corpora/$1/train.spacy \
          --paths.dev corpora/$1/dev.spacy \
          --paths.kb temp/$1/kb \
          --paths.base_nlp temp/$1/nlp \
          --training.max_steps $4 \
          -c scripts/custom_functions.py \
          --gpu-id $gpu_id