#!/usr/bin/env bash

DATA_LOCATION="datasets/twenty_newsgroups.txt"
OUTPUT_LOCATION="datasets/sentencepiece/"
VOCAB_SIZE=100000
FILENAME=$(basename $DATA_LOCATION)

SP_MODEL_PREFIX=sp_${FILENAME%.*}

spm_train \
    --input=${DATA_LOCATION} \
    --model_prefix=${SP_MODEL_PREFIX} \
    --vocab_size=${VOCAB_SIZE} \
    --hard_vocab_limit=false \
    --model_type=unigram

mkdir -p $OUTPUT_LOCATION
mv ${SP_MODEL_PREFIX}.* $OUTPUT_LOCATION

echo "This is a test sentence." \
    | spm_encode --model=${OUTPUT_LOCATION}${SP_MODEL_PREFIX}.model
