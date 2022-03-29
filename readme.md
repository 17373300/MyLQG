python CollectVocab.py data/train/train.txt.source.txt data/train/train.txt.target.txt   data/train/vocab.txt

python CollectVocab.py  data/train/train.txt.bio data/train/bio.vocab.txt

python CollectVocab.py  data/train/train.txt.pos data/train/feat.vocab.txt

head -n 50000 data/train/vocab.txt > data/train/vocab.txt.20k


#!/bin/bash

set -x

DATAHOME=/home/lxy/NQG/mydata
EXEHOME=/home/lxy/NQG/code/NQG/seq2seq_pt

SAVEPATH=${DATAHOME}/models/NQG_plus

mkdir -p ${SAVEPATH}

cd ${EXEHOME}

python train.py \
       -save_path ${SAVEPATH} -log_home ${SAVEPATH} \
       -online_process_data \
       -train_src ${DATAHOME}/train/train.txt.source.txt -src_vocab ${DATAHOME}/train/vocab.txt.20k \
       -train_bio ${DATAHOME}/train/train.txt.bio -bio_vocab ${DATAHOME}/train/bio.vocab.txt \
       -train_feats ${DATAHOME}/train/train.txt.pos \
       -feat_vocab ${DATAHOME}/train/feat.vocab.txt \
       -train_tgt ${DATAHOME}/train/train.txt.target.txt -tgt_vocab ${DATAHOME}/train/vocab.txt.20k \
       -layers 1 \
       -enc_rnn_size 512 -brnn \
       -word_vec_size 300 \
       -dropout 0.5 \
       -batch_size 64 \
       -beam_size 5 \
       -epochs 20 -optim adam -learning_rate 0.001 \
       -gpus 0 \
       -curriculum 0 -extra_shuffle \
       -start_eval_batch 1000 -eval_per_batch 500 -halve_lr_bad_count 3 \
       -seed 12345 -cuda_seed 12345 \
       -log_interval 100 \
       -dev_input_src ${DATAHOME}/dev/dev.txt.shuffle.dev.source.txt \
       -dev_bio ${DATAHOME}/dev/dev.txt.shuffle.dev.bio \
       -dev_feats ${DATAHOME}/dev/dev.txt.shuffle.dev.pos \
       -dev_ref ${DATAHOME}/dev/dev.txt.shuffle.dev.target.txt \
       -max_sent_length 500
