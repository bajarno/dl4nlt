#!/usr/bin/env bash

cd ~/Studie/DeepLearningNLP/miniproject/code/code/

# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 128 --encoder_type Attn --num_epochs 3 --order 3 --learning_rate 0.01
# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 128 --encoder_type Attn --num_epochs 3 --order 3 --learning_rate 0.001
# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 128 --encoder_type Attn --num_epochs 3 --order 3 --learning_rate 0.0001
# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 128 --encoder_type Attn --num_epochs 3 --order 5 --learning_rate 0.01
# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 128 --encoder_type Attn --num_epochs 3 --order 5 --learning_rate 0.001
# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 128 --encoder_type Attn --num_epochs 3 --order 5 --learning_rate 0.0001

# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 256 --encoder_type Attn --num_epochs 3 --order 3 --learning_rate 0.01
# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 256 --encoder_type Attn --num_epochs 3 --order 3 --learning_rate 0.001
# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 256 --encoder_type Attn --num_epochs 3 --order 3 --learning_rate 0.0001
# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 256 --encoder_type Attn --num_epochs 3 --order 5 --learning_rate 0.01
# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 256 --encoder_type Attn --num_epochs 3 --order 5 --learning_rate 0.001
# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 256 --encoder_type Attn --num_epochs 3 --order 5 --learning_rate 0.0001


# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 256 --encoder_type Attn --num_epochs 20 --order 5 --learning_rate 0.001
# python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 256 --encoder_type BOW --num_epochs 20 --order 5 --learning_rate 0.001
python train.py --dataset ../data/kaggle_preprocessed_sub_4000.csv --hidden_size 64 --encoder_type Conv --num_epochs 20 --order 5 --learning_rate 0.001
