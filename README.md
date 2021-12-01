# Keyphrase Extraction

2021 Social Network and Public Opinion Processing

Paper recurrence: EMNLP2016 paper [Keyphrase Extraction Using Deep Recurrent Neural Networks on Twitter](http://jkx.fudan.edu.cn/~qzhang/paper/keyphrase.emnlp2016.pdf)

Rewritten on the basis of [Source codes](https://github.com/fudannlp16/KeyPhrase-Extraction) by *PyTorch implementation* & *Modified tensorflow implementation*

## Preparation
You may need to prepare the pre-trained word vectors (or not).
* Pre-trained word vectors. Download [GoogleNews-vectors-negative300.bin.gz](https://code.google.com/archive/p/word2vec/)

## Details
Joint RNN model

* data: Dataset

* checkpoints: Store the parameters obtained from model training

* main.py: Main program (tensorflow implementation)

* main_v2.py: Main program (PyTorch with 2 separate RNN layers)

* main_v2_cell.py: Main program (PyTorch with 2-layer RNNCell）

* model.py: Defined the joint-rnn model (tensorflow)

* model_v2.py: Defined the joint-rnn model (PyTorch)

* bi_lstm_model.py: Use bidirectional lstm instead of rnn (part of the code has not been modified, if you want to run, please modify it according to the format of model.py first)

* load.py: Load the dataset

* tools.py: Some utility functions

## Requirement
- tensorflow 2.2.0 + tensorlayer
- PyTorch

## Run
```bash
python data/data_process.py
python main.py  // original tensorflow
python main_v2.py  // PyTorch
```
