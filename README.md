# Keyphrase Extraction

2021 Social Network and Public Opinion Processing Course Paper recurrence 《2016Keyphrase Extraction Using Deep Recurrent Neural Networks on Twitter》

Source codes of our EMNLP2016 paper [Keyphrase Extraction Using Deep Recurrent Neural Networks on Twitter](http://jkx.fudan.edu.cn/~qzhang/paper/keyphrase.emnlp2016.pdf)

## Preparation
You may need to prepare the pre-trained word vectors.
* Pre-trained word vectors. Download [GoogleNews-vectors-negative300.bin.gz](https://code.google.com/archive/p/word2vec/)

## Details
Joint RNN model

* data文件夹存储数据集

* checkpoints文件夹存储模型训练得到的参数

* main.py是主程序（tensorflow）

* main_v2.py是主程序（PyTorch + RNN * 2）

* main_v2_cell.py是主程序（PyTorch + RNNCell）

* model.py定义了joint-rnn模型（tensorflow）

* bi_lstm_model.py 用双向lstm代替rnn（部分代码没有修改，如果想要运行，请先按照model.py的格式修改）

* load.py用于加载数据集

* tools.py定义了一些工具函数

## Requirement
tensorflow 2.2.0 + tensorlayer

