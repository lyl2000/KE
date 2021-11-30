# -*- coding: utf-8 -*-
import numpy as np
import re
import pickle
from collections import Counter

def getlist(filename):
    # split dataset into sentence and tag
    with open(filename, 'r', encoding='utf-8') as f:
        datalist, taglist = [], []
        for line in f:
            line = line.strip()
            datalist.append(line.split('\t')[0])
            taglist.append(line.split('\t')[1])
    return datalist[:1000], taglist[:1000]


def get_dict(filenames):
    # build vocabulary using train and TEST dataset
    trnTweet, testTweet = filenames
    trn_data = getlist(trnTweet)
    test_data = getlist(testTweet)
    sentence_list = trn_data[0] + test_data[0]

    words = []
    for sentence in sentence_list:
        word_list = sentence.split()
        words.extend(word_list)

    word_counts = Counter(words)
    # 空出0作为padding的idx
    words2idx = {word[0]: i+1 for i, word in enumerate(word_counts.most_common())}
    labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
    dicts = {'words2idx': words2idx, 'labels2idx': labels2idx}

    return dicts, trn_data, test_data


def get_train_test_dicts(filenames):    
    """[summary]

    Args:
        filenames: [trnTweet, testTweet, tag_id_cnt]

    Returns:
        dataset: [train_set, test_set, dicts]
            train_set = [train_lex, train_y, train_z]
            test_set = [test_lex, test_y, test_z]
            dicts = {'words2idx': words2idx, 'labels2idx': labels2idx}
    """
    trnTweetCnn, testTweetCnn = filenames
    dicts, trn_data, test_data = get_dict([trnTweetCnn, testTweetCnn])

    trn_sentence_list, trn_tag_list = trn_data
    test_sentence_list, test_tag_list = test_data

    words2idx, labels2idx = dicts['words2idx'], dicts['labels2idx']

    def get_lex_y(sentence_list, tag_list):
        lex, y, z = [], [], []
        bad_cnt = 0
        for s, tag in zip(sentence_list, tag_list):
            word_list = s.split()
            t_list = tag.split()
            # word -> idx for word in word_list
            emb = [words2idx[x] for x in word_list]
            # emb = map(lambda x: words2idx[x], word_list)

            begin = -1
            for i in range(len(word_list)):
                ok = True
                for j in range(len(t_list)):
                    if word_list[i+j] != t_list[j]:
                        ok = False
                        break
                if ok:
                    begin = i
                    break
            if begin == -1:
                bad_cnt += 1
                continue

            lex.append(emb)

            labels_y = [0]*len(word_list)
            for i in range(len(t_list)):
                labels_y[begin+i] = 1
            y.append(labels_y)

            labels_z = [0]*len(word_list)
            if len(t_list) == 1:
                labels_z[begin] = labels2idx['S']
            elif len(t_list) > 1:
                labels_z[begin] = labels2idx['B']
                for i in range(len(t_list)-2):
                    labels_z[begin+i+1] = labels2idx['I']
                labels_z[begin+len(t_list)-1] = labels2idx['E']
            z.append(labels_z)
        return lex, y, z

    train_lex, train_y, train_z = get_lex_y(trn_sentence_list, trn_tag_list)
    test_lex, test_y, test_z = get_lex_y(test_sentence_list, test_tag_list)
    train_set = [train_lex, train_y, train_z]
    test_set = [test_lex, test_y, test_z]
    data_set = [train_set, test_set, dicts]
    with open(r'data/data_set.pkl', 'wb') as f:
        pickle.dump(data_set, f)
    return data_set


def load_bin_vec(frame, vocab):
    k = 0
    word_vecs = {}
    with open(frame, 'r') as f:
        for line in f:
            word = line.strip().split(' ', 1)[0]
            embeding = line.strip().split(' ', 1)[1].split()
            if word in vocab:
                word_vecs[word] = np.asarray(embeding, dtype=np.float32)
            k += 1
            if k % 10000 == 0:
                print("load_bin_vec %d" % k)

    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, dim=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    k = 0
    for w in vocab:
        if w not in word_vecs:
            word_vecs[w] = np.asarray(np.random.uniform(-0.25, 0.25, dim), dtype=np.float32)
            k += 1
            if k % 10000 == 0:
                print("add_unknow_words %d" % k)
    return word_vecs


def get_embedding(w2v, words2idx, k=300):
    embedding = np.zeros((len(w2v) + 2, k), dtype=np.float32)
    for w, idx in words2idx.items():
        embedding[idx] = w2v[w]
    with open(r'data/embedding.pkl', 'wb') as f:
        pickle.dump(embedding, f)
    return embedding


if __name__ == '__main__':
    
    data_folder = [r'data/original_data/trnTweet', r'data/original_data/testTweet']
    data_set = get_train_test_dicts(data_folder)
    print("data_set complete!")
    train_set, test_set, dicts = data_set
    vocab = set(dicts['words2idx'].keys())
    print("total num words: {} total num train set: {} total num test set: {}\ndataset created!".format(len(vocab), len(train_set[0]), len(test_set[0])))

    # GoogleNews-vectors-negative300.txt为预先训练的词向量
    w2v_file = 'original_data/GoogleNews-vectors-negative300.txt'
    # w2v = load_bin_vec(w2v_file, vocab)
    w2v = {}
    print("word2vec loaded")
    add_unknown_words(w2v, vocab)
    embedding = get_embedding(w2v, dicts['words2idx'])
    print("embedding created")
