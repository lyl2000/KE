import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import os
import time
import random
import numpy as np

import tools
import load

torch.cuda.set_device(1)

nh1 = 300
nh2 = 300
win = 3
emb_dimension = 300
lr = 0.1
lr_decay = 0.5
max_grad_norm = 5
seed = 2021
checkpoint_dir = './checkpoints'
nepochs = 15
batch_size = 1
display_test_per = 3
lr_decay_per = 10

torch.manual_seed(seed)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every x epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (lr_decay ** (epoch // lr_decay_per))

class Model(nn.Module):
    def __init__(self, vocab_size, ny, nz, win_size=win, embedding_size=emb_dimension, hidden_size1=nh1, hidden_size2=nh2, batch_size=batch_size, model_cell='rnn'):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.win_size = win_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.model_cell = model_cell
        if model_cell == 'rnn':
            self.single_cell1 = nn.RNNCell(input_size=embedding_size * win_size, hidden_size=hidden_size1)
            self.single_cell2 = nn.RNNCell(input_size=hidden_size1, hidden_size=hidden_size2)
        elif model_cell == 'gru':
            self.single_cell1 = nn.GRUCell(input_size=embedding_size * win_size, hidden_size=hidden_size1)
            self.single_cell2 = nn.GRUCell(input_size=hidden_size1, hidden_size=hidden_size2)
        else:
            raise 'model_cell error!'
        self.fc1 = nn.Linear(hidden_size1, ny)
        self.fc2 = nn.Linear(hidden_size2, nz)

    def forward(self, x):
        batch_size, seq_size, win = x.shape

        x = x.permute((1, 0, 2))
        # idx -> embedding
        x = self.embedding(x)
        # embedding in win -> embedding
        x = x.reshape(seq_size, batch_size, -1)
        x = F.relu(x)

        # init h
        h1 = Variable(torch.zeros(batch_size, self.hidden_size1)).cuda()
        h2 = Variable(torch.zeros(batch_size, self.hidden_size2)).cuda()

        # out1 -> y, out2 -> z
        out1, out2 = [], []
        for i in range(seq_size):
            # 按顺序放进去，因此seq_size是第一维
            h1 = self.single_cell1(x[i], h1)
            h2 = self.single_cell2(h1, h2)
            # seq_size, batch_size, hidden_size
            out1.append(h1)
            out2.append(h2)

        out1 = torch.cat(out1, dim=0).reshape(seq_size, batch_size, self.hidden_size1)
        out2 = torch.cat(out2, dim=0).reshape(seq_size, batch_size, self.hidden_size2)

        out1 = out1.permute((1, 0, 2)).reshape(batch_size * seq_size, -1)
        out2 = out2.permute((1, 0, 2)).reshape(batch_size * seq_size, -1)

        out1 = self.fc1(out1)
        out2 = self.fc2(out2)
        # seq_size, batch_size
        return out1, out2

def data_pad(data_set, padding_word=0, forced_sequence_length=None):
    if forced_sequence_length is None:
        sequence_length = max(len(x['lex']) for x in data_set)
    else:
        sequence_length = forced_sequence_length
    padded_lex, padded_y, padded_z = [], [], []
    for data in data_set:
        num = sequence_length - len(data['lex'])
        if num <= 0:
            lex = data['lex'][:sequence_length]
            y = data['y'][:sequence_length]
            z = data['z'][:sequence_length]
        else:
            lex = data['lex'] + [padding_word] * num
            y = data['y'] + [padding_word] * num
            z = data['z'] + [padding_word] * num
        padded_lex.append(lex)
        padded_y.append(y)
        padded_z.append(z)
    padded_lex = tools.contextwin_2(padded_lex, win)
    padded_lex = torch.tensor(padded_lex, dtype=torch.int64)
    padded_y = torch.tensor(padded_y, dtype=torch.int64)
    padded_z = torch.tensor(padded_z, dtype=torch.int64)
    return padded_lex, padded_y, padded_z

def iterData(data, batchsize):
    bucket = random.sample(data, len(data))
    bucket = [bucket[i: i+batchsize] for i in range(0, len(bucket), batchsize)]
    random.shuffle(bucket)
    for batch in bucket:
        yield data_pad(batch)

def train_model(model, criterion, optimizer, train_set):
    for epoch in range(nepochs):
        trainloader = iterData(train_set, batch_size)
        model.train()
        train_loss = []
        data_size = 0
        t_start = time.time()
        for i, (lex, y, z) in enumerate(trainloader):
            lex = lex.cuda()
            y = y.cuda()
            z = z.cuda()
            y = y.reshape(-1)
            z = z.reshape(-1)
            y_pred, z_pred = model(lex)
            loss = (0.5 * criterion(y_pred, y) + 0.5 * criterion(z_pred, z)) / lex.size(0)
            optimizer.zero_grad()
            loss.backward()
            train_loss.append([float(loss), lex.size(0)])
            
            # print(data_size, 'loss:', float(loss))
            # print('loss %.8f [learning] epoch %i  >> %2.2f%% completed in %.2f (sec) <<' % (float(loss), epoch, batch_size * i * 100 / len(train_set), time.time() - t_start))
                    
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2)
            optimizer.step()
            adjust_learning_rate(optimizer, epoch)
            data_size += lex.size(0)
        train_loss = np.array(train_loss)
        train_loss = np.sum(train_loss[:, 0] * train_loss[:, 1]) / np.sum(train_loss[:, 1])
        print('train loss: {:.8f} {}'.format(train_loss, time.time() - t_start))
        
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        
    return model

def eval_model(model, valid_set):
    model.eval()
    data_size = 0
    acc = 0
    validloader = iterData(valid_set, batch_size)
    for i, (lex, y, z) in enumerate(validloader):
        lex = lex.cuda()
        y = y.cuda()
        z = z.cuda()
        y_pred, z_pred = model(lex)
        assert len(lex) == batch_size
        y_pred = torch.argmax(y_pred, dim=-1).reshape(len(lex), -1)
        z_pred = torch.argmax(z_pred, dim=-1).reshape(len(lex), -1)
        print('{}\n{}\n{} {} {}'.format(z_pred, z, (z_pred == z).sum(), z.shape[1], (z_pred == z).sum() / z.shape[1]))
        acc += (z_pred == z).sum() / z.shape[1]
        data_size += lex.size(0)
        # res, good_cnt, all_cnt, p_cnt, r_cnt, pr_cnt = conlleval(z_pred.numpy(), z.numpy(), '')
        # print(res, good_cnt, all_cnt, p_cnt, r_cnt, pr_cnt)
    assert data_size == len(valid_set)
    print('acc: {:.8f}'.format(acc / data_size))

def getKeyphraseList(l):
    res, now = [], []
    for i in range(len(l)):
        if l[i] != 0:
            now.append(str(i))
        if l[i] == 0 or i == len(l) - 1:
            if len(now) != 0:
                res.append(' '.join(now))
            now = []
    return set(res)

def conlleval(predictions, groundtruth, file):
    assert len(predictions) == len(groundtruth)
    res = {}
    all_cnt, good_cnt = len(predictions), 0
    p_cnt, r_cnt, pr_cnt = 0, 0, 0
    for i in range(all_cnt):
        # print i
        if all(predictions[i][:len(groundtruth[i])] == groundtruth[i]):
            good_cnt += 1
        pKeyphraseList = getKeyphraseList(predictions[i][:len(groundtruth[i])])
        gKeyphraseList = getKeyphraseList(groundtruth[i])
        p_cnt += len(pKeyphraseList)
        r_cnt += len(gKeyphraseList)
        pr_cnt += len(pKeyphraseList & gKeyphraseList)
    res['a'] = good_cnt / all_cnt if all_cnt > 0 else 0
    res['p'] = good_cnt / p_cnt if p_cnt > 0 else 0
    res['r'] = good_cnt / r_cnt if r_cnt > 0 else 0
    res['f'] = 2 * res['p'] * res['r'] / (res['p'] + res['r']) if res['p'] + res['r'] > 0 else 0
    return res, good_cnt, all_cnt, p_cnt, r_cnt, pr_cnt


if __name__ == '__main__':
    train_set, test_set, dic, embedding = load.atisfold()
        
    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    train_lex, train_y, train_z = train_set

    tr = int(len(train_lex)*0.9)
    valid_lex, valid_y, valid_z = train_lex[tr:], train_y[tr:], train_z[tr:]
    train_lex, train_y, train_z = train_lex[:tr], train_y[:tr], train_z[:tr]
    test_lex,  test_y, test_z = test_set

    train_data = []
    for lex, y, z in zip(train_lex, train_y, train_z):
        train_data.append({'lex': lex, 'y': y, 'z': z})

    valid_data = []
    for lex, y, z in zip(valid_lex, valid_y, valid_z):
        valid_data.append({'lex': lex, 'y': y, 'z': z})

    test_data = []
    for lex, y, z in zip(test_lex, test_y, test_z):
        test_data.append({'lex': lex, 'y': y, 'z': z})

    vocab = set(dic['words2idx'].keys())
    vocab_size = len(vocab)
    y_nclasses, z_nclasses = 2, 5

    print('train {} valid {} test {} vocab {}'.format(len(train_data), len(valid_data), len(test_data), vocab_size))
    print('Train started!')
    
    model = Model(vocab_size=vocab_size, ny=y_nclasses, nz=z_nclasses, model_cell='gru').cuda()
    criterion = F.cross_entropy
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    model = train_model(model, criterion, optimizer, train_data)
    eval_model(model, random.choices(train_data, k=10))
    eval_model(model, random.choices(valid_data, k=10))
    eval_model(model, random.choices(test_data, k=10))
