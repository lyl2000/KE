import os
import time
import random
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import load
import tools

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.cuda.set_device(1)

"""NEXT:
save model
"""

nh1 = 300
nh2 = 300
win = 3
emb_dimension = 300
lr = 0.1
lr_decay = 0.1
max_grad_norm = 5
seed = 2021
checkpoint_dir = './checkpoints'
nepochs = 15
batch_size = 16
display_test_per = 3
lr_decay_per = 10

torch.manual_seed(seed)


class Model(nn.Module):
    def __init__(self, vocab_size, ny, nz, win_size=win, embedding_size=emb_dimension, hidden_size1=nh1, hidden_size2=nh2, batch_size=batch_size, model_cell='rnn'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(0.1)
        self.bidirectional = True
        self.dim1 = (2 if self.bidirectional else 1) * hidden_size1
        self.dim2 = (2 if self.bidirectional else 1) * hidden_size2

        if model_cell == 'rnn':
            self.single_cell1 = nn.RNN(input_size=embedding_size * win_size, hidden_size=hidden_size1, bidirectional=self.bidirectional)
            self.single_cell2 = nn.RNN(input_size=self.dim1, hidden_size=hidden_size2, bidirectional=self.bidirectional)
        elif model_cell == 'gru':
            self.single_cell1 = nn.GRU(input_size=embedding_size * win_size, hidden_size=hidden_size1, bidirectional=self.bidirectional)
            self.single_cell2 = nn.GRU(input_size=self.dim1, hidden_size=hidden_size2, bidirectional=self.bidirectional)
        elif model_cell == 'lstm':
            self.single_cell1 = nn.LSTM(input_size=embedding_size * win_size, hidden_size=hidden_size1, bidirectional=self.bidirectional)
            self.single_cell2 = nn.LSTM(input_size=self.dim1, hidden_size=hidden_size2, bidirectional=self.bidirectional)
        else:
            raise 'model_cell error!'

        self.fc1 = nn.Linear(self.dim1, ny)
        self.fc2 = nn.Linear(self.dim2, nz)

    def forward(self, x, seq_lengths):
        batch, seq_len, win = x.shape
        # idx -> embedding
        x = self.embedding(x)
        x = self.dropout(x)

        # embedding in win -> embedding
        x = x.reshape(batch, seq_len, -1)
        x = x.permute(1, 0, 2)
        x = pack_padded_sequence(x, seq_lengths)

        out, h = self.single_cell1(x)
        y, _ = pad_packed_sequence(out)
        y = y.permute(1, 0, 2).reshape(-1, self.dim1)
        y = self.dropout(y)
        y = self.fc1(y)

        out, h = self.single_cell2(out)
        z, _ = pad_packed_sequence(out)
        z = z.permute(1, 0, 2).reshape(-1, self.dim2)
        z = self.dropout(z)
        z = self.fc2(z)
        
        return y, z

def batchPadding(batch, padding_word=0, forced_sequence_length=None):
    batch = sorted(batch, key=lambda x: len(x['lex']), reverse=True)

    if forced_sequence_length is not None:
        seq_lengths = [forced_sequence_length] * len(batch)
        sequence_length = forced_sequence_length
    else:
        seq_lengths = [len(x['lex']) for x in batch]
        sequence_length = max(seq_lengths)
        
    padded_lex = [F.pad(data['lex'], (0, sequence_length - len(data['lex'])), value=padding_word).tolist() for data in batch]
    padded_y = [F.pad(data['y'], (0, sequence_length - len(data['y'])), value=padding_word).tolist() for data in batch]
    padded_z = [F.pad(data['z'], (0, sequence_length - len(data['z'])), value=padding_word).tolist() for data in batch]
    mask = [[1] * min(sequence_length, len(data['lex'])) + [0] * (max(0, sequence_length - len(data['lex']))) for data in batch]

    padded_lex = tools.contextwin_2(padded_lex, win)
    padded_lex = torch.tensor(padded_lex, dtype=torch.int64)
    padded_y = torch.tensor(padded_y, dtype=torch.int64)
    padded_z = torch.tensor(padded_z, dtype=torch.int64)
    mask = torch.tensor(mask, dtype=torch.bool)
    # print('{}\n{}\n{}\n{}\n{}\n{}'.format('-' * 50, padded_lex, padded_y, padded_z, mask, '-' * 50))
    return padded_lex, seq_lengths, padded_y, padded_z, mask

def iterData(data, batchsize):

    bucket = random.sample(data, len(data))
    bucket = [bucket[i: i+batchsize] for i in range(0, len(bucket), batchsize)]
    random.shuffle(bucket)
    for batch in bucket:
        yield batchPadding(batch)

def trainModel(model, cnt, optimizer, train_set):
    weight_y = torch.tensor([1 / (cnt[0][idx] + 1) for idx in range(2)]).cuda()
    weight_z = torch.tensor([1 / (cnt[1][idx] + 1) for idx in range(5)]).cuda()
    print('cnt: {} {}\nweight: {} {}'.format(dict(cnt[0]), dict(cnt[1]), weight_y.cpu().numpy(), weight_z.cpu().numpy()))
    criterion = F.cross_entropy
    alpha = 0.5
    for epoch in range(nepochs):
        trainloader = iterData(train_set, batch_size)
        model.train()
        train_loss = []
        data_size = 0
        t_start = time.time()
        for i, (lex, seq_lengths, y, z, mask) in enumerate(trainloader):
            lex = lex.cuda()
            y = y.cuda()
            z = z.cuda()
            mask = mask.cuda()

            y_pred, z_pred = model(lex, seq_lengths)
            y = y.reshape(-1)
            z = z.reshape(-1)
            
            # print('{} {} {} {}\n{}\n{}\n'.format(y_pred.shape, y.shape, z_pred.shape, z.shape, z_pred, z))
            loss = (alpha * criterion(y_pred, y, weight_y).masked_select(mask).mean() + (1 - alpha) * criterion(z_pred, z, weight_z).masked_select(mask).mean())# / lex.size(0)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2)
            optimizer.step()
            # Sets the learning rate to the initial LR decayed by "lr_decay" every "lr_decay_per" epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] *= (lr_decay ** (epoch // lr_decay_per))

            train_loss.append([float(loss), mask.cpu().sum()])
            data_size += mask.cpu().sum()

        train_loss = np.array(train_loss)
        train_loss = np.sum(train_loss[:, 0] * train_loss[:, 1]) / np.sum(train_loss[:, 1])
        print('training loss: {:.8f} epoch {:2d} completed in {:.2f} (sec)'.format(train_loss, epoch + 1, time.time() - t_start))
        
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model.pt')

        torch.save(model, checkpoint_prefix)

    return model

def evalModel(model, valid_set):
    model.eval()
    data_size = 0
    accuracy, precision, recall, f1 = 0, 0, 0, 0
    labels = [1, 2, 3, 4]
    validloader = iterData(valid_set, batch_size)
    for i, (lex, seq_lengths, y, z, mask) in enumerate(validloader):
        lex = lex.cuda()
        y_pred, z_pred = model(lex, seq_lengths)
        y_pred = torch.argmax(y_pred, dim=-1).reshape(lex.size(0), -1)
        z_pred = torch.argmax(z_pred, dim=-1).reshape(lex.size(0), -1)
        y_pred = y_pred.cpu()
        z_pred = z_pred.cpu()
        # print('y pred: {} z pred: {}\ny true: {} z true: {}'.format(y_pred, z_pred, y, z))
        accuracy += sum([accuracy_score(z1[:seq_length], z2[:seq_length]) for z1, z2, seq_length in zip(z, z_pred, seq_lengths)])
        precision += sum([precision_score(z1[:seq_length], z2[:seq_length], labels=labels, average='micro', zero_division=0) for z1, z2, seq_length in zip(z, z_pred, seq_lengths)])
        recall += sum([recall_score(z1[:seq_length], z2[:seq_length], labels=labels, average='micro', zero_division=0) for z1, z2, seq_length in zip(z, z_pred, seq_lengths)])
        f1 += sum([f1_score(z1[:seq_length], z2[:seq_length], labels=labels, average='micro', zero_division=0) for z1, z2, seq_length in zip(z, z_pred, seq_lengths)])
        data_size += lex.size(0)
        
    print('accuracy: {:.3f} precision: {:.3f} recall: {:.3f} f1: {:.3f}'.format(accuracy / data_size, precision / data_size, recall / data_size, f1 / data_size))


if __name__ == '__main__':

    train_set, test_set, dic, embedding = load.atisfold()
    train_lex, train_y, train_z = train_set
    tr = int(len(train_lex) * 0.9)

    valid_lex, valid_y, valid_z = train_lex[tr:], train_y[tr:], train_z[tr:]
    train_lex, train_y, train_z = train_lex[:tr], train_y[:tr], train_z[:tr]
    test_lex,  test_y, test_z = test_set

    cnt_y = Counter()
    cnt_z = Counter()

    train_data = []
    for lex, y, z in zip(train_lex, train_y, train_z):
        cnt_y += Counter(y)
        cnt_z += Counter(z)
        train_data.append({'lex': torch.tensor(lex), 'y': torch.tensor(y), 'z': torch.tensor(z)})

    valid_data = []
    for lex, y, z in zip(valid_lex, valid_y, valid_z):
        cnt_y += Counter(y)
        cnt_z += Counter(z)
        valid_data.append({'lex': torch.tensor(lex), 'y': torch.tensor(y), 'z': torch.tensor(z)})

    test_data = []
    for lex, y, z in zip(test_lex, test_y, test_z):
        cnt_y += Counter(y)
        cnt_z += Counter(z)
        test_data.append({'lex': torch.tensor(lex), 'y': torch.tensor(y), 'z': torch.tensor(z)})

    mode = 'testing'
    if mode == 'training':
        vocab = set(dic['words2idx'].keys())
        vocab_size = len(vocab) + 1
        ny, nz = 2, 5
        print('train {} valid {} test {} vocab {}'.format(len(train_data), len(valid_data), len(test_data), vocab_size))
        print('Train started!')
        model = Model(vocab_size=vocab_size, ny=ny, nz=nz, model_cell='lstm').cuda()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        model = trainModel(model, (cnt_y, cnt_z), optimizer, train_data)
    else:
        model = torch.load(os.path.join(checkpoint_dir, 'model.pt'))
    
    print('-' * 60)
    # evalModel(model, train_data)
    # print('-' * 60)
    evalModel(model, valid_data)
    print('-' * 60)
    # evalModel(model, test_data)
    # print('-' * 60)
