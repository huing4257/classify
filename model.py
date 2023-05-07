import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

MAX_L = 500

class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()

    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])
        pass


class TextCNN(nn.Module):
    def __init__(self, embedding, kernel_sizes, num_channels, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.pool = MaxPool()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv2d(1, c, (k, self.embedding.embedding_dim)))
        self.fc = nn.Linear(sum(num_channels), 2)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        embedded = embedded.unsqueeze(1)

        convd = [conv(embedded).squeeze(-1) for conv in self.convs]
        pooled = [self.pool(c).squeeze(-1) for c in convd]
        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)
    
class MLP(nn.Module):
    def __init__(self, embedding, hidden_sizes, num_layers, dropout):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        self.mlp = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.mlp.add_module('linear%d' % i, nn.Linear(self.embedding.embedding_dim*MAX_L, hidden_sizes[i]))
            else:
                self.mlp.add_module('linear%d' % i, nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.mlp.add_module('relu%d' % i, nn.ReLU())
            self.mlp.add_module('dropout%d' % i, nn.Dropout(dropout))
        self.mlp.add_module('output', nn.Linear(hidden_sizes[-1], 2))

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        embedded = embedded.view(embedded.shape[0], -1)
        return self.mlp(embedded)


class GRU(nn.Module):
    def __init__(self, embedding, hidden_dim, num_layers, dropout):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        self.gru = nn.GRU(input_size=self.embedding.embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=True,
                          bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        output, h_n = self.gru(embedded)
        return self.fc(torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1))
    
class RNN(nn.Module):
    def __init__(self, embedding, hidden_dim, num_layers, dropout):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        self.rnn = nn.RNN(input_size=self.embedding.embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=True,
                          bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        output, h_n = self.rnn(embedded)
        return self.fc(torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1))

class LSTM(nn.Module):
    def __init__(self, embedding, hidden_dim, num_layers, dropout):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        output, (h_n, c_n) = self.lstm(embedded)
        return self.fc(torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1))