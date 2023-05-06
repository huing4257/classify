import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


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
