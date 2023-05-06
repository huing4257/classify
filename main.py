from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
from model import TextCNN
import torch
from load import DataUtil
import numpy as np
import torch.nn.functional as F

if __name__ == '__main__':
    word2vec_path = 'Dataset/wiki_word2vec_50.bin'
    train_path = 'Dataset/train.txt'
    test_path = 'Dataset/test.txt'
    batch_size = 64
    data_util = DataUtil(word2vec_path, train_path, test_path, batch_size)
    embed = data_util.embed
    train_iter = data_util.train_iter
    test_iter = data_util.test_iter

    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    net = TextCNN(embed, kernel_sizes, nums_channels, 0.5)
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        epoch_loss = 0
        net.train()
        batch_idx = 0
        for batch_idx, (data, target) in enumerate(train_iter):
            data = torch.as_tensor(data, dtype=torch.long)
            target = target.long()
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = net(data)
            # labels = output.argmax(dim= 1)
            # acc = accuracy_score(target, labels)

            correct += int(torch.sum(torch.argmax(output, dim=1) == target))
            total += len(target)

            # 梯度清零；反向传播；
            optimizer.zero_grad()
            loss = F.cross_entropy(output, target)  # 交叉熵损失函数；
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss = epoch_loss / (batch_idx + 1)
        print('epoch:%s' % epoch, 'accuracy：%.3f%%' % (correct * 100 / total), 'loss = %s' % loss)

    print('————————进行测试集验证————————')
    for epoch in range(1):
        correct = 0
        total = 0
        epoch_loss = 0
        net.train()
        batch_idx = 0
        for batch_idx, (data, target) in enumerate(test_iter):
            # print (data.shape)

            data = torch.as_tensor(data, dtype=torch.float32)
            target = target.long()  ##要保证label的数据类型是long
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)  # 将数据放入GPU
            output = net(data)
            # labels = output.argmax(dim= 1)
            # acc = accuracy_score(target, labels)

            correct += int(torch.sum(torch.argmax(output, dim=1) == target))
            total += len(target)

            # 梯度清零；反向传播；
            optimizer.zero_grad()
            loss = F.cross_entropy(output, target)  # 交叉熵损失函数；
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss = epoch_loss / (batch_idx + 1)
        print('epoch:%s' % epoch, 'accuracy：%.3f%%' % (correct * 100 / total), 'loss = %s' % loss)
