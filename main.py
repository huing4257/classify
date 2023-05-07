from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
from model import LSTM, TextCNN,MLP,GRU
import torch
from load import DataUtil
import numpy as np
import torch.nn.functional as F
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', '-p',dest='pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--model', type=str, default='cnn',choices=['cnn', 'mlp', 'gru','lstm'], help='model type')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--epoch', type=int, default=5, help='epoch')

    args = parser.parse_args()

    word2vec_path = 'Dataset/wiki_word2vec_50.bin'
    train_path = 'Dataset/train.txt'
    test_path = 'Dataset/test.txt'
    valid_path = 'Dataset/validation.txt'
    batch_size = 64
    data_util = DataUtil(word2vec_path, train_path, test_path, valid_path,batch_size)
    embed = data_util.embed
    train_iter = data_util.train_iter
    test_iter = data_util.test_iter
    valid_iter = data_util.valid_iter

    
    lr, num_epochs = args.lr, args.epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'cnn':
        embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
        cnn =TextCNN(embed, kernel_sizes, nums_channels, 0.1) 
        net = cnn.to(device)
    elif args.model == 'mlp':
        net =  MLP(embed, [10000,1000,1000], 3, 0.1).to(device)
    elif args.model == 'gru':
        net = GRU(embed, 100, 3, 0.1).to(device)
    elif args.model == 'lstm':
        net = LSTM(embed, 100, 3, 0.1).to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    

    if args.pretrained:
        net = torch.load(f'pre_{args.model}_model.pkl')
    else:
        print('——————————进行训练———————————')
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
            print('epoch:%s' % epoch, 'accuracy:%.3f%%' % (correct * 100 / total), 'loss = %s' % loss)
        torch.save(net, 'tmp.pkl')

    print('————————进行验证集验证————————')
    for epoch in range(1):
        correct = 0
        total = 0
        tp = 0
        fp = 0
        p = 0
        epoch_loss = 0
        net.train()
        batch_idx = 0
        for batch_idx, (data, target) in enumerate(valid_iter):
            data = torch.as_tensor(data, dtype=torch.long)
            target = target.long()  ##要保证label的数据类型是long
            data, target = data.to(device), target.to(device)  # 将数据放入GPU
            output = net(data)

            output=torch.argmax(output, dim=1)

            correct += int(torch.sum(output == target))
            total += len(target)
            tp += int(torch.sum(output * target))
            fp += int(torch.sum( output * (1 - target)))
            p += int(torch.sum(target))
            
        print( 'accuracy:%.3f%%' % (correct * 100 / total))
        precision= tp / (tp + fp)
        recall= tp / p
        f1=2*precision*recall/(precision + recall)
        print(f'f1:{f1}')


    print('————————进行测试集验证————————')
    for epoch in range(1):
        correct = 0
        total = 0
        epoch_loss = 0
        tp = 0
        fp = 0
        p = 0
        net.train()
        batch_idx = 0
        for batch_idx, (data, target) in enumerate(test_iter):
            data = torch.as_tensor(data, dtype=torch.long)
            target = target.long()  ##要保证label的数据类型是long
            data, target = data.to(device), target.to(device)  # 将数据放入GPU
            output = net(data)

            output=torch.argmax(output, dim=1)

            correct += int(torch.sum(output == target))
            total += len(target)
            tp += int(torch.sum(output * target))
            fp += int(torch.sum( output* (1 - target)))
            p+=int(torch.sum(target))
            
        print('accuracy:%.3f%%' % (correct * 100 / total))
        precision= tp / (tp + fp)
        recall= tp / p
        f1=2*precision*recall/(precision + recall)
        print(f'f1:{f1}')
    


