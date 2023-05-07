from model import LSTM, TextCNN, MLP, GRU
import torch
from load import DataUtil
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from visdom import Visdom


def get_acc_f1(data_iter):
    for epoch_id in range(1):
        correct1 = 0
        total1 = 0
        tp = 0
        fp = 0
        p = 0
        for batch_id, (data1, target1) in enumerate(data_iter):
            data1 = torch.as_tensor(data1, dtype=torch.long)
            target1 = target1.long()
            data1, target1 = data1.to(device), target1.to(device)
            out = net(data1)

            out = torch.argmax(out, dim=1)

            correct1 += int(torch.sum(out == target1))
            total1 += len(target)
            tp += int(torch.sum(output * target1))
            fp += int(torch.sum(output * (1 - target1)))
            p += int(torch.sum(target))

        precision = tp / (tp + fp)
        recall = tp / p
        f1 = 2 * precision * recall / (precision + recall)
    return correct * 100 / total, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', '-p', dest='pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp', 'gru', 'lstm'], help='model type')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--epoch', type=int, default=5, help='epoch')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--manual_stop', '-m', dest='manual_stop', action='store_true', help='stop manually')

    args = parser.parse_args()

    vis = Visdom(env='Text_Classification')
    vis.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))

    word2vec_path = 'Dataset/wiki_word2vec_50.bin'
    train_path = 'Dataset/train.txt'
    test_path = 'Dataset/test.txt'
    valid_path = 'Dataset/validation.txt'
    batch_size = 64
    data_util = DataUtil(word2vec_path, train_path, valid_path, test_path, batch_size)
    embed = data_util.embed
    train_iter = data_util.train_iter
    test_iter = data_util.test_iter
    valid_iter = data_util.valid_iter

    lr, num_epochs = args.lr, args.epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = None
    if args.model == 'cnn':
        embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
        cnn = TextCNN(embed, kernel_sizes, nums_channels, 0.1)
        net = cnn.to(device)
    elif args.model == 'mlp':
        net = MLP(embed, 100, 0.1).to(device)
    elif args.model == 'gru':
        net = GRU(embed, 100, 3, 0.1).to(device)
    elif args.model == 'lstm':
        net = LSTM(embed, 100, 3, 0.1).to(device)

    if net is None:
        raise ValueError('model type error')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    if args.pretrained:
        net = torch.load(f'pre_{args.model}_model.pkl')
    else:
        print('--------train--------')
        for epoch in range(num_epochs):
            correct = 0
            total = 0
            epoch_loss = 0
            net.train()
            batch_idx = 0
            for batch_idx, (data, target) in enumerate(tqdm(train_iter)):
                data = torch.as_tensor(data, dtype=torch.long)
                target = target.long()
                optimizer.zero_grad()
                data, target = data.to(device), target.to(device)
                output = net(data)

                correct += int(torch.sum(torch.argmax(output, dim=1) == target))
                total += len(target)

                # 梯度清零；反向传播；
                optimizer.zero_grad()
                loss = F.cross_entropy(output, target)  # 交叉熵损失函数；
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            loss = epoch_loss / (batch_idx + 1)
            vis.line([loss], [epoch], win='train_loss', update='append')

            print('epoch:%s' % epoch, 'accuracy:%.3f%%' % (correct * 100 / total), 'loss = %s' % loss)
            valid_acc, f1 = get_acc_f1(valid_iter)
            print('valid_acc:%.3f%%' % valid_acc)
            print(f'f1:{f1}')
            if args.manual_stop:
                is_continue = input('是否继续训练？（y/n）')
                if is_continue == 'n':
                    break

        torch.save(net, 'tmp.pkl')

    net.eval()

    print('--------test--------')
    acc, f1 = get_acc_f1(test_iter)
    print('accuracy:%.3f%%' % acc)
    print(f'f1:{f1}')
