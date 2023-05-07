import random
import collections
import torch
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.utils.data as Data


class DataUtil:
    def __init__(self, word2vec_path, train_path, test_path,valid_path, batch_size):
        word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

        with open(train_path, 'r',encoding="utf-8") as f:
            lines = f.readlines()
            words = []
            for line in lines:
                label, review = line.strip().split('\t')
                words += review.split(' ')

        self.word2idx = {}
        self.embed = torch.zeros(len(words) + 1, word2vec.vectors[0].shape[0])  # 初始化为0

        oov_count = 0  # out of vocabulary
        for i, word in enumerate(words):
            try:
                self.embed[i + 1, :] = torch.from_numpy(word2vec[word].copy())
                self.word2idx[word] = i + 1
            except KeyError:
                oov_count += 1
        if oov_count > 0:
            pass
            # print("There are %d oov words." % oov_count)

        self.train_set, _ = self.process_corpus(train_path)
        self.test_set, _ = self.process_corpus(test_path)
        self.valid_set, _ = self.process_corpus(valid_path)

        self.train_iter = DataLoader(self.train_set, batch_size, shuffle=True)
        self.test_iter = DataLoader(self.test_set, batch_size)
        self.valid_iter = DataLoader(self.valid_set, batch_size)

    def word_to_id(self, word):
        return self.word2idx[word] if self.word2idx[word] else 0

    def process_corpus(self, path):
        max_l = 500

        def pad(x):
            return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

        with open(path, 'r',encoding='utf-8') as f:
            lines = f.readlines()
            reviews = []
            labels = []
            for line in lines:
                label, review = line.strip().split('\t')
                review = review.split(' ')
                data = []
                for word in review:
                    try:
                        data.append(self.word_to_id(word))
                    except KeyError:
                        pass
                data = pad(data)
                reviews.append(torch.tensor(data))
                labels.append(int(label))

        review_set = torch.stack(reviews)
        label_set = torch.tensor(labels)
        data_set = Data.TensorDataset(review_set, label_set)
        return data_set, data
