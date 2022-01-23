# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'RNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        # self.save_path = '/kaggle/working' + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        # self.log_path = '/kaggle/working' + '/log/' + self.model_name
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 5000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-4                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 64                                         # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.RNN = nn.RNN(config.embed, config.hidden_size, config.num_layers,
                            batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.RNN(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
#Iter:  27500,  Train Loss:  0.51,  Train Acc: 82.03%,  Val Loss:  0.57,  Val Acc: 83.35%,  Time: 0:04:44 *
# Test Loss:  0.56,  Test Acc: 83.78%
# Precision, Recall and F1-Score...
#                precision    recall  f1-score   support
#
#       finance     0.9009    0.7640    0.8268      1000
#        realty     0.8737    0.8720    0.8729      1000
#        stocks     0.7289    0.7530    0.7408      1000
#     education     0.8816    0.9160    0.8985      1000
#       science     0.7408    0.7430    0.7419      1000
#       society     0.8898    0.8400    0.8642      1000
#      politics     0.7792    0.8400    0.8085      1000
#        sports     0.9199    0.9300    0.9249      1000
#          game     0.8722    0.8460    0.8589      1000
# entertainment     0.8123    0.8740    0.8420      1000
#
#      accuracy                         0.8378     10000
#     macro avg     0.8399    0.8378    0.8379     10000
#  weighted avg     0.8399    0.8378    0.8379     10000
#
# Confusion Matrix...
# [[764  37 134  16  15   3  26   1   1   3]
#  [ 13 872  30  17  29   4  10   7   4  14]
#  [ 60  34 753   3  66   5  69   2   7   1]
#  [  1  10   1 916   4  11  18   6   5  28]
#  [  1  18  74   7 743  18  54   5  58  22]
#  [  2  12   8  25  11 840  43   1  19  39]
#  [  4   3  25  40  18  43 840   7   4  16]
#  [  1   4   0   3   6   4   3 930   1  48]
#  [  1   3   7   2  94   8   5   3 846  31]
#  [  1   5   1  10  17   8  10  49  25 874]]
