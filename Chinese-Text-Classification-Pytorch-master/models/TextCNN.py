# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        # self.save_path = '/kaggle/working'+ '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        # self.log_path = '/kaggle/working' + '/log/' + self.model_name
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)

'''Convolutional Neural Networks for Sentence Classification'''

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        # print(x.shape)
        x = F.relu(conv(x)).squeeze(3)
        # print(x.shape)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # print(x.shape)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        # print(out.shape)#128,32,300
        out = out.unsqueeze(1)
        # print(out.shape)#128,1,32,300
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        # print(out.shape)#128,10
        return out
#Iter:   4700,  Train Loss:  0.41,  Train Acc: 85.94%,  Val Loss:  0.32,  Val Acc: 90.39%,  Time: 0:31:09 *
# Test Loss:   0.3,  Test Acc: 91.00%
# Precision, Recall and F1-Score...
#                precision    recall  f1-score   support
#
#       finance     0.9144    0.8970    0.9056      1000
#        realty     0.9146    0.9420    0.9281      1000
#        stocks     0.8767    0.8460    0.8611      1000
#     education     0.9626    0.9510    0.9567      1000
#       science     0.8581    0.8830    0.8704      1000
#       society     0.8945    0.9070    0.9007      1000
#      politics     0.8840    0.9070    0.8954      1000
#        sports     0.9737    0.9250    0.9487      1000
#          game     0.9142    0.9160    0.9151      1000
# entertainment     0.9123    0.9260    0.9191      1000
#
#      accuracy                         0.9100     10000
#     macro avg     0.9105    0.9100    0.9101     10000
#  weighted avg     0.9105    0.9100    0.9101     10000
#
# Confusion Matrix...
# [[897  19  44   3   6  11  11   2   4   3]
#  [ 13 942  11   0   2  14   7   0   4   7]
#  [ 49  24 846   1  32   5  34   2   6   1]
#  [  1   3   2 951   8  13   9   2   1  10]
#  [  1   5  29   5 883  17  14   3  29  14]
#  [  5  18   2  13  11 907  29   0   4  11]
#  [  7   7  18   6  17  28 907   1   2   7]
#  [  2   4   6   4   7   6  10 925  10  26]
#  [  2   2   6   1  48   7   1   7 916  10]
#  [  4   6   1   4  15   6   4   8  26 926]]