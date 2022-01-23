# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN'
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
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

    '''变长RNN，效果差不多，甚至还低了点...'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out
#Iter:   4700,  Train Loss:  0.35,  Train Acc: 90.62%,  Val Loss:   0.3,  Val Acc: 90.44%,  Time: 0:02:56 *
#Test Loss:   0.3,  Test Acc: 90.48%
# Precision, Recall and F1-Score...
#                precision    recall  f1-score   support
#
#       finance     0.9093    0.9020    0.9056      1000
#        realty     0.9248    0.9230    0.9239      1000
#        stocks     0.8912    0.8030    0.8448      1000
#     education     0.9229    0.9450    0.9338      1000
#       science     0.8118    0.8670    0.8385      1000
#       society     0.8729    0.9340    0.9024      1000
#      politics     0.9026    0.8620    0.8818      1000
#        sports     0.9866    0.9600    0.9731      1000
#          game     0.9318    0.9150    0.9233      1000
# entertainment     0.9036    0.9370    0.9200      1000
#
#      accuracy                         0.9048     10000
#     macro avg     0.9057    0.9048    0.9047     10000
#  weighted avg     0.9057    0.9048    0.9047     10000
#
# Confusion Matrix...
# [[902  17  32   7  15  13   8   1   1   4]
#  [  8 923  12   2  20  15   6   3   2   9]
#  [ 63  26 803   7  54   3  33   0   9   2]
#  [  0   0   0 945  11  22   5   1   1  15]
#  [  4   5  23  12 867  21  16   0  40  12]
#  [  1  11   0  18   4 934  13   0   1  18]
#  [ 10   2  18  15  31  43 862   0   2  17]
#  [  0   2   3   3   4   5   6 960   0  17]
#  [  2   2   9   5  50   6   2   3 915   6]
#  [  2  10   1  10  12   8   4   5  11 937]]
