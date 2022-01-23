# Chinese-Text-Classification-Pytorch
中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于pytorch。

数据以字为单位输入模型，预训练词向量使用 [搜狗新闻 Word+Character 300d](https://github.com/Embedding/Chinese-Word-Vectors)，[点这里下载](https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ)  

## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX

## 中文数据集
我从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万


### 更换自己的数据集
 - 如果用字，按照我数据集的格式来格式化你的数据。  
 - 如果用词，提前分好词，词之间用空格隔开，`python run.py --model TextCNN --word True`  
 - 使用预训练词向量：utils.py的main函数可以提取词表对应的预训练词向量。  

## 使用说明
```
# 训练并测试：
# TextCNN
python run.py --model TextCNN

# TextRNN
python run.py --model TextRNN

# TextRNN_Att
python run.py --model TextRNN_Att

# TextRCNN
python run.py --model TextRCNN

# FastText, embedding层是随机初始化的
python run.py --model FastText --embedding random 

# DPCNN
python run.py --model DPCNN

# Transformer
python run.py --model Transformer
```
