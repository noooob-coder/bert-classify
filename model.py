import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class Config(object):
    def __init__(self):
        self.model_name='bert'
        self.train_path='./data/train.txt'
        self.dev_path='./data/dev.txt'
        self.test_path='./data/test.txt'
        self.class_list=[x.strip() for x in open("./data/class.txt",encoding='utf-8').readlines()]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 32  # mini-batch大小
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)  #全连接层

    def forward(self, x):
        context = x[0]  # 输入的句子，x矩阵的第一维度为句子本身
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _,pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
