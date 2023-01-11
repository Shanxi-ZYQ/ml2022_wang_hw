import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig, AdamW
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from sklearn.svm import SVC

class MyDataset(Dataset):
    def __init__(self, filename,model_name):
        # 数据集初始化
        self.labels = ['0', '1']
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_id = []
        self.load_data(filename)

    def load_data(self, filename):
        # 加载数据
        print('loading data from:', filename)
        with open(filename, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
#         i=0
        for line in tqdm(lines[1:], ncols=100):
            label, text = line.strip().split('\t')
            label_id = self.labels.index(label)
            token = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=70)
            self.input_ids.append(np.array(token['input_ids']))
            self.token_type_ids.append(np.array(token['token_type_ids']))
            self.attention_mask.append(np.array(token['attention_mask']))
            self.label_id.append(label_id)
#             i+=1
#             if i>64:
#                 break

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.label_id[index]

    def __len__(self):
        return len(self.input_ids)

# Bertmodel
class BertClassifier(nn.Module):
    def __init__(self, bert_config, num_labels, model_name,dropout_prob):
        super().__init__()
        self.num_labels = num_labels
        # 定义BERT模型
        self.bert = BertModel.from_pretrained(model_name)
        #dropout
        self.dropout = nn.Dropout(dropout_prob)
        self.encoder = nn.Sequential(
            nn.Linear(bert_config.hidden_size,384),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(384,32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(16,4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4,16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 384),
            nn.ReLU(),
            nn.Linear(384, bert_config.hidden_size),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # BERT的输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 取[CLS]位置的pooled output
        pooled = bert_output[1]
        # 分类
        encoder=self.encoder(pooled)
        decoder=self.decoder(encoder)
        # 返回softmax后结果
        return pooled,encoder,decoder

def train_model():
    # 参数设置
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    learning_rate = 2e-6  # Learning Rate不宜太大
    model_name = 'bert-base-chinese'
    num_labels = 2

    # 获取到dataset
    train_dataset = MyDataset('/kaggle/input/binju-classify/data/train/train_split.csv',model_name)
    valid_dataset = MyDataset('/kaggle/input/binju-classify/data/train/vaild_split.csv',model_name)
    # test_dataset = CNewsDataset('data/cnews/cnews.test.txt')

    # 生成Batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained(model_name)

    # 初始化模型
    model = BertClassifier(bert_config, num_labels,model_name,dropout_prob=0.1).to(device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    model.train()
    for epoch in range(1, epochs + 1):
        losses = 0  # 损失
        accuracy = 0  # 准确率


        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_bar = tqdm(train_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            # 梯度清零
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)

            # 传入数据，调用model.forward()
            pooled,encoder,decoder = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )

            # 计算loss
            loss = criterion(pooled,decoder)
            losses += loss.item()

#             pred_labels = torch.argmax(output, dim=1)  # 预测出的label
#             acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
#             accuracy += acc

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item())

        average_loss = losses / len(train_dataloader)
#         average_acc = accuracy / len(train_dataloader)

        print('\tLoss:', average_loss)
    torch.save(model.state_dict(), 'models/best_bert2.pkl')
        # 验证
#         model.eval()
#         losses = 0  # 损失
#         accuracy = 0  # 准确率
#         valid_bar = tqdm(valid_dataloader, ncols=100)
#         for input_ids, token_type_ids, attention_mask, label_id in valid_bar:
#             valid_bar.set_description('Epoch %i valid' % epoch)

#             encoder,decoder = model(
#                 input_ids=input_ids.to(device),
#                 attention_mask=attention_mask.to(device),
#                 token_type_ids=token_type_ids.to(device),
#             )

#             loss = criterion(encoder,decoder)
#             losses += loss.item()

#             pred_labels = torch.argmax(output, dim=1)  # 预测出的label
#             acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
#             accuracy += acc
#             valid_bar.set_postfix(loss=loss.item(), acc=acc)

#         average_loss = losses / len(valid_dataloader)
#         average_acc = accuracy / len(valid_dataloader)

#         print('\tValid ACC:', average_acc, '\tLoss:', average_loss)

#         if not os.path.exists('models'):
#             os.makedirs('models')

#         # 判断并保存验证集上表现最好的模型
#         if average_acc > best_acc:
#             best_acc = average_acc
#             torch.save(model.state_dict(), 'models/best_bert2.pkl')
#         model.train()


if __name__ == '__main__':
    train_model()