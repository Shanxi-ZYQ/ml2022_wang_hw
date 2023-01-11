import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel
import pandas as pd
import os
from sklearn.metrics import classification_report

# Bertmodel
class BertClassifier(nn.Module):
    def __init__(self, bert_config, num_labels, model_name,dropout_prob=0):
        super().__init__()
        self.num_labels = num_labels
        # 定义BERT模型
        self.bert = BertModel.from_pretrained(model_name)
        #dropout
        self.dropout = nn.Dropout(dropout_prob)
        # 定义分类器
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # BERT的输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 取[CLS]位置的pooled output
        pooled = bert_output[1]
        pooled = self.dropout(pooled)
        # 分类
        logits = self.classifier(pooled)
        # 返回softmax后结果
        return torch.softmax(logits, dim=1)

def predict():
    model_name = 'bert-base-chinese'
    labels = ['0', '1']
    bert_config = BertConfig.from_pretrained(model_name)

    # 定义模型
    model = BertClassifier(bert_config, len(labels), model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 加载训练好的模型
    model.load_state_dict(torch.load('models/best_bert.pkl', map_location=torch.device('cpu')))
    model.eval()

    filename = "CoCLSA-main/CoCLSA-main/test.tsv"

    with open(filename, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()
    test_text=[]
    test_labels=[]
    for line in lines:
        label, text = line.strip().split('\t')
        test_labels.append(int(label))
        test_text.append(text)


    pred_labels = []
    # while True:
    for text in test_text:
        token = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=80)
        input_ids = token['input_ids']
        attention_mask = token['attention_mask']
        token_type_ids = token['token_type_ids']

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)

        predicted = model(
            input_ids,
            attention_mask,
            token_type_ids,
        )
        # print(predicted)
        pred_label = int(torch.argmax(predicted, dim=1))
        pred_labels.append(pred_label)

    # print(pred_labels)
    print(classification_report(test_labels,pred_labels,target_names=['class0','class1']))

if __name__ == '__main__':
    predict()
