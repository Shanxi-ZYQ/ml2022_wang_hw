from sklearn.svm import SVC
import joblib
import pickle
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel

# Bertextract
class Bertextract(nn.Module):
    def __init__(self, bert_config, num_labels, model_name, dropout_prob=0):
        super().__init__()
        self.num_labels = num_labels
        # 定义BERT模型
        self.bert = BertModel.from_pretrained(model_name)
        # dropout
        self.dropout = nn.Dropout(dropout_prob)
        self.encoder = nn.Sequential(
            nn.Linear(bert_config.hidden_size, 384),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(384, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(16, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
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
        encoder = self.encoder(pooled)
        decoder = self.decoder(encoder)
        # 返回softmax后结果
        return pooled, encoder, decoder

def svm_train():
    model_name = 'bert-base-chinese'
    labels = ['0', '1']
    bert_config = BertConfig.from_pretrained(model_name)
    filename = 'CoCLSA-main/CoCLSA-main/train.tsv'

    # 定义模型
    model = Bertextract(bert_config, len(labels), model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 加载训练好的模型
    model.load_state_dict(torch.load('models/best_bert2.pkl', map_location=torch.device('cpu')))
    model.eval()


    with open(filename, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()
    test_text = []
    test_labels = []
    for line in lines:
        label, text = line.strip().split('\t')
        test_labels.append(int(label))
        test_text.append(text)

    sentences_encoder = []
    for text in test_text:
        token = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=80)
        input_ids = token['input_ids']
        attention_mask = token['attention_mask']
        token_type_ids = token['token_type_ids']

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)

        pooled, encoder, decoder = model(
            input_ids,
            attention_mask,
            token_type_ids,
        )
        encoder = encoder.squeeze().detach().numpy()
        sentences_encoder.append(encoder)

    #svm
    svc = SVC(C=2,kernel="rbf")
    svc.fit(sentences_encoder,test_labels)
    joblib.dump(svc, 'svm_model_rbf_C2.pkl')

if __name__ == "__main__":
    svm_train()

