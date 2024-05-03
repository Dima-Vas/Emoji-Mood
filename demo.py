import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn

text = "OMG nothing better than some kids stealing my bike!😊"

class SentimentClassifier(nn.Module):
    def __init__(self, bert_model):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(768, 5)
        self.dropout = nn.Dropout(0.28)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_model = BertModel.from_pretrained('bert-base-uncased')
chk = torch.load("SHBERT_Emoji.pth")

model = SentimentClassifier(bert_model)
encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=255)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
model.load_state_dict(chk)
output = model(input_ids, attention_mask)[0][1:]
output_probs = F.sigmoid(output)
emojis = ["😂", "😠", "😒", "😔"]
for i in range(4) :
    print(f"{emojis[i]} : {output_probs[i]*100:.1f}%")
