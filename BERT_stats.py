
from sklearn.metrics import confusion_matrix
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn

device = "cuda:0" if torch.cuda.is_available() else "cpu:0"

class SentimentClassifier(nn.Module):
    def __init__(self, bert_model):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits

class EmojiDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Text']
        acc_em = self.data.iloc[idx][' Accurate Emojis']
        suit_em = self.data.iloc[idx][' Suitable Emojis']
        labels = [1 if acc_em == ' T' else 0] + list(map(int, suit_em[1:].split(" ")))
        
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        labels = torch.tensor(labels, dtype=torch.float32)

        return input_ids, attention_mask, labels

# Example predictions and true labels
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = EmojiDataset(
    r"E:\Users\Dmytro\Desktop\PyTorch\ML_Emoji\data\chatgpt_sorted.csv.csv",
    tokenizer=tokenizer,
    max_length=255
)
num_classes = 2
num_epochs = 4
batch_size = 1

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

bert_model = BertModel.from_pretrained('bert-base-uncased')
chk = torch.load("SHBERT_Emoji.pth")
model = SentimentClassifier(bert_model)
model.load_state_dict(chk)

predictions = []
true_labels = []

i = 0
model.eval()
with torch.no_grad():
    for batch in train_loader:
        if i > 100:
            break
        i += 1
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2]
        logits = model(input_ids, attention_mask=attention_mask)
        outputs = F.sigmoid(logits[0][1:])
        to_add = (outputs > 0.26).cpu().numpy().astype(np.float32)
        predictions.extend(to_add)
        true_labels.extend(labels[0][1:].cpu().numpy())


# Compute confusion matrix
print(true_labels)
print(predictions)
cm = confusion_matrix(true_labels, predictions)

# Visualize the confusion matrix (optional)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

