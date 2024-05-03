import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu:0 "

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

class SarcasmDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['tweets']
        label = self.data.iloc[idx]['class']
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = torch.tensor(label, dtype=torch.float32)
        return input_ids, attention_mask, label


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, inp in enumerate(train_loader):
            input_ids, attention_mask, emojis = inp
            input_ids, attention_mask, emojis = torch.tensor(input_ids).to(device), attention_mask.to(device), torch.tensor(emojis).to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)# .logits
            outputs = F.sigmoid(logits).squeeze(0)
            loss = criterion(logits, emojis)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 3 == 0:
                print(f"Iteration {i + 1} : Loss: {total_loss / batch_size}")
                total_loss = 0.0
        print(f"Epoch {epoch + 1} : Loss: {total_loss / len(train_loader)}")


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = EmojiDataset(
    r"E:\Users\Dmytro\Desktop\PyTorch\ML_Emoji\data\chatgpt_sorted.csv.csv",
    tokenizer=tokenizer,
    max_length=255
)
num_epochs = 4
batch_size = 32
learning_rate = 0.00001

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

bert_model = BertModel.from_pretrained('bert-base-uncased')
chk = torch.load("SHBERT_Emoji.pth")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(bert_model.parameters(), lr=learning_rate)
model = SentimentClassifier(bert_model)
model.load_state_dict(chk)
train_model(model.to(device), train_loader, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), "SHBERT_Emoji2.pth")
