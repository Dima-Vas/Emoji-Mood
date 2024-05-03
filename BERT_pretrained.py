import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu:0 "


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
        label = torch.tensor(0.0 if label == "regular" else 0.2 if label == "figurative" else 0.4 if label == "irony" else 1.0, dtype=torch.float32)
        return input_ids, attention_mask, label

class SentimentClassifier(nn.Module):
    def __init__(self, bert_model):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, inp in enumerate(train_loader):
            input_ids, attention_mask, emojis = inp
            input_ids, attention_mask, emojis = torch.tensor(input_ids).to(device), attention_mask.to(device), torch.tensor(emojis).to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            outputs = F.sigmoid(logits).reshape(-1, batch_size)
            loss = criterion(logits.reshape(-1, batch_size), emojis.reshape(-1, batch_size))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 3 == 0:
                print(f"Iteration {i + 1} : Loss: {total_loss / batch_size}")
                total_loss = 0.0
        print(f"Epoch {epoch + 1} : Loss: {total_loss / len(train_loader)}")


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = SarcasmDataset(
    r"E:\Users\Dmytro\Desktop\PyTorch\ML_Emoji\data\train.csv",
    tokenizer=tokenizer,
    max_length=255
)

num_epochs = 4
batch_size = 32
learning_rate = 0.0001

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

bert_model = BertModel.from_pretrained('bert-base-uncased')
chk = torch.load("SHBERT_FC_0001.pth")
# bert_model.num_hidden_layers = batch_size

model = SentimentClassifier(bert_model)

model.load_state_dict(chk)
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.SmoothL1Loss(beta=0.99)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# bert_model.sequence_summary.summary = torch.nn.Linear(in_features=bert_model.sequence_summary.summary.in_features, out_features=1, bias=True)


train_model(model.to(device), train_loader, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), "SHBERT_FC_0001.pth")
