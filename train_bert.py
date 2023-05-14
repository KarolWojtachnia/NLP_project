from transformers import BertModel, BertTokenizerFast
import torch.nn as nn
import torch
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder

# class BERT_Arch(nn.Module):
#     def __init__(self, bert):
#         super(BERT_Arch, self).__init__()
#         self.bert = bert 
#         self.dropout = nn.Dropout(0.1)
#         self.relu =  nn.ReLU()
#         self.fc1 = nn.Linear(768,512)
#         self.fc2 = nn.Linear(512,2)
#         self.softmax = nn.LogSoftmax(dim=1)

#     #define the forward pass
#     def forward(self, sent_id, mask):
#         _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
#         x = self.fc1(cls_hs)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x


EPOCHS = 10
BATCH_SIZE = 32


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

bert = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
label_encoder = LabelEncoder()

tokens_train = tokenizer.batch_encode_plus(
    X_train.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True,
    return_tensors='pt'
)

X_train = torch.tensor(label_encoder.fit_transform(X_train))
y_train = torch.tensor(y_train.astype(int))

# tokens_val = tokenizer.batch_encode_plus(
#     X_val.tolist(),
#     max_length = 25,
#     pad_to_max_length=True,
#     truncation=True
# )
# X_val = torch.tensor(label_encoder.fit_transform(X_val))
# y_val = torch.tensor(y_val.astype(int))


train_mask = tokens_train['attention_mask']
train_data = TensorDataset(X_train, train_mask, y_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

val_mask = torch.tensor(tokens_val['attention_mask'])
val_data = TensorDataset(X_val, val_mask, y_val)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)


# Model
# model = BERT_Arch(bert).to(device)
model = bert()

optimizer = AdamW(model.parameters(), lr=1e-5)
cross_entropy  = nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    total_preds = []
  
    for batch in train_dataloader:
        model.zero_grad()
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch

        preds = model(sent_id.unsqueeze(0), mask)
        loss = cross_entropy(preds, labels)
        total_loss = total_loss + loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = preds.detach().cpu().numpy()

    total_preds.append(preds)
    avg_loss = total_loss / len(train_dataloader)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


def evaluate():
    print("\nEvaluating...")
    total_loss = 0
    total_preds = []

    model.eval()
    for batch in tqdm(val_dataloader):
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch

        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds,labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    avg_loss = total_loss / len(val_dataloader) 
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


# main
best_valid_loss = float('inf')
train_losses, valid_losses =[], []

#for each epoch
for epoch in range(EPOCHS):
    print('\n Epoch {:} / {:}'.format(epoch + 1, EPOCHS))

    train_loss, _ = train()
    valid_loss, _ = evaluate()
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')