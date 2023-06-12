# Na podstawie https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/

import numpy as np
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset

# Class responsible for data handling

class PolEvalDataset(Dataset):
  def __init__(self, data_type):

    self.texts = []
    self.labels = []
    X = None
    y = None

    if data_type == "train":
        X = np.load("PolEval/processed/X_train.npy")
        y = np.load("PolEval/processed/y_train.npy")
    elif data_type == "validation":
        X = np.load("PolEval/processed/X_val.npy")
        y = np.load("PolEval/processed/y_val.npy")
    elif data_type == "test":
        X = np.load("PolEval/processed/X_forTest.npy")
        y = np.load("PolEval/processed/Y_forTest.npy")
    
    self.texts = np.ndarray.tolist(X)
    self.labels = np.ndarray.tolist(y)
    self.n_examples = len(self.labels)
    self.n_labels = np.unique(y).size
    return

  def __len__(self):
    return self.n_examples

  def __getitem__(self, item):
    return {'text':self.texts[item],
            'label':self.labels[item]}
  
class HateTweetsDataset(Dataset):
  def __init__(self, data_type):

    self.texts = []
    self.labels = []
    X = None
    y = None

    if data_type == "train":
        X = np.load("hate_tweets_eng/processed/X_train.npy")
        y = np.load("hate_tweets_eng/processed/y_train.npy")
    elif data_type == "validation":
        X = np.load("hate_tweets_eng/processed/X_val.npy")
        y = np.load("hate_tweets_eng/processed/y_val.npy")
    elif data_type == "test":
        X = np.load("hate_tweets_eng/processed/X_forTest.npy")
        y = np.load("hate_tweets_eng/processed/Y_forTest.npy")
    
    self.texts = np.ndarray.tolist(X)
    self.labels = np.ndarray.tolist(y)
    self.n_examples = len(self.labels)
    self.n_labels = np.unique(y).size
    return

  def __len__(self):
    return self.n_examples

  def __getitem__(self, item):
    return {'text':self.texts[item],
            'label':self.labels[item]}
  
 

# Collaor for classification, used in the DataLoader to create the bathes of data that get fed to the model
class ClassificationCollator(object):

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder
        return

    def __call__(self, sequences):
        
        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        print(labels)
        # labels = [self.labels_encoder[label] for label in labels]
        # print(labels)
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        inputs.update({'labels':torch.tensor(labels)})
        return inputs
    
def train(dataloader, optimizer_, scheduler_, device_, model):

    predictions_labels = []
    true_labels = []
    total_loss = 0

    model.train()

    for batch in tqdm(dataloader):

        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

        model.zero_grad()

        outputs = model(**batch)

        loss, logits = outputs[:2]
        total_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer_.step()
        scheduler_.step()
        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    avg_epoch_loss = total_loss / len(dataloader)

    return true_labels, predictions_labels, avg_epoch_loss

# Validation function to evaluate model performance on a separate set of data.
def validation(dataloader, device_, model):

    predictions_labels = []
    true_labels = []

    total_loss = 0

    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

        with torch.no_grad():        
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

    avg_epoch_loss = total_loss / len(dataloader)

    return true_labels, predictions_labels, avg_epoch_loss
