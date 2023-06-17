#https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/

from torch.utils.data import DataLoader
from utils import ClassificationCollator, PolEvalDataset, HateTweetsDataset ,train, validation
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          GPT2Config,
                          GPT2Tokenizer,
                          GPT2ForSequenceClassification,
                          get_linear_schedule_with_warmup,
                          AdamW)
import torch
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from tqdm import tqdm

set_seed(1235)
model_name_or_path = 'gpt2'

data_name = 'PolEval'
train_dataset = PolEvalDataset("train")
validation_dataset = PolEvalDataset("validation")
test_dataset = PolEvalDataset("test")

# data_name = 'HateTweets'
# train_dataset = HateTweetsDataset("train")
# validation_dataset = HateTweetsDataset("validation")
# test_dataset = HateTweetsDataset("test")

train
labels_ids = {'neg': 0, 'pos': 1}
epochs = 4
batch_size = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_length = None

print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=train_dataset.n_labels)

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)

model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = model.config.eos_token_id

model.to(device)

print('Model loaded to `%s`'%device)
gpt2_classificaiton_collator = ClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, 
                                                          max_sequence_len=max_length)


print('Dealing with Train...')
print('Created `train_dataset` with %d examples!'%len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

print('Dealing with Validation...')
print('Created `valid_dataset` with %d examples!'%len(validation_dataset))
valid_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
print('Created `valid_dataloader` with %d batches!'%len(valid_dataloader))

print('Dealing with Test....')
print('Created `test_dataset` with %d examples!'%len(test_dataset))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
print('Created `test_dataloader` with %d batches!'%len(test_dataloader))

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                  )

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

all_loss = {'train_loss':[], 'val_loss':[]}
all_acc = {'train_acc':[], 'val_acc':[]}

for epoch in tqdm(range(epochs)):
  print('Training on batches...')
  train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device, model)
  train_acc = accuracy_score(train_labels, train_predict)

  print('Validation on batches...')
  valid_labels, valid_predict, val_loss = validation(valid_dataloader, device, model)
  val_acc = accuracy_score(valid_labels, valid_predict)

  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))

  all_loss['train_loss'].append(train_loss)
  all_loss['val_loss'].append(val_loss)
  all_acc['train_acc'].append(train_acc)
  all_acc['val_acc'].append(val_acc)

plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

true_labels, predictions_labels, avg_epoch_loss = validation(test_dataloader, device, model)

evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))
print(evaluation_report)

plot_confusion_matrix(y_true=true_labels, y_pred=predictions_labels, 
                      classes=list(labels_ids.keys()), normalize=True, 
                      magnify=3, path=f"plots/{model_name_or_path}_{data_name}"
                      )

torch.save(model.state_dict(), f"models/{model_name_or_path}_{data_name}")