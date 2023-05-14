# from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import torch
from sklearn.model_selection import train_test_split

y_array = []
x_array = []

banned_words = ["\n", "@anonymized_account"]

# Function to delete emojis
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


with open('training_set_clean_only_text.txt', 'r', encoding='utf-8') as tags:
    content = tags.readlines()

    for line in content:
        line = ' '.join(w for w in line.split() if w not in banned_words)
        line = deEmojify(line)
        x_array.append(line)



# zmienna = np.load('X.npy')


# Save Y.npy
with open('training_set_clean_only_tags.txt', 'r', encoding='utf-8') as tags:
    content = tags.readlines()
    for line in content:
        y_array.append(int(line))

X = x_array
y = y_array

with open ("X.npy", 'wb') as file:
    np.save(file, x_array)

with open ("Y.npy", 'wb') as file:
    np.save(file, y_array)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_class_ratios = np.bincount(y_train) / len(y_train)
val_class_ratios = np.bincount(y_val) / len(y_val)

print("Training class ratios: ", train_class_ratios)
print("Validation class ratios: ", val_class_ratios)


# Save train.npy
with open ('X_train.npy', 'wb') as file:
    np.save(file, X_train)

with open ('y_train.npy', 'wb') as file:
    np.save(file, y_train)

# # Save validation.npy
with open ('X_val.npy', 'wb') as file:
    np.save(file, X_val)

with open ('y_val.npy', 'wb') as file:
    np.save(file, y_val)

