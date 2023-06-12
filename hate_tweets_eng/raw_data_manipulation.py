# from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

y_array = []
x_array = []

banned_words = ["\n", "@user", "?", ".", ]

# Function to delete emojisP
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def prepare_text(text):
    text = ' '.join(w for w in text.split() if w not in banned_words)
    text = deEmojify(text)
    text = text.replace(r'#', '')
    text = text.replace(r'ð', '')
    text = re.sub("\s\s+" , " ", text)
    return text


train_set = pd.read_csv('raw_data/train.csv')
train_set = train_set.reset_index()
train_set['tweet'] = train_set["tweet"].apply(prepare_text)
# print(train_set)

for index, row in train_set.iterrows():
    x_array.append(row['tweet'])
    y_array.append(int(row['label'])) 


X = x_array
y = y_array

with open ("processed/X.npy", 'wb') as file:
    np.save(file, x_array)

with open ("processed/Y.npy", 'wb') as file:
    np.save(file, y_array)

X_training, x_array_forTest, y_training, y_array_forTest = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
print(len(X_training), len(x_array_forTest), len(y_training), len(y_array_forTest))
X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, stratify=y_training, random_state=42)
train_class_ratios = np.bincount(y_train) / len(y_train)
val_class_ratios = np.bincount(y_val) / len(y_val)

# print("Training class ratios: ", train_class_ratios)
# print("Validation class ratios: ", val_class_ratios)


# Save train.npy
with open ('processed/X_train.npy', 'wb') as file:
    np.save(file, X_train)

with open ('processed/y_train.npy', 'wb') as file:
    np.save(file, y_train)

# # Save validation.npy
with open ('processed/X_val.npy', 'wb') as file:
    np.save(file, X_val)

with open ('processed/y_val.npy', 'wb') as file:
    np.save(file, y_val)


# # And the test set now

with open ("processed/X_forTest.npy", 'wb') as file:
    np.save(file, x_array_forTest)

with open ("processed/Y_forTest.npy", 'wb') as file:
    np.save(file, y_array_forTest)