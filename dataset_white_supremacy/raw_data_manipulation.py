import pandas as pd
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split

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

annotations = pd.read_csv("raw_data/annotations_metadata.csv")
X_array = []
Y_array = []

# Train dataset
input_path = "raw_data/sampled_train"
dir = os.listdir(input_path)
for i, file in enumerate(dir):
    # print(i, file)
    X = None
    with open(f"{input_path}/{file}", 'r', encoding="utf-8") as text_file:
        X = text_file.read()
    X = prepare_text(X)
    file = file.replace(".txt","")
    y = list(annotations[annotations["file_id"]==file]["label"])[0]

    if y == "noHate":
        y = 0
    else:
        y = 1
    X_array.append(X)
    Y_array.append(y)

print(X_array[0], Y_array[0])
print(X_array[1], Y_array[1])

with open ("processed/X.npy", 'wb') as file:
    np.save(file, X_array)

with open ("processed/Y.npy", 'wb') as file:
    np.save(file, Y_array)

X_train, X_val, y_train, y_val = train_test_split(X_array, Y_array, test_size=0.2, stratify=Y_array, random_state=42)
train_class_ratios = np.bincount(y_train) / len(y_train)
val_class_ratios = np.bincount(y_val) / len(y_val)

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
input_path = "raw_data/sampled_test"
dir = os.listdir(input_path)
for i, file in enumerate(dir):
    print(i, file)
    X = None
    with open(f"{input_path}/{file}", 'r', encoding="utf-8") as text_file:
        X = text_file.read()
    X = prepare_text(X)
    file = file.replace(".txt","")
    y = list(annotations[annotations["file_id"]==file]["label"])[0]
    if y == "noHate":
        y = 0
    else:
        y = 1
    X_array.append(X)
    Y_array.append(y)

print(X_array[0], Y_array[0])
print(X_array[1], Y_array[1])

with open ("processed/X_forTest.npy", 'wb') as file:
    np.save(file, X_array)

with open ("processed/Y_forTest.npy", 'wb') as file:
    np.save(file, Y_array)