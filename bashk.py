import numpy as np
import nltk
import json
from nltk.stem.porter import PorterStemmer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
stemmer = PorterStemmer()
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
       
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
     return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, words):
             sentence_words = [stem(word) for word in tokenized_sentence]
             bag = np.zeros(len(words), dtype=np.float32)
             for idx,w in enumerate(words):
                         if w in sentence_words: 
                                  bag[idx] = 1
             return bag
x=open('D:\\details\\archive\\draw.json','r')
data=x.read()
obj=json.loads(data)
for i in obj:
       print(i['sQuestion'])
bas=[]
bos=[]
all_words=[]
for i in obj:
    ho=str(i['iIndex'])
    bos.append(ho)
for i,j in zip(obj,bos):
    b=tokenize((i['sQuestion']))
    all_words.extend(b)
    bas.append((b,j))

ignore_words = ['?', '.', '!']
all_words = [stem(b) for b in all_words if b not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(bos))
X_train = []
y_train = []
for (pattern_sentence, tag) in bas:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train)
print(y_train)
num_epochs = 10000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "D:\\details\\data5.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
