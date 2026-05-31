from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch 

# loading dataset
ds = load_dataset("Nan-Do/code-search-net-python")

# using tokenised text
code_tokens=[]

for token_list in ds['train']['code_tokens']:
  code_tokens.append(token_list)

# train-test split to split data
# train - 80%
# test - 10%
# valid - 10%
train_data,temp_data=train_test_split(code_tokens,test_size=0.2,random_state=42)
test_data,valid_data=train_test_split(temp_data,test_size=0.5,random_state=42)

# build vocab of max 50000 most common words
vocab={}
from collections import Counter
vocab['<pad>']=0
vocab['<unk>']=1

counter=Counter()
for token_list in train_data:
  counter.update(token for token in token_list)

for token,freq in counter.most_common(50000):
  if freq>=2:
    vocab[token]=len(vocab)

def encode(token_list):
  return [vocab[token] if token in vocab else vocab['unk'] for token in token_list]

mapped_train_tokens=[encode(token_list) for token_list in train_data]
mapped_test_tokens=[encode(token_list) for token_list in test_data]
mapped_valid_tokens=[encode(token_list) for token_list in valid_data]

# generating training data
def data_prep(mapped_tokens):
  inputs=[]
  target=[]

  seq_len=10

  for token_list in mapped_tokens:
    for i in range(seq_len,len(token_list)):
      inputs.append(token_list[i-seq_len:i])
      target.append(token_list[i])

  return inputs,target

train_inputs,train_labels=data_prep(mapped_train_tokens)
test_inputs,test_labels=data_prep(mapped_test_tokens)
valid_inputs,valid_labels=data_prep(mapped_valid_tokens)

class data(Dataset):
  def __init__(self,X,y):
    super().__init__()
    self.X=X
    self.y=y

  def __len__(self):
    return len(self.X)

  def __getitem__(self,index):
    return torch.tensor(self.X[index]),torch.tensor(self.y[index])
  
def get_dataloaders():
    train_data = data(train_inputs,train_labels)
    test_data = data(test_inputs,test_labels)
    valid_data = data(valid_inputs,valid_labels)

    train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=64,shuffle=True)
    valid_loader = DataLoader(valid_data,batch_size=64,shuffle=True)

    return train_loader,test_loader,valid_loader,vocab