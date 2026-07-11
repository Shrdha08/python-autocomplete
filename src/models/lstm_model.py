import torch.nn as nn

# for optuna
class LSTMnn(nn.Module):
  def __init__(self,vocab_size,emb_dim,hidden_dim,num_classes,no_of_lstm_layers):
    super().__init__()

    layers=[]
    layers.append(nn.Embedding(vocab_size,emb_dim))

    layers.append(nn.LSTM(emb_dim,hidden_dim,bidirectional = True,batch_first=True,num_layers=no_of_lstm_layers))

    layers.append(nn.Linear(hidden_dim,num_classes))

  def forward(self,input_ids):
    emb=self.embedding(input_ids)
    lstm_res,(h_n,c_n) = self.LSTM(emb) #two outputs are output of lstm layer and (h_n,c_n)
    pred = self.linear(h_n[-1])
    return pred

    ##no use

def objective(trial):

  # next hyperparameter values from search space
  num_lstm_layers=trial.suggest_int('num_hidden_layers',1,4)
  hidden_dim=trial.suggest_categorical('hidden_dim',[32,64,128,256])
  emb_dim=trial.suggest_int('emb_dim',100,500)

  # model initialise
  input_dim=len(vocab)
  output_dim=len(vocab)

  model=LSTMnn(input_dim,emb_dim,output_dim,num_lstm_layers,hidden_dim)
  model.to(device)

  # params init


  # training loop

  # evaluation