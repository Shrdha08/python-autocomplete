import torch.nn as nn

def objective(trial):

  # next hyperparameter values from search space
  num_lstm_layers=trial.suggest_int('num_hidden_layers',1,4)
  hidden_dim=trial.suggest_categorical('hidden_dim',[32,64,128,256])
  emb_dim=trial.suggest_int('emb_dim',100,500)
  lr=trial.suggest_float('lr',1e-5,1e-3,log=True)
  dropout = trial.suggest_float("dropout", 0.1, 0.5)

  weight_decay = trial.suggest_float(
      "weight_decay",
      1e-8,
      1e-3,
      log=True
  )

  # model initialise
  input_dim=len(vocab)
  output_dim=len(vocab)

  model=LSTMnn(input_dim,emb_dim,output_dim,num_lstm_layers,hidden_dim,dropout)
  model.to(device)

  # params init
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  epochs = 10


  # training loop
  from tqdm import tqdm

  for epoch in range(epochs):
      total_loss = 0
      total_acc = 0

      model.train()

      loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

      for input_ids, labels in loop:
          input_ids = input_ids.to(device)
          labels = labels.to(device)

          optimizer.zero_grad()

          outputs = model(input_ids)

          loss = loss_fn(outputs, labels)
          acc = accuracy_fn(outputs, labels)

          loss.backward()
          optimizer.step()

          total_loss += loss.item()
          total_acc += acc.item()

          loop.set_postfix(loss=loss.item(), acc=acc.item())

      print(
          f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Acc: {total_acc/len(train_loader):.4f}"
      )

  # evaluation
  model.eval()
  total_loss = 0
  accuracy = 0
  with torch.no_grad():
      for input_ids, labels in test_loader:
          input_ids = input_ids.to(device)
          labels = labels.to(device)

          outputs = model(input_ids)
          batch_loss = loss_fn(outputs, labels)
          batch_accuracy = accuracy_fn(outputs, labels)
          total_loss += batch_loss.item()
          accuracy += batch_accuracy.item()
      print(
          f"Test loss is {total_loss/len(test_loader):.4f} and accuracy is {accuracy/len(test_loader):.4f}"
      )

  return total_loss/len(test_loader)
