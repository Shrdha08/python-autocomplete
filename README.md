# Python Code Autocomplete

## Progress

### Completed Preprocessing Pipeline

- Loaded the CodeSearchNet Python dataset using Hugging Face Datasets.
- Extracted tokenized code from the `code_tokens` field.
- Split data into train, validation, and test sets (80/10/10).
- Built a vocabulary using only the training data.
- Added special tokens: `<pad>` and `<unk>`.
- Encoded tokens into integer IDs.
- Generated input-target pairs using a sliding window (`seq_len = 10`).
- Implemented a custom PyTorch `Dataset` class.
- Created train, validation, and test `DataLoader`s through `get_dataloaders()`.

## Current Pipeline

CodeSearchNet Dataset  
↓  
Train / Validation / Test Split  
↓  
Vocabulary Creation  
↓  
Token Encoding  
↓  
Sequence Generation  
↓  
PyTorch Dataset  
↓  
DataLoader

## Next Steps

- Implement LSTM model
- Train model
- Evaluate model performance
- Build autocomplete inference module
- Generate top-k code completion predictions