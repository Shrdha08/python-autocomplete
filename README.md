# Python Code Autocomplete

Line-level code completion for Python: given the code before the cursor, predict the rest of the current line.
Three models of increasing capability are trained and compared on the CodeSearchNet Python dataset
(`Nan-Do/code-search-net-python`).

| # | Model | Idea |
|---|---|---|
| 1 | LSTM encoder-decoder (seq2seq) | RNN baseline trained from scratch |
| 2 | CodeT5-small fine-tuning | pretrained encoder-decoder transformer |
| 3 | Decoder-only transformer (GPT-style) | transformer written from scratch, same data as the LSTM |

The LSTM and the GPT model share the same vocabulary and test set, so comparing them shows the effect of the
**architecture**. CodeT5 shows the effect of **pretraining**.

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_encoding.ipynb` | Lex functions with Python's `tokenize` (comments/docstrings removed, `<NEWLINE>` / `<INDENT>` / `<DEDENT>` markers), split by dataset partition, build a 50k vocab, save encoded data |
| `02_chunking_dataset_preparation.ipynb` | Chunking into (32-token context → rest of line) examples, `Dataset` and `DataLoader` |
| `03_lstm_model.ipynb` | LSTM seq2seq, Optuna hyperparameter search, training with early stopping, test evaluation |
| `04_codet5_data.ipynb` | Raw-text (prefix, rest-of-line) pairs for CodeT5, saved as JSONL to Google Drive |
| `05_codet5_finetune.ipynb` | Fine-tune `Salesforce/codet5-small`; exact match + edit similarity |
| `06_decoder_only_transformer.ipynb` | GPT-style decoder from scratch (causal self-attention, pre-norm blocks, weight tying); greedy generation |

All notebooks are written for Google Colab (GPU runtime). Notebooks 05 and 06 are ready to run; results pending.

## Pipeline

```
CodeSearchNet → lexing → train / valid / test split
   ├── vocab + encoding → 32-token chunks → LSTM seq2seq (03) / GPT decoder (06)
   └── raw-text prefix/target pairs (04) → CodeT5 fine-tuning (05)
```

## Results

To be filled in after training (see the comparison table at the end of notebook 06).
