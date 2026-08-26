"""Data pipeline for the Python code-completion model.

Mirrors notebooks 01 and 02: tokenize CodeSearchNet Python source, strip
comments and docstrings, build a vocabulary from the training split only,
encode to integer IDs, and window each function into
(context -> rest-of-line) pairs for the seq2seq model.

Nothing runs at import time; call `build_and_cache()` once to produce the
artifacts, then `get_dataloaders()` to consume them.
"""
from __future__ import annotations

import io
import pickle
import tokenize
from collections import Counter
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

DATASET_NAME = "Nan-Do/code-search-net-python"
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"

SEQ_LENGTH = 32
STRIDE = 16
MAX_OUTPUT_LENGTH = 50
MAX_VOCAB_SIZE = 50_000
MIN_TOKEN_FREQ = 2
BATCH_SIZE = 64
RANDOM_STATE = 42

# `<NEWLINE>`, `<INDENT>` and `<DEDENT>` are emitted by tokenize_code() as
# literal tokens. They are registered explicitly rather than being left to the
# frequency cutoff, because the dataset windowing depends on `<NEWLINE>` being
# present in the vocabulary.
SPECIAL_TOKENS = (
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<NEWLINE>",
    "<INDENT>",
    "<DEDENT>",
)


def tokenize_code(source: str) -> list[str] | None:
    """Tokenize Python source, dropping comments and docstrings.

    Returns None for sources that cannot be tokenized.
    """
    tokens: list[str] = []
    prev_toktype = tokenize.INDENT

    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            tok_type, tok_str = tok.type, tok.string

            if tok_type == tokenize.COMMENT:
                continue

            # A string in statement position is a docstring, not a value.
            if tok_type == tokenize.STRING and prev_toktype in (
                tokenize.INDENT,
                tokenize.NEWLINE,
                tokenize.DEDENT,
            ):
                prev_toktype = tok_type
                continue

            if tok_type == tokenize.INDENT:
                tokens.append("<INDENT>")
            elif tok_type == tokenize.DEDENT:
                tokens.append("<DEDENT>")
            elif tok_type in (tokenize.NEWLINE, tokenize.NL):
                tokens.append("<NEWLINE>")
            else:
                tokens.append(tok_str)

            prev_toktype = tok_type

        return tokens

    except (IndentationError, tokenize.TokenError, TabError, SyntaxError):
        return None


def load_code_tokens(limit: int | None = None) -> list[list[str]]:
    """Download CodeSearchNet Python and tokenize every function."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME)
    sources = ds["train"]["code"]
    if limit is not None:
        sources = sources[:limit]

    code_tokens = []
    for source in sources:
        tokens = tokenize_code(source)
        if tokens:
            code_tokens.append(tokens)
    return code_tokens


def split_data(
    code_tokens: list[list[str]],
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Split 80/10/10 into train, validation, test."""
    from sklearn.model_selection import train_test_split

    train_data, temp_data = train_test_split(
        code_tokens, test_size=0.2, random_state=RANDOM_STATE
    )
    valid_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=RANDOM_STATE
    )
    return train_data, valid_data, test_data


def build_vocab(train_data: list[list[str]]) -> dict[str, int]:
    """Build the vocabulary from the training split only, to avoid leakage."""
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}

    counter: Counter[str] = Counter()
    for token_list in train_data:
        counter.update(token_list)

    for token, freq in counter.most_common(MAX_VOCAB_SIZE):
        if freq >= MIN_TOKEN_FREQ and token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def encode(token_list: list[str], vocab: dict[str, int]) -> list[int]:
    unk = vocab["<UNK>"]
    return [vocab.get(token, unk) for token in token_list]


class CodeCompletionDataset(Dataset):
    """Windows encoded functions into (context -> rest-of-line) examples.

    For each split point, the encoder sees the preceding `seq_length` tokens and
    the decoder is supervised on the tokens up to the next `<NEWLINE>` (capped at
    `max_output_length`), so the model learns to finish one logical line.
    """

    def __init__(
        self,
        data: list[list[int]],
        vocab: dict[str, int],
        seq_length: int = SEQ_LENGTH,
        max_output_length: int = MAX_OUTPUT_LENGTH,
        stride: int = STRIDE,
    ):
        self.data = data
        self.vocab = vocab
        self.seq_length = seq_length
        self.max_output_length = max_output_length

        self.bos = vocab["<BOS>"]
        self.eos = vocab["<EOS>"]
        self.newline = vocab["<NEWLINE>"]

        # (row index, split position) for every window in the corpus.
        self.indices: list[tuple[int, int]] = []
        for row_idx, row in enumerate(data):
            for i in range(seq_length, len(row), stride):
                # Skip split points with no target tokens left.
                if i < len(row):
                    self.indices.append((row_idx, i))

    def prep_data(self, row: list[int], i: int):
        encoder_input = [self.bos] + row[i - self.seq_length : i] + [self.eos]

        # Target runs to the next newline, or the cap, whichever comes first.
        nearest_line_end = len(row)
        for j in range(i, len(row)):
            if row[j] == self.newline:
                nearest_line_end = j
                break

        end = min(i + self.max_output_length, nearest_line_end + 1)
        labels = row[i:end]

        decoder_input = [self.bos] + labels
        decoder_output = labels + [self.eos]

        return (
            torch.tensor(encoder_input, dtype=torch.long),
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(decoder_output, dtype=torch.long),
        )

    def __getitem__(self, idx: int):
        row_idx, i = self.indices[idx]
        return self.prep_data(self.data[row_idx], i)

    def __len__(self) -> int:
        return len(self.indices)


def make_collate_fn(pad_idx: int):
    """Pad a batch; targets vary in length because lines do."""

    def collate_fn(batch):
        encoder_inputs, decoder_inputs, decoder_outputs = zip(*batch)
        return (
            pad_sequence(encoder_inputs, batch_first=True, padding_value=pad_idx),
            pad_sequence(decoder_inputs, batch_first=True, padding_value=pad_idx),
            pad_sequence(decoder_outputs, batch_first=True, padding_value=pad_idx),
        )

    return collate_fn


def build_and_cache(limit: int | None = None, artifact_dir: Path = ARTIFACT_DIR) -> None:
    """Run the full pipeline once and cache vocab plus encoded splits."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("Tokenizing corpus...")
    code_tokens = load_code_tokens(limit=limit)
    print(f"  {len(code_tokens)} functions tokenized")

    train_data, valid_data, test_data = split_data(code_tokens)
    print(f"  split: {len(train_data)} train / {len(valid_data)} valid / {len(test_data)} test")

    vocab = build_vocab(train_data)
    print(f"  vocabulary: {len(vocab)} tokens")

    splits = {
        "encoded_train": train_data,
        "encoded_valid": valid_data,
        "encoded_test": test_data,
    }
    for name, split in splits.items():
        encoded = [encode(token_list, vocab) for token_list in split]
        with open(artifact_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(encoded, f)

    with open(artifact_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print(f"Artifacts written to {artifact_dir}")


def load_artifacts(artifact_dir: Path = ARTIFACT_DIR):
    def read(name):
        path = artifact_dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python -m src.data.preprocess' first."
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    return (
        read("encoded_train"),
        read("encoded_valid"),
        read("encoded_test"),
        read("vocab"),
    )


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    artifact_dir: Path = ARTIFACT_DIR,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Return train, validation and test loaders plus the vocabulary."""
    encoded_train, encoded_valid, encoded_test, vocab = load_artifacts(artifact_dir)

    train_dataset = CodeCompletionDataset(encoded_train, vocab)
    valid_dataset = CodeCompletionDataset(encoded_valid, vocab)
    test_dataset = CodeCompletionDataset(encoded_test, vocab)

    collate_fn = make_collate_fn(vocab["<PAD>"])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, valid_loader, test_loader, vocab


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="tokenize only the first N functions"
    )
    args = parser.parse_args()
    build_and_cache(limit=args.limit)
