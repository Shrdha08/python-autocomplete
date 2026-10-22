"""Seq2seq LSTM for Python line completion.

The encoder compresses a fixed context window into its final hidden and cell
states; the decoder starts from those states and generates the completion one
token at a time, optionally fed the ground-truth previous token (teacher
forcing).
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            n_layers,
            # nn.LSTM ignores dropout with a single layer and warns; pass 0 there.
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, src: torch.Tensor):
        _, (hidden, cell) = self.lstm(self.embedding(src))
        return hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        # (batch,) -> (batch, 1) so the LSTM sees a single timestep.
        emb = self.embedding(input_token.unsqueeze(1))
        output, (hidden, cell) = self.lstm(emb, (hidden, cell))
        return self.linear(output.squeeze(1)), hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: LSTMEncoder, decoder: LSTMDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src: torch.Tensor,
        target: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size, target_len = target.shape
        vocab_size = self.decoder.linear.out_features

        # Position 0 is the <BOS> the decoder is primed with, so it stays zero
        # and is excluded from loss and accuracy downstream.
        outputs = torch.zeros(batch_size, target_len, vocab_size, device=src.device)

        hidden, cell = self.encoder(src)
        input_token = target[:, 0]

        for t in range(1, target_len):
            pred, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = pred

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = target[:, t] if teacher_force else pred.argmax(dim=1)

        return outputs

    @torch.no_grad()
    def complete(
        self,
        context: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_length: int = 50,
    ) -> list[int]:
        """Greedily generate a completion for a single encoded context.

        `context` is a 1-D tensor of token ids. Returns the generated ids,
        stopping at `eos_idx`.
        """
        self.eval()
        hidden, cell = self.encoder(context.unsqueeze(0))

        input_token = torch.tensor([bos_idx], device=context.device)
        generated: list[int] = []

        for _ in range(max_length):
            pred, hidden, cell = self.decoder(input_token, hidden, cell)
            next_token = int(pred.argmax(dim=1).item())
            if next_token == eos_idx:
                break
            generated.append(next_token)
            input_token = torch.tensor([next_token], device=context.device)

        return generated


def build_model(
    vocab_size: int, hidden_dim: int, n_layers: int, dropout: float
) -> Seq2Seq:
    return Seq2Seq(
        LSTMEncoder(vocab_size, hidden_dim, n_layers, dropout),
        LSTMDecoder(vocab_size, hidden_dim, n_layers, dropout),
    )
