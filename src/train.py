"""Training, evaluation and Optuna tuning for the seq2seq code-completion model.

Mirrors notebook 03. The Optuna objective previously lived in
`src/models/lstm_model.py`, where it had no access to the dataloaders,
criterion or device it referenced; it belongs here instead.

Usage:
    python -m src.train tune --trials 10
    python -m src.train train --epochs 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from src.data.preprocess import get_dataloaders
from src.models.lstm_model import build_model

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
BEST_MODEL_PATH = ARTIFACT_DIR / "best_lstm_code_completion.pt"
BEST_PARAMS_PATH = ARTIFACT_DIR / "best_params.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_metrics(outputs, decoder_outputs, criterion, pad_idx):
    """Loss and masked token accuracy, both ignoring timestep 0 and padding."""
    pred = outputs[:, 1:, :].reshape(-1, outputs.shape[-1])
    target = decoder_outputs[:, 1:].reshape(-1)
    loss = criterion(pred, target)

    predictions = outputs[:, 1:, :].argmax(dim=2)
    target_tokens = decoder_outputs[:, 1:]
    mask = target_tokens != pad_idx

    correct = ((predictions == target_tokens) & mask).sum()
    total = mask.sum()
    return loss, correct, total


def train_one_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio, pad_idx):
    model.train()
    epoch_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for encoder_inputs, decoder_inputs, decoder_outputs in loader:
        encoder_inputs = encoder_inputs.to(device)
        decoder_inputs = decoder_inputs.to(device)
        decoder_outputs = decoder_outputs.to(device)

        optimizer.zero_grad()
        outputs = model(encoder_inputs, decoder_inputs, teacher_forcing_ratio)
        loss, correct, total = batch_metrics(outputs, decoder_outputs, criterion, pad_idx)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        total_correct += correct.item()
        total_tokens += total.item()

    accuracy = total_correct / total_tokens if total_tokens else 0.0
    return epoch_loss / len(loader), accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, pad_idx):
    """Free-running evaluation - no teacher forcing."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for encoder_inputs, decoder_inputs, decoder_outputs in loader:
        encoder_inputs = encoder_inputs.to(device)
        decoder_inputs = decoder_inputs.to(device)
        decoder_outputs = decoder_outputs.to(device)

        outputs = model(encoder_inputs, decoder_inputs, teacher_forcing_ratio=0.0)
        loss, correct, total = batch_metrics(outputs, decoder_outputs, criterion, pad_idx)

        total_loss += loss.item()
        total_correct += correct.item()
        total_tokens += total.item()

    accuracy = total_correct / total_tokens if total_tokens else 0.0
    return total_loss / len(loader), accuracy


def tune(trials: int = 10, epochs_per_trial: int = 2, subset: int = 5000):
    """Optuna search over the seq2seq hyperparameters, minimising val loss."""
    import optuna
    from torch.utils.data import DataLoader, Subset

    train_loader, valid_loader, _, vocab = get_dataloaders()
    pad_idx = vocab["<PAD>"]
    vocab_size = len(vocab)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Search on a subset so the study stays affordable.
    collate_fn = train_loader.collate_fn
    small_train = DataLoader(
        Subset(train_loader.dataset, range(min(subset, len(train_loader.dataset)))),
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
    )
    small_valid = DataLoader(
        Subset(valid_loader.dataset, range(min(subset // 5, len(valid_loader.dataset)))),
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
    )

    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 384, 512])
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        teacher_forcing_ratio = trial.suggest_float("teacher_forcing_ratio", 0.3, 0.7)

        model = build_model(vocab_size, hidden_dim, n_layers, dropout).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        val_loss = float("inf")
        for epoch in range(epochs_per_trial):
            train_one_epoch(
                model, small_train, optimizer, criterion, DEVICE, teacher_forcing_ratio, pad_idx
            )
            val_loss, _ = evaluate(model, small_valid, criterion, DEVICE, pad_idx)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_loss

    study = optuna.create_study(
        study_name="lstm_code_completion",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(objective, n_trials=trials)

    print(f"\nBest validation loss: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(study.best_params, indent=2))
    print(f"Saved to {BEST_PARAMS_PATH}")
    return study.best_params


def train(epochs: int = 20, patience: int = 4, params: dict | None = None):
    """Train on the full training set, keeping the best checkpoint by val loss."""
    if params is None:
        if not BEST_PARAMS_PATH.exists():
            raise FileNotFoundError(
                f"{BEST_PARAMS_PATH} not found. Run 'python -m src.train tune' first."
            )
        params = json.loads(BEST_PARAMS_PATH.read_text())

    train_loader, valid_loader, test_loader, vocab = get_dataloaders()
    pad_idx = vocab["<PAD>"]

    model = build_model(
        len(vocab), params["hidden_dim"], params["n_layers"], params["dropout"]
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            params["teacher_forcing_ratio"],
            pad_idx,
        )
        val_loss, val_acc = evaluate(model, valid_loader, criterion, DEVICE, pad_idx)
        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] "
            f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
            f"| Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}% "
            f"| LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_params": params,
                },
                BEST_MODEL_PATH,
            )
            print("   * best model saved")
        else:
            epochs_without_improvement += 1
            print(f"   no improvement ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE, pad_idx)

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    return test_loss, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_tune = sub.add_parser("tune", help="run the Optuna search")
    p_tune.add_argument("--trials", type=int, default=10)
    p_tune.add_argument("--epochs-per-trial", type=int, default=2)

    p_train = sub.add_parser("train", help="train with the tuned hyperparameters")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--patience", type=int, default=4)

    args = parser.parse_args()
    print(f"Device: {DEVICE}")

    if args.command == "tune":
        tune(trials=args.trials, epochs_per_trial=args.epochs_per_trial)
    else:
        train(epochs=args.epochs, patience=args.patience)
