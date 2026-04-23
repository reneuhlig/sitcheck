from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TorchTrainConfig:
    """Training hyperparameters for Torch forecaster."""

    rnn_type: str = "lstm"
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    patience: int = 10
    seed: int = 42


def set_seed_torch(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TorchForecaster(nn.Module):
    """LSTM/GRU forecaster with multi-horizon output."""

    def __init__(
        self,
        input_size: int,
        horizon: int,
        rnn_type: str = "lstm",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be one of ['lstm', 'gru']")

        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


def train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TorchTrainConfig,
    device: torch.device,
) -> tuple[TorchForecaster, list[dict[str, float]], int, float]:
    """Train Torch model with early stopping on validation loss."""
    set_seed_torch(config.seed)

    model = TorchForecaster(
        input_size=X_train.shape[-1],
        horizon=y_train.shape[-1],
        rnn_type=config.rnn_type,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.float32)),
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    wait = 0

    for epoch in range(config.epochs):
        model.train()
        train_losses: list[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_epoch, best_val


def predict_torch(
    model: TorchForecaster,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Run inference in batches and return numpy predictions."""
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append(out.detach().cpu().numpy())

    if not preds:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(preds, axis=0).astype(np.float32)
