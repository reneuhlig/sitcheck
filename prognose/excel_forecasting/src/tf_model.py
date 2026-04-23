from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass
class TfTrainConfig:
    """Training hyperparameters for TensorFlow forecaster."""

    rnn_type: str = "lstm"
    units: int = 64
    dropout: float = 0.1
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    patience: int = 10
    seed: int = 42


def set_seed_tf(seed: int) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_tf_model(
    input_shape: tuple[int, int],
    horizon: int,
    config: TfTrainConfig,
) -> tf.keras.Model:
    """Build a Keras LSTM/GRU forecaster."""
    rnn_type = config.rnn_type.lower()
    if rnn_type not in {"lstm", "gru"}:
        raise ValueError("rnn_type must be one of ['lstm', 'gru']")

    inputs = tf.keras.Input(shape=input_shape)
    if rnn_type == "lstm":
        x = tf.keras.layers.LSTM(config.units, return_sequences=False)(inputs)
    else:
        x = tf.keras.layers.GRU(config.units, return_sequences=False)(inputs)

    x = tf.keras.layers.Dropout(config.dropout)(x)
    outputs = tf.keras.layers.Dense(horizon)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr),
        loss="mse",
    )
    return model


def train_tf_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TfTrainConfig,
    checkpoint_path: str,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train TensorFlow model with early stopping and model checkpoint."""
    set_seed_tf(config.seed)

    model = build_tf_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        horizon=y_train.shape[1],
        config=config,
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)

    return model, history
