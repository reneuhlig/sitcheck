# Excel Multi-Horizon Forecasting (PyTorch + TensorFlow)

Dieses Teilprojekt trainiert und bewertet Auslastungsprognosen aus Excel-Zeitreihen mit:
- PyTorch LSTM/GRU
- TensorFlow/Keras LSTM/GRU
- Baselines (naive, seasonal-naive falls Frequenz geeignet)

Das Projekt liegt bewusst isoliert unter `excel_forecasting/`.

## Projektstruktur

```text
excel_forecasting/
  requirements.txt
  README.md
  src/
    data.py
    windows.py
    metrics.py
    baselines.py
    torch_model.py
    tf_model.py
  scripts/
    train_torch.py
    train_tf.py
    evaluate.py
  artifacts/
```

## Setup

Python 3.11+ empfohlen.

```bash
cd excel_forecasting
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datenanforderungen

Pflichtparameter:
- `--file`: Pfad zur Excel-Datei
- `--date_col`: Zeitstempel-Spalte
- `--target_col`: numerische Zielspalte (z. B. `Auslastung`)

Optional:
- `--sheet` (default `0`)
- `--freq` (Resampling, z. B. `15min`, `H`, `D`)
- `--features` (kommagetrennte Feature-Spalten; sonst alle numerischen Spalten + Kalenderfeatures)

Pipeline-Handling:
- Sortierung nach Datum
- Deduplikation nach Zeitstempel
- NaN-Behandlung (Interpolation + ffill/bfill)
- Kein Data Leakage: Scaler werden nur auf Train fitten
- Multi-Horizon-Windows: `y.shape = (samples, horizon)`

## Training: PyTorch

```bash
python scripts/train_torch.py \
  --file /pfad/zu/deiner_datei.xlsx \
  --date_col timestamp \
  --target_col Auslastung \
  --lookback 48 \
  --horizon 12 \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.001
```

GRU statt LSTM:

```bash
python scripts/train_torch.py ... --rnn_type gru
```

## Training: TensorFlow/Keras

```bash
python scripts/train_tf.py \
  --file /pfad/zu/deiner_datei.xlsx \
  --date_col timestamp \
  --target_col Auslastung \
  --lookback 48 \
  --horizon 12 \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.001
```

GRU statt LSTM:

```bash
python scripts/train_tf.py ... --rnn_type gru
```

## Evaluation (rein artefaktbasiert)

`evaluate.py` nutzt nur gespeicherte Artefakte und berechnet keine Trainingsläufe neu.

```bash
python scripts/evaluate.py --artifacts_dir artifacts
```

Outputs:
- `artifacts/metrics.json`
- `artifacts/predictions.csv`
- `artifacts/plots/horizon_example.png`
- `artifacts/plots/rolling_step1.png`

## Artefakte

### Common (`artifacts/common/`)
- `y_test.npy`
- `timestamps_test.csv`
- `config.json`
- `target_scaler.pkl`

### PyTorch (`artifacts/torch/`)
- `model_best.pt`
- `x_scaler.pkl`, `y_scaler.pkl`
- `pred_test.npy`
- `train_history.csv`
- `meta.json`

### TensorFlow (`artifacts/tf/`)
- `model_best.keras`
- `x_scaler.pkl`, `y_scaler.pkl`
- `pred_test.npy`
- `history.csv`
- `meta.json`

### Baselines (`artifacts/baselines/`)
- `pred_naive.npy`
- `pred_seasonal_naive.npy` (nur wenn verfügbar)
- `meta.json`

## Reproduzierbarkeit

Seeds werden in beiden Trainingspipelines gesetzt:
- `random`, `numpy`
- `torch` inkl. cudnn deterministic
- `tensorflow`

CPU-first. Wenn GPU verfügbar ist, wird sie vom jeweiligen Framework genutzt.

## Troubleshooting

1. Datei nicht gefunden
- Fehler zeigt absoluten Pfad und einen Beispiel-Hinweispfad.

2. Spaltenname falsch
- Fehler listet verfügbare Spalten aus der Excel-Datei.

3. Zu wenige Daten für `lookback + horizon`
- Reduziere `lookback`/`horizon` oder nutze mehr Daten.

4. Seasonal-naive fehlt
- Bei ungeeigneter Frequenz/zu kurzem Lookback wird seasonal-naive deaktiviert; Grund steht in `artifacts/baselines/meta.json`.

5. Keine GPU erkannt
- Erwartetes Verhalten. Training läuft auf CPU.
