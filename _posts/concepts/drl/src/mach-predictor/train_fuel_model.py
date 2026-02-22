import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEFAULT_DATA_ROOT = Path("/Users/samrat.kar/cio/airlines-data/glo/737-800/baseline")
DEFAULT_MODEL_PATH = Path("models/fuel_model.pt")
DEFAULT_SCALER_PATH = Path("models/fuel_model_scaler.json")
DEFAULT_MAX_ROWS = 300000
DEFAULT_SAMPLE_ROWS = 120000
DEFAULT_EPOCHS = 30

REQUIRED_COLS = [
    "selectedFuelFlow1",
    "selectedFuelFlow2",
    "altitude",
    "grossWeight",
    "totalAirTemperatureCelsius",
    "Airspeed",
    "groundAirSpeed",
    "mach",
    "angleOfAttackVoted",
    "horizontalStabilizerPosition",
    "totalFuelWeight",
    "trackAngleTrue",
    "fmcMach",
    "latitude",
    "longitude",
    "GMTHours",
    "Day",
    "Month",
    "YEAR",
]


def load_qar_rows(input_csv, data_root, max_rows, sample_rows, seed):
    if input_csv:
        qar = pd.read_csv(input_csv)
        if not set(REQUIRED_COLS).issubset(qar.columns):
            missing = sorted(set(REQUIRED_COLS) - set(qar.columns))
            raise RuntimeError(f"Input CSV missing required columns: {missing}")
        qar = qar[REQUIRED_COLS].dropna()
    else:
        rows = []
        row_count = 0
        for f in Path(data_root).rglob("*.csv"):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if not set(REQUIRED_COLS).issubset(df.columns):
                continue
            df = df[REQUIRED_COLS].dropna()
            rows.append(df)
            row_count += len(df)
            if row_count >= max_rows:
                break

        if not rows:
            raise RuntimeError("No usable QAR rows found.")
        qar = pd.concat(rows, ignore_index=True)

    # Cruise-ish filter to reduce variance in regimes
    qar = qar[(qar["groundAirSpeed"] > 100) & (qar["altitude"] > 30000)]
    if len(qar) > sample_rows:
        qar = qar.sample(sample_rows, random_state=seed)
    return qar


def main():
<<<<<<< Updated upstream
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        print("Fuel model already exists, skipping training.")
        return

    qar = load_qar_rows()
=======
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=None, help="Single QAR CSV input")
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT), help="QAR folder root")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Output model path")
    parser.add_argument(
        "--scaler_path", type=str, default=str(DEFAULT_SCALER_PATH), help="Output scaler json path"
    )
    parser.add_argument("--max_rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--sample_rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    qar = load_qar_rows(args.input_csv, args.data_root, args.max_rows, args.sample_rows, args.seed)
>>>>>>> Stashed changes

    # Features and target
    X = qar[
        [
            "altitude",
            "grossWeight",
            "totalAirTemperatureCelsius",
            "Airspeed",
            "groundAirSpeed",
            "mach",
            "angleOfAttackVoted",
            "horizontalStabilizerPosition",
            "totalFuelWeight",
            "trackAngleTrue",
            "fmcMach",
            "latitude",
            "longitude",
            "GMTHours",
            "Day",
            "Month",
            "YEAR",
        ]
    ].to_numpy(dtype=np.float32)
    fuel_flow = (
        qar["selectedFuelFlow1"].to_numpy(dtype=np.float32)
        + qar["selectedFuelFlow2"].to_numpy(dtype=np.float32)
    )
    fpn = fuel_flow / np.maximum(qar["groundAirSpeed"].to_numpy(dtype=np.float32), 100.0)
    y = fpn.reshape(-1, 1)

    # Train/test split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    train_size = int(0.8 * len(X))
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Standardize
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-6
    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0) + 1e-6
    y_train_n = (y_train - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    Xtr = torch.FloatTensor(X_train)
    ytr = torch.FloatTensor(y_train_n)
    Xte = torch.FloatTensor(X_test)

    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(args.epochs):
        pred = model(Xtr)
        loss = loss_fn(pred, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy().reshape(-1)

    true = y_test_n.reshape(-1)
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    # Back-transform metrics to fuel per NM space
    y_mean_s = float(y_mean.reshape(-1)[0])
    y_std_s = float(y_std.reshape(-1)[0])
    pred_fpn = pred * y_std_s + y_mean_s
    true_fpn = true * y_std_s + y_mean_s
    mae_fpn = float(np.mean(np.abs(pred_fpn - true_fpn)))
    rmse_fpn = float(np.sqrt(np.mean((pred_fpn - true_fpn) ** 2)))

    model_path = Path(args.model_path)
    scaler_path = Path(args.scaler_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    scaler_path.write_text(
        json.dumps(
            {
                "mean": x_mean.tolist(),
                "std": x_std.tolist(),
                "target": "fpn",
                "y_mean": y_mean_s,
                "y_std": y_std_s,
            }
        ),
        encoding="utf-8",
    )

    print("ROWS", len(qar))
    print("MAE_NORM", mae)
    print("RMSE_NORM", rmse)
    print("R2", r2)
    print("MAE_FPN", mae_fpn)
    print("RMSE_FPN", rmse_fpn)
    print("MODEL_PATH", model_path)
    print("SCALER_PATH", scaler_path)


if __name__ == "__main__":
    main()
