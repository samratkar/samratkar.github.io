#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

python build_qar_dataset.py
python train_fuel_model.py
python train_agent.py --config configs/approach_expanded_fuel_model_golden.json
python evaluate_qar.py
