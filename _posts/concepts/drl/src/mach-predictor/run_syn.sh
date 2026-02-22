#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

python generate_synthetic_qar.py --field_csv data/qar_737800_cruise.csv --output_csv data/qar_737800_synthetic.csv
python train_fuel_model.py --input_csv data/qar_737800_synthetic.csv --model_path models/syn/fuel_model.pt --scaler_path models/syn/fuel_model_scaler.json --sample_rows 120000
python train_agent.py --config configs/approach_expanded_fuel_model_syn.json
python evaluate_qar.py --input_csv data/qar_737800_synthetic.csv --policy_path models/syn/tail_policy.pt --fuel_model_path models/syn/fuel_model.pt --fuel_scaler_path models/syn/fuel_model_scaler.json --sample_rows 20000 --results_json models/syn/results.json
