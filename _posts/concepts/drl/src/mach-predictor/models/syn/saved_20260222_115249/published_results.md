# Synthetic QAR Training Results

Run date: February 22, 2026

## Inputs
- Synthetic QAR CSV: `data/qar_737800_synthetic.csv`
- Config: `configs/approach_expanded_fuel_model_syn.json`
- Fuel model output: `models/syn/fuel_model.pt`
- Policy output: `models/syn/tail_policy.pt`

## Evaluation (QAR-aligned)
- Rows: `20000`
- Baseline fuel per NM: `12.25299747494486`
- Policy fuel per NM: `11.94069350943155`
- Improvement: `2.5487964569642076%`

## Artifact
- Machine-readable metrics: `models/syn/results.json`
