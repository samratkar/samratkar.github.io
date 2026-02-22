# Synthetic DRL Model Card (`models/syn`)

Run date: February 22, 2026

## 1) Source Data

### Primary synthetic training dataset
- File: `data/qar_737800_synthetic.csv`
- Rows generated: `120000`
- Generator: `generate_synthetic_qar.py`
- Generation mode: bootstrap from field cruise distribution + bounded perturbations
- Seed: `42`

### Field data lineage used to shape synthetic distribution
- Reference file: `data/qar_737800_cruise.csv`
- Reference rows: `120000`
- This field dataset defines the baseline cruise-state distributions (altitude, weight, Mach, temperature, wind, controls, position/time features).

### Synthetic QAR columns used by training/evaluation
- Raw fuel/speed columns:
  - `selectedFuelFlow1`, `selectedFuelFlow2`, `groundAirSpeed`, `Airspeed`
- State/context columns:
  - `altitude`, `grossWeight`, `totalAirTemperatureCelsius`, `mach`
  - `angleOfAttackVoted`, `horizontalStabilizerPosition`, `totalFuelWeight`, `trackAngleTrue`
  - `fmcMach`, `latitude`, `longitude`, `GMTHours`, `Day`, `Month`, `YEAR`
  - plus env-friendly columns: `CAS`, `tempDevC`, `windKts`, `phase`, `targetAltitude`, `turbulence`, `regime`

## 2) Model Parameters

### Config parameter glossary (`configs/approach_expanded_fuel_model_syn.json`)
- `name`: Identifier for the experiment/config preset.
  - Current value: `expanded_feature_fuel_model_synthetic_qar`
- `description`: Human-readable text describing the run objective.
  - Current value: `Expanded-feature fuel-per-NM model + PPO trained on synthetic QAR`
- `learning_rate`: Optimizer step size for PPO updates (how aggressively weights change each step).
  - Current value: `0.0002`
- `gamma`: Discount factor for future rewards in return/advantage computation.
  - Current value: `0.99`
- `eps_clip`: PPO trust-region style clipping range for policy ratio.
  - Current value: `0.2`
- `k_epochs`: Number of optimization passes over each collected PPO batch.
  - Current value: `4`
- `batch_size`: Number of rollout steps collected before a PPO update cycle.
  - Current value: `128`
- `total_timesteps`: Total environment interaction steps for PPO training.
  - Current value: `30000`
- `hidden_size`: Width of hidden layers in both actor and critic networks.
  - Current value: `64`
- `pretrain_steps`: Number of behavior-cloning warmup iterations using oracle Mach targets.
  - Current value: `300`
- `pretrain_batch`: Batch size per pretraining iteration.
  - Current value: `128`
- `reward_shaping_strength`: Weight of oracle-deviation penalty in reward; higher means stronger pull toward oracle behavior.
  - Current value: `1.0`
- `curriculum_steps`: Step index at which curriculum would switch to mixed phases.
  - Current value: `50000`
  - Note: with `phase_mode = cruise_only` and `total_timesteps = 30000`, this threshold is not reached.
- `qar_data_path`: CSV used to sample realistic initial states in the environment.
  - Current value: `data/qar_737800_synthetic.csv`
- `phase_mode`: Training regime selector (`cruise_only` vs mixed phases).
  - Current value: `cruise_only`
- `fuel_flow_scale`: Linear multiplier on physics-based fuel flow path.
  - Current value: `1.0`
- `fuel_flow_bias`: Linear additive offset on physics-based fuel flow path.
  - Current value: `0.0`
- `fuel_fpn_quad`: Optional quadratic calibration `[a,b,c]` for fuel-per-NM; `null` disables this path.
  - Current value: `null`
- `fuel_model_path`: Path to trained supervised fuel model used by the env reward model.
  - Current value: `models/syn/fuel_model.pt`
- `fuel_model_scaler_path`: Path to scaler stats (`mean/std` plus target normalization) for fuel-model inference.
  - Current value: `models/syn/fuel_model_scaler.json`
- `output_model_path`: Save path for trained PPO policy weights.
  - Current value: `models/syn/tail_policy.pt`
- `seed`: Random seed for reproducible numpy/torch training behavior.
  - Current value: `42`

### Golden vs syn config differences
- `total_timesteps`: golden `250000` vs syn `30000`
- `pretrain_steps`: golden `2000` vs syn `300`
- `pretrain_batch`: golden `256` vs syn `128`
- `reward_shaping_strength`: golden `0.5` vs syn `1.0`
- `curriculum_steps`: golden `80000` vs syn `50000`
- `qar_data_path`: golden field CSV vs syn synthetic CSV
- `fuel_model_path`: golden `models/fuel_model.pt` vs syn `models/syn/fuel_model.pt`
- `fuel_model_scaler_path`: golden `models/fuel_model_scaler.json` vs syn `models/syn/fuel_model_scaler.json`
- Syn config explicitly sets `output_model_path` and `seed`; golden config does not.

### Fuel model
- Artifact: `models/syn/fuel_model.pt`
- Scaler/normalization: `models/syn/fuel_model_scaler.json`
- Script: `train_fuel_model.py`
- Architecture: MLP `17 -> 64 -> 64 -> 1` with `ReLU`
- Target: fuel per NM (`fpn = (FF1 + FF2) / groundAirSpeed`)
- Train args used:
  - `--input_csv data/qar_737800_synthetic.csv`
  - `--sample_rows 120000`
  - `--epochs 30`

### DRL policy model (PPO actor-critic)
- Artifact: `models/syn/tail_policy.pt`
- Script: `train_agent.py`
- Config: `configs/approach_expanded_fuel_model_syn.json`
- Seed: `42`
- Policy network:
  - Actor: `21 -> 64 -> 64 -> 1` with `Tanh` layers and Gaussian action head
  - Critic: `21 -> 64 -> 64 -> 1` with `Tanh` layers
- Action range mapping: normalized `[-1, 1]` -> Mach `[0.70, 0.86]`

### PPO training hyperparameters (from config)
- `learning_rate`: `0.0002`
- `gamma`: `0.99`
- `eps_clip`: `0.2`
- `k_epochs`: `4`
- `batch_size`: `128`
- `total_timesteps`: `30000`
- `pretrain_steps`: `300`
- `pretrain_batch`: `128`
- `reward_shaping_strength`: `1.0`
- `curriculum_steps`: `50000`
- `phase_mode`: `cruise_only`

## 3) Model Performance

### Fuel model fit (held-out split from training run)
- Rows used after filtering: `119890`
- `MAE_NORM`: `0.511628270149231`
- `RMSE_NORM`: `0.6882007122039795`
- `R2`: `0.5229887504469024`
- `MAE_FPN`: `0.21794001758098602`
- `RMSE_FPN`: `0.2931551933288574`

### End-to-end policy performance (QAR-aligned eval)
- Evaluation script: `evaluate_qar.py`
- Eval set: `20000` sampled rows from `data/qar_737800_synthetic.csv`
- Baseline fuel per NM: `12.25299747494486`
- Policy fuel per NM: `11.94069350943155`
- Improvement (fuel savings): `2.5487964569642076%`
- Machine-readable output: `models/syn/results.json`

## 4) Reproducibility Commands

```bash
python generate_synthetic_qar.py --field_csv data/qar_737800_cruise.csv --output_csv data/qar_737800_synthetic.csv
python train_fuel_model.py --input_csv data/qar_737800_synthetic.csv --model_path models/syn/fuel_model.pt --scaler_path models/syn/fuel_model_scaler.json --sample_rows 120000 --epochs 30
python train_agent.py --config configs/approach_expanded_fuel_model_syn.json
python evaluate_qar.py --input_csv data/qar_737800_synthetic.csv --policy_path models/syn/tail_policy.pt --fuel_model_path models/syn/fuel_model.pt --fuel_scaler_path models/syn/fuel_model_scaler.json --sample_rows 20000 --results_json models/syn/results.json
```
