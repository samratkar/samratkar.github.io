# Model Snapshot — 2026-02-22 12:07:40

Freshly trained by `run_golden.sh` (full pipeline run, 250,000 PPO timesteps).
Evaluation result: **baseline 12.0146 kg/NM → policy 11.1214 kg/NM (+7.43% improvement)**.

---

## Files

| File | Description |
|---|---|
| `tail_policy.pt` | PPO actor-critic policy weights |
| `fuel_model.pt` | Neural fuel-per-NM model weights |
| `fuel_model_scaler.json` | Feature normalization stats for the fuel model |
| `approach_expanded_fuel_model_golden.json` | Training config used for this run |

---

## 0. Performance Results

### Policy vs Baseline (QAR Cruise Evaluation)

| Metric | Value |
|---|---|
| Evaluation rows | 20,000 (random sample from `qar_737800_cruise.csv`) |
| Baseline fuel/NM (recorded mach) | 12.0146 kg/NM |
| Policy fuel/NM (RL-recommended mach) | 11.1214 kg/NM |
| **Improvement** | **+7.43%** |

Both baseline and policy are evaluated through the same neural fuel model:
- **Baseline**: fuel model queried at the mach actually recorded in QAR
- **Policy**: fuel model queried at the Mach recommended by the RL actor

### Training Loss Progression

| Step | PPO Loss | Notes |
|---|---|---|
| 80,000 | 41.5886 | End of cruise-only curriculum phase |
| 160,000 | 14.9621 | Mixed phases, loss converging |
| 240,000 | 15.7042 | Stable |

### Timing

| Phase | Duration |
|---|---|
| Oracle pretraining (2,000 steps × 256 batch) | Not separately recorded for this run |
| PPO training (250,000 steps) | Not separately recorded for this run |
| **Total wall-clock time** | **~71 minutes** |

> Timing was observed via `ps` CPU time during the run. From the next training run onward,
> `train_agent.py` prints per-step elapsed time and total duration directly in the output.

---

## 1. Source Data (`data/qar_737800_cruise.csv`)

### Provenance

| Property | Value |
|---|---|
| Aircraft type | Boeing 737-800 |
| Source | Real QAR (Quick Access Recorder) fleet data |
| Built by | `build_qar_dataset.py` |
| Filters applied | `altitude > 30,000 ft` AND `groundAirSpeed > 100 kts` |
| Phase | Cruise only (phase = 1 throughout) |

### Shape

| Property | Value |
|---|---|
| Rows | 120,000 |
| Columns | 22 |
| Null values | 0 (none) |
| Random seed (sample) | 42 |

### Column Statistics

| Column | Min | Mean | Median | Max | Std |
|---|---|---|---|---|---|
| altitude (ft) | 30,001 | 36,790 | 36,987 | 40,988 | 2,258 |
| grossWeight | 113,881 | 136,953 | 138,961 | 153,842 | 8,923 |
| totalAirTemperatureCelsius (°C) | -41.00 | -23.62 | -23.75 | 0.25 | 6.16 |
| CAS (kts) | 212.25 | 248.19 | 247.00 | 302.00 | 11.34 |
| mach | 0.598 | 0.7641 | 0.766 | 0.812 | 0.0193 |
| tempDevC (°C) | 15.50 | 31.74 | 32.50 | 45.51 | 4.66 |
| windKts (kts) | 58.00 | 194.56 | 199.50 | 281.75 | 33.09 |
| phase | 1 | 1 | 1 | 1 | 0 |
| targetAltitude (ft) | 30,001 | 36,790 | 36,987 | 40,988 | 2,258 |
| turbulence | 0 | 0 | 0 | 0 | 0 |
| regime | 0 | 0 | 0 | 0 | 0 |
| angleOfAttackVoted (°) | -2.99 | 3.23 | 4.30 | 6.93 | 2.14 |
| horizontalStabilizerPosition (°) | -4.04 | 2.82 | 2.99 | 6.86 | 0.90 |
| totalFuelWeight (kg) | 0 | 10,416 | 11,230 | 25,900 | 6,523 |
| trackAngleTrue (°) | 0 | 115.82 | 62.97 | 365.27 | 107.70 |
| fmcMach | 0.000 | 0.7644 | 0.767 | 3.072 | 0.0400 |
| latitude (°) | -28.95 | -18.02 | -18.10 | -5.29 | 5.21 |
| longitude (°) | -62.28 | -44.87 | -44.67 | -35.11 | 4.93 |
| GMTHours (h) | 0 | 14.94 | 16 | 23 | 6.14 |
| Day | 1 | 15.08 | 13 | 30 | 9.21 |
| Month | 6 | 7.96 | 8 | 10 | 1.22 |
| YEAR | 2023 | 2023.44 | 2023 | 2025 | 0.83 |

### Coverage

| Property | Value |
|---|---|
| Latitude | -28.95° to -5.29° S (South America / tropical routes) |
| Longitude | -62.28° to -35.11° W |
| Years | 2023 – 2025 |
| Months | June – October (Southern Hemisphere winter/spring) |
| Time of day | 0 – 23 h UTC |

### Derived Columns

| Column | Formula |
|---|---|
| `tempDevC` | `TAT − ISA_temp(altitude)` |
| `windKts` | `groundAirSpeed − Airspeed` |
| `targetAltitude` | Set equal to `altitude` (cruise, no altitude change) |
| `phase` | Fixed = 1 |
| `turbulence` | Fixed = 0 |
| `regime` | Fixed = 0 |

---

## 2. PPO Policy (`tail_policy.pt`)

**Config:** `approach_expanded_fuel_model_golden.json` (included in this folder)

### Architecture

```
Actor:  Linear(21→64, Tanh) → Linear(64→64, Tanh) → Linear(64→1, Tanh)
Critic: Linear(21→64, Tanh) → Linear(64→64, Tanh) → Linear(64→1)
log_std: learnable scalar (shape [1,1])
```

### Layer Shapes and Parameter Count

| Layer | Shape | Parameters |
|---|---|---|
| actor.0.weight | [64, 21] | 1,344 |
| actor.0.bias | [64] | 64 |
| actor.2.weight | [64, 64] | 4,096 |
| actor.2.bias | [64] | 64 |
| actor.4.weight | [1, 64] | 64 |
| actor.4.bias | [1] | 1 |
| critic.0.weight | [64, 21] | 1,344 |
| critic.0.bias | [64] | 64 |
| critic.2.weight | [64, 64] | 4,096 |
| critic.2.bias | [64] | 64 |
| critic.4.weight | [1, 64] | 64 |
| critic.4.bias | [1] | 1 |
| log_std | [1, 1] | 1 |
| **Total** | | **11,267** |

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | PPO (Proximal Policy Optimization) |
| Total timesteps | 250,000 |
| Batch size | 128 |
| K epochs per update | 4 |
| Learning rate | 2e-4 (Adam) |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| PPO clip ε | 0.2 |
| Entropy coefficient | 0.01 |
| Hidden size | 64 |
| Activation | Tanh |
| Oracle pretrain steps | 2,000 |
| Oracle pretrain batch | 256 |
| Reward shaping strength | 0.5 |
| Curriculum steps | 80,000 (cruise-only → mixed phases) |
| Training data | `data/qar_737800_cruise.csv` |
| Reward signal | Neural fuel model (`fuel_model.pt`) |

### Input Features (21)

| # | Feature | Obs mean | Obs std |
|---|---|---|---|
| 0 | altitude (ft) | 35,000 | 5,000 |
| 1 | grossWeight (kg) | 69,000 | 8,000 |
| 2 | TAT (°C) | -35 | 10 |
| 3 | CAS (kts) | 280 | 30 |
| 4 | tempDevC (°C) | 0 | 5 |
| 5 | windKts (kts) | 0 | 20 |
| 6 | phase | 1 | 0.8 |
| 7 | targetAltitude (ft) | 35,000 | 6,000 |
| 8 | turbulence | 0.2 | 0.2 |
| 9 | regime | 0 | 1 |
| 10 | angleOfAttackVoted (°) | 2 | 3 |
| 11 | horizontalStabilizerPosition (°) | 0 | 2 |
| 12 | totalFuelWeight (kg) | 8,000 | 3,000 |
| 13 | trackAngleTrue (°) | 180 | 90 |
| 14 | fmcMach | 0.78 | 0.05 |
| 15 | latitude (°) | -10 | 20 |
| 16 | longitude (°) | -50 | 20 |
| 17 | GMTHours (h) | 12 | 6 |
| 18 | Day | 15 | 10 |
| 19 | Month | 6 | 4 |
| 20 | YEAR | 2023 | 2 |

### Output

| # | Output | Range | Mapping |
|---|---|---|---|
| 0 | Mach command | [-1, 1] | `Mach = 0.78 + a × 0.08` → [0.70, 0.86] |

At inference the **deterministic actor mean** is used (no sampling).

---

## 3. Neural Fuel Model (`fuel_model.pt`)

### Architecture

```
Linear(17→64, ReLU) → Linear(64→64, ReLU) → Linear(64→1)
Target: fuel_per_nm (normalized), denormalized at inference
```

### Layer Shapes and Parameter Count

| Layer | Shape | Parameters |
|---|---|---|
| 0.weight | [64, 17] | 1,088 |
| 0.bias | [64] | 64 |
| 2.weight | [64, 64] | 4,096 |
| 2.bias | [64] | 64 |
| 4.weight | [1, 64] | 64 |
| 4.bias | [1] | 1 |
| **Total** | | **5,377** |

### Training

| Parameter | Value |
|---|---|
| Training rows | 120,000 (cruise filter: alt > 30k ft, GS > 100 kts) |
| Train / test split | 80 / 20 |
| Loss | MSE on normalized fuel_per_nm |
| Optimizer | Adam (lr=1e-3) |
| Epochs | 30 |

### Performance

| Metric | Value |
|---|---|
| MAE (normalized) | 0.3535 |
| RMSE (normalized) | 0.5626 |
| R² | 0.6938 |
| MAE (kg/NM) | 1.18 |
| RMSE (kg/NM) | 1.88 |

### Input Features (17)

| # | Feature | Mean | Std |
|---|---|---|---|
| 0 | altitude (ft) | 36,789 | 2,257 |
| 1 | grossWeight | 136,968 | 8,924 |
| 2 | TAT (°C) | -23.63 | 6.16 |
| 3 | Airspeed / CAS (kts) | 248.18 | 11.33 |
| 4 | groundAirSpeed (kts) | 442.70 | 32.34 |
| 5 | mach | 0.7642 | 0.0193 |
| 6 | angleOfAttackVoted (°) | 3.229 | 2.139 |
| 7 | horizontalStabilizerPosition (°) | 2.824 | 0.898 |
| 8 | totalFuelWeight (kg) | 10,418 | 6,533 |
| 9 | trackAngleTrue (°) | 115.94 | 107.76 |
| 10 | fmcMach | 0.7645 | 0.0397 |
| 11 | latitude (°) | -18.02 | 5.21 |
| 12 | longitude (°) | -44.87 | 4.93 |
| 13 | GMTHours (h) | 14.92 | 6.15 |
| 14 | Day | 15.09 | 9.22 |
| 15 | Month | 7.96 | 1.22 |
| 16 | YEAR | 2022.57 | 1.21 |

### Output

| Output | Target | Denormalization |
|---|---|---|
| fuel_per_nm (kg/NM) | normalized scalar | `fpn = pred × y_std + y_mean` → `fuel_flow = fpn × ground_speed` |

**Normalization constants (`fuel_model_scaler.json`):**
- `y_mean = 11.9788 kg/NM`
- `y_std  =  3.3369 kg/NM`
