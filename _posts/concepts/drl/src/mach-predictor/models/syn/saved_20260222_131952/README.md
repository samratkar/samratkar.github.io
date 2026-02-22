# Training Run Summary

- Timestamp: `20260222_131952`
- Config: `configs/approach_expanded_fuel_model_syn.json`
- QAR data path: `data/qar_737800_synthetic.csv`
- Phase mode: `cruise_only`

## Timings
- Pretraining: `124.1s` (`2.1 min`)
- Total training: `168.3s` (`2.8 min`)

## Key Hyperparameters
- learning_rate: `0.0002`
- gamma: `0.99`
- eps_clip: `0.2`
- k_epochs: `4`
- batch_size: `128`
- total_timesteps: `30000`
- hidden_size: `64`
- pretrain_steps: `300`
- pretrain_batch: `128`
- reward_shaping_strength: `1.0`
- curriculum_steps: `50000`

## Artifacts
- Policy (run folder): `models/syn/saved_20260222_131952/tail_policy.pt`
- Policy (latest): `models/syn/tail_policy.pt`
- Fuel model copy: `not copied (path missing)`
- Fuel scaler copy: `not copied (path missing)`
- Stats JSON: `models/syn/saved_20260222_131952/training_stats.json`
