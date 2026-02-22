import numpy as np
import pandas as pd
from pathlib import Path
import torch

from aircraft_env import AircraftEnv, isa_atmosphere
from train_agent import ActorCritic


def main():
    qar_path = Path("data/qar_737800_cruise.csv")
    sample_rows = 20000

    # Load policy + env (uses expanded-feature fuel model if configured in env)
    env = AircraftEnv(
        fuel_model_path="models/fuel_model.pt",
        fuel_model_scaler_path="models/fuel_model_scaler.json",
    )
    state_dim = env.observation_space.shape[0]
    policy = ActorCritic(state_dim, 1)
    policy.load_state_dict(torch.load("models/tail_policy.pt"))
    policy.eval()

    if not qar_path.exists():
        raise RuntimeError(f"No usable QAR record found at {qar_path}. Run build_qar_dataset.py first.")

    qar = pd.read_csv(qar_path)
    if len(qar) > sample_rows:
        qar = qar.sample(sample_rows, random_state=42)

    # qar_737800_cruise.csv uses 'CAS' for airspeed and 'windKts' for groundSpeed-CAS
    wind = qar["windKts"].astype(float).to_numpy()
    alt = qar["altitude"].to_numpy()
    alt_m = alt * 0.3048
    isa_temp = np.array([isa_atmosphere(a)[0] for a in alt_m]) - 273.15
    temp_dev = qar["tempDevC"].to_numpy()

    weight = qar["grossWeight"].to_numpy()
    cas = qar["CAS"].to_numpy()
    gs = (cas + wind).clip(min=100.0)

    _turb = np.full_like(alt, 0.2, dtype=float)
    regime = np.zeros_like(alt, dtype=float)
    phase = np.ones_like(alt, dtype=float)
    target_alt = alt.copy()

    states = np.stack(
        [
            alt,
            weight,
            qar["totalAirTemperatureCelsius"].to_numpy(),
            cas,
            temp_dev,
            wind,
            phase,
            target_alt,
            _turb,
            regime,
            qar["angleOfAttackVoted"].to_numpy(),
            qar["horizontalStabilizerPosition"].to_numpy(),
            qar["totalFuelWeight"].to_numpy(),
            qar["trackAngleTrue"].to_numpy(),
            qar["fmcMach"].to_numpy(),
            qar["latitude"].to_numpy(),
            qar["longitude"].to_numpy(),
            qar["GMTHours"].to_numpy(),
            qar["Day"].to_numpy(),
            qar["Month"].to_numpy(),
            qar["YEAR"].to_numpy(),
        ],
        axis=1,
    )

    states_n = env.normalize_obs(states.astype(np.float32))

    with torch.no_grad():
        action, _, _, _ = policy.get_action_and_value(torch.FloatTensor(states_n), deterministic=True)

    action = action.numpy().reshape(-1)
    policy_mach = 0.78 + action * 0.08

    tat_arr = qar["totalAirTemperatureCelsius"].to_numpy()
    aoa_arr = qar["angleOfAttackVoted"].to_numpy()
    hstab_arr = qar["horizontalStabilizerPosition"].to_numpy()
    tfw_arr = qar["totalFuelWeight"].to_numpy()
    track_arr = qar["trackAngleTrue"].to_numpy()
    fmc_arr = qar["fmcMach"].to_numpy()
    lat_arr = qar["latitude"].to_numpy()
    lon_arr = qar["longitude"].to_numpy()
    gmt_arr = qar["GMTHours"].to_numpy()
    day_arr = qar["Day"].to_numpy()
    month_arr = qar["Month"].to_numpy()
    year_arr = qar["YEAR"].to_numpy()
    mach_arr = qar["mach"].to_numpy()

    def _set_env_state(i):
        env.altitude = float(alt[i])
        env.weight = float(weight[i])
        env.tat = float(tat_arr[i])
        env.cas = float(cas[i])
        env.wind_kts = float(wind[i])
        env.aoa = float(aoa_arr[i])
        env.hstab = float(hstab_arr[i])
        env.total_fuel_weight = float(tfw_arr[i])
        env.track_angle = float(track_arr[i])
        env.fmc_mach = float(fmc_arr[i])
        env.latitude = float(lat_arr[i])
        env.longitude = float(lon_arr[i])
        env.gmt_hours = float(gmt_arr[i])
        env.day = float(day_arr[i])
        env.month = float(month_arr[i])
        env.year = float(year_arr[i])

    # Baseline: fuel model evaluated at actual recorded mach
    fpn_baseline = []
    for i in range(len(qar)):
        _set_env_state(i)
        ff_hr = env._fuel_flow_from_model(float(np.clip(mach_arr[i], env.mach_min, env.mach_max)))
        fpn_baseline.append(ff_hr / float(gs[i]))
    fpn_baseline = np.array(fpn_baseline)

    # Policy: fuel model evaluated at policy-recommended mach
    fpn_policy = []
    for i in range(len(policy_mach)):
        _set_env_state(i)
        m = float(np.clip(policy_mach[i], env.mach_min, env.mach_max))
        ff_hr = env._fuel_flow_from_model(m)
        fpn_policy.append(ff_hr / float(gs[i]))
    fpn_policy = np.array(fpn_policy)

    baseline_mean = float(np.mean(fpn_baseline))
    policy_mean = float(np.mean(fpn_policy))
    improvement = (baseline_mean - policy_mean) / baseline_mean * 100.0

    print("QAR_ROWS", len(qar))
    print("baseline_fuel_per_nm", baseline_mean)
    print("policy_fuel_per_nm", policy_mean)
    print("improvement_pct", improvement)


if __name__ == "__main__":
    main()
