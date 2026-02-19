import numpy as np
import pandas as pd
from pathlib import Path
import torch

from aircraft_env import AircraftEnv, isa_atmosphere
from train_agent import ActorCritic


def main():
    root = Path("/Users/samrat.kar/cio/airlines-data/glo/737-800/baseline")
    max_rows = 200000
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

    rows = []
    row_count = 0
    for f in root.rglob("*.csv"):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        cols = [
            "selectedFuelFlow1",
            "selectedFuelFlow2",
            "groundAirSpeed",
            "Airspeed",
            "altitude",
            "grossWeight",
            "totalAirTemperatureCelsius",
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
        if not set(cols).issubset(df.columns):
            continue
        df = df[cols].dropna()
        rows.append(df)
        row_count += len(df)
        if row_count >= max_rows:
            break

    qar = pd.concat(rows, ignore_index=True)
    qar = qar[(qar["groundAirSpeed"] > 100) & (qar["altitude"] > 30000)]
    if len(qar) > sample_rows:
        qar = qar.sample(sample_rows, random_state=42)

    ff = qar["selectedFuelFlow1"] + qar["selectedFuelFlow2"]
    fpn_baseline = (ff / qar["groundAirSpeed"]).astype(float)

    wind = (qar["groundAirSpeed"] - qar["Airspeed"]).astype(float).to_numpy()
    alt = qar["altitude"].to_numpy()
    alt_m = alt * 0.3048
    isa_temp = np.array([isa_atmosphere(a)[0] for a in alt_m]) - 273.15
    temp_dev = qar["totalAirTemperatureCelsius"].to_numpy() - isa_temp

    weight = qar["grossWeight"].to_numpy()
    cas = qar["Airspeed"].to_numpy()

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

    fpn_policy = []
    for i in range(len(policy_mach)):
        m = float(np.clip(policy_mach[i], env.mach_min, env.mach_max))
        env.altitude = float(alt[i])
        env.weight = float(weight[i])
        env.tat = float(qar["totalAirTemperatureCelsius"].to_numpy()[i])
        env.cas = float(cas[i])
        env.wind_kts = float(wind[i])
        env.aoa = float(qar["angleOfAttackVoted"].to_numpy()[i])
        env.hstab = float(qar["horizontalStabilizerPosition"].to_numpy()[i])
        env.total_fuel_weight = float(qar["totalFuelWeight"].to_numpy()[i])
        env.track_angle = float(qar["trackAngleTrue"].to_numpy()[i])
        env.fmc_mach = float(qar["fmcMach"].to_numpy()[i])
        env.latitude = float(qar["latitude"].to_numpy()[i])
        env.longitude = float(qar["longitude"].to_numpy()[i])
        env.gmt_hours = float(qar["GMTHours"].to_numpy()[i])
        env.day = float(qar["Day"].to_numpy()[i])
        env.month = float(qar["Month"].to_numpy()[i])
        env.year = float(qar["YEAR"].to_numpy()[i])

        ff_hr = env._fuel_flow_from_model(m)
        gs = max(float(qar["groundAirSpeed"].to_numpy()[i]), 100.0)
        fpn_policy.append(ff_hr / gs)

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
