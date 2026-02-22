import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aircraft_env import AircraftEnv, isa_atmosphere
from train_agent import ActorCritic

DEFAULT_ROOT = Path("/Users/samrat.kar/cio/airlines-data/glo/737-800/baseline")
REQUIRED_COLS = [
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


def load_qar_rows(input_csv, data_root, max_rows, sample_rows, seed):
    if input_csv:
        qar = pd.read_csv(input_csv)
        if not set(REQUIRED_COLS).issubset(qar.columns):
            missing = sorted(set(REQUIRED_COLS) - set(qar.columns))
            raise RuntimeError(f"Input CSV missing required columns: {missing}")
        qar = qar.dropna(subset=REQUIRED_COLS)
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
            df = df.dropna(subset=REQUIRED_COLS)
            rows.append(df)
            row_count += len(df)
            if row_count >= max_rows:
                break
        if not rows:
            raise RuntimeError("No usable QAR rows found.")
        qar = pd.concat(rows, ignore_index=True)

    qar = qar[(qar["groundAirSpeed"] > 100) & (qar["altitude"] > 30000)]
    if len(qar) > sample_rows:
        qar = qar.sample(sample_rows, random_state=seed)
    return qar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=None, help="Single QAR CSV input")
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--policy_path", type=str, default="models/tail_policy.pt")
    parser.add_argument("--fuel_model_path", type=str, default="models/fuel_model.pt")
    parser.add_argument("--fuel_scaler_path", type=str, default="models/fuel_model_scaler.json")
    parser.add_argument("--max_rows", type=int, default=200000)
    parser.add_argument("--sample_rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_json", type=str, default=None, help="Optional JSON output")
    args = parser.parse_args()

    # Load policy + env (uses expanded-feature fuel model if configured in env)
    env = AircraftEnv(
        fuel_model_path=args.fuel_model_path,
        fuel_model_scaler_path=args.fuel_scaler_path,
    )
    state_dim = env.observation_space.shape[0]
    policy = ActorCritic(state_dim, 1)
    policy.load_state_dict(torch.load(args.policy_path))
    policy.eval()

    qar = load_qar_rows(args.input_csv, args.data_root, args.max_rows, args.sample_rows, args.seed)

    # Use explicit wind/temp/CAS if present; otherwise derive from raw QAR columns.
    if "windKts" in qar.columns:
        wind = qar["windKts"].astype(float).to_numpy()
    else:
        wind = (qar["groundAirSpeed"] - qar["Airspeed"]).astype(float).to_numpy()

    alt = qar["altitude"].to_numpy()
    if "tempDevC" in qar.columns:
        temp_dev = qar["tempDevC"].astype(float).to_numpy()
    else:
        alt_m = alt * 0.3048
        isa_temp = np.array([isa_atmosphere(a)[0] for a in alt_m]) - 273.15
        temp_dev = qar["totalAirTemperatureCelsius"].to_numpy() - isa_temp

    weight = qar["grossWeight"].to_numpy()
    if "CAS" in qar.columns:
        cas = qar["CAS"].astype(float).to_numpy()
    else:
        cas = qar["Airspeed"].astype(float).to_numpy()
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
    if args.results_json:
        out = {
            "qar_rows": int(len(qar)),
            "baseline_fuel_per_nm": baseline_mean,
            "policy_fuel_per_nm": policy_mean,
            "improvement_pct": improvement,
            "policy_path": args.policy_path,
            "fuel_model_path": args.fuel_model_path,
            "fuel_scaler_path": args.fuel_scaler_path,
            "input_csv": args.input_csv,
            "data_root": args.data_root,
        }
        p = Path(args.results_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("RESULTS_JSON", p)


if __name__ == "__main__":
    main()
