import pandas as pd
import numpy as np
from pathlib import Path
from aircraft_env import isa_atmosphere

DATA_ROOT = Path("/Users/samrat.kar/cio/airlines-data/glo/737-800/baseline")
OUT_PATH = Path("data/qar_737800_cruise.csv")
MAX_ROWS = 300000
SAMPLE_ROWS = 120000


def main():
    if OUT_PATH.exists():
        print(f"QAR dataset already exists at {OUT_PATH}, skipping build.")
        return

    rows = []
    row_count = 0
    for f in DATA_ROOT.rglob("*.csv"):
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
        if row_count >= MAX_ROWS:
            break

    if not rows:
        raise RuntimeError("No usable QAR rows found.")

    qar = pd.concat(rows, ignore_index=True)
    qar = qar[(qar["groundAirSpeed"] > 100) & (qar["altitude"] > 30000)]
    if len(qar) > SAMPLE_ROWS:
        qar = qar.sample(SAMPLE_ROWS, random_state=42)

    alt = qar["altitude"].to_numpy()
    alt_m = alt * 0.3048
    isa_temp = np.array([isa_atmosphere(a)[0] for a in alt_m]) - 273.15
    temp_dev = qar["totalAirTemperatureCelsius"].to_numpy() - isa_temp
    wind = (qar["groundAirSpeed"] - qar["Airspeed"]).to_numpy()

    out = pd.DataFrame(
        {
            "altitude": alt,
            "grossWeight": qar["grossWeight"].to_numpy(),
            "totalAirTemperatureCelsius": qar["totalAirTemperatureCelsius"].to_numpy(),
            "CAS": qar["Airspeed"].to_numpy(),
            "mach": qar["mach"].to_numpy(),
            "tempDevC": temp_dev,
            "windKts": wind,
            "phase": np.ones_like(alt),
            "targetAltitude": alt,
            "turbulence": np.full_like(alt, 0.2),
            "regime": np.zeros_like(alt),
            "angleOfAttackVoted": qar["angleOfAttackVoted"].to_numpy(),
            "horizontalStabilizerPosition": qar["horizontalStabilizerPosition"].to_numpy(),
            "totalFuelWeight": qar["totalFuelWeight"].to_numpy(),
            "trackAngleTrue": qar["trackAngleTrue"].to_numpy(),
            "fmcMach": qar["fmcMach"].to_numpy(),
            "latitude": qar["latitude"].to_numpy(),
            "longitude": qar["longitude"].to_numpy(),
            "GMTHours": qar["GMTHours"].to_numpy(),
            "Day": qar["Day"].to_numpy(),
            "Month": qar["Month"].to_numpy(),
            "YEAR": qar["YEAR"].to_numpy(),
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print("saved", OUT_PATH, "rows", len(out))


if __name__ == "__main__":
    main()
