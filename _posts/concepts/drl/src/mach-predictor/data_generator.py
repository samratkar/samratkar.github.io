import pandas as pd
import numpy as np
import os

# Keep physics aligned with AircraftEnv
from aircraft_env import (
    isa_atmosphere,
    mach_to_tas,
    tas_to_cas,
    tat_from_mach,
    G0,
    T0,
)

S_REF = 125.0
AR = 9.5
E = 0.82
CD0 = 0.02
K = 1.0 / (np.pi * E * AR)
CD_WAVE = 0.015
MACH_CRIT = 0.78
TSFC_BASE = 1.6e-5

def fuel_flow_model(altitude_ft, weight_kg, mach, temp_dev_c):
    alt_m = altitude_ft * 0.3048
    temp_k, _, rho = isa_atmosphere(alt_m)
    temp_k = temp_k + temp_dev_c
    tas = mach_to_tas(mach, temp_k)
    q = 0.5 * rho * tas**2
    weight_n = weight_kg * G0
    cl = weight_n / (q * S_REF)
    wave = max(0.0, (mach - MACH_CRIT) / 0.08)
    cd = CD0 + K * cl**2 + CD_WAVE * wave**2
    drag_n = q * S_REF * cd
    tsfc = TSFC_BASE * (1.0 + 0.12 * (alt_m / 11000.0)) * (1.0 + 0.01 * (T0 - temp_k))
    fuel_flow_kg_s = drag_n * tsfc
    return fuel_flow_kg_s * 3600.0

def generate_qar_data(tail_number, num_rows=1000):
    """
    Generates synthetic QAR data for a specific aircraft tail.
    """
    print(f"Generating data for {tail_number}...")
    
    # 1. Date Simulation
    dates = pd.date_range(start='2024-01-01', periods=num_rows, freq='h')
    
    # 2. Flight Parameters
    # Altitude: Cruising altitude typically between 30,000 and 41,000 ft
    altitude = np.random.uniform(30000, 41000, num_rows)
    # Gross Weight: 60k - 78k kg
    gross_weight = np.random.uniform(60000, 78000, num_rows)
    # Mach Number: typically 0.72 to 0.84 for narrowbodies
    mach = np.random.uniform(0.72, 0.84, num_rows)

    # Phase and target altitude
    phase = np.zeros(num_rows)
    target_altitude = np.zeros(num_rows)
    for i in range(num_rows):
        phase[i] = np.random.choice([0.0, 1.0, 2.0], p=[0.2, 0.6, 0.2])
        if phase[i] == 0.0:
            target_altitude[i] = np.random.uniform(33000, 39000)
        elif phase[i] == 2.0:
            target_altitude[i] = np.random.uniform(15000, 25000)
        else:
            target_altitude[i] = altitude[i] + np.random.uniform(-500, 500)

    # Temperature deviation and wind (simple profile)
    temp_dev = np.zeros(num_rows)
    wind_kts = np.zeros(num_rows)
    turbulence = np.zeros(num_rows)
    regime = np.zeros(num_rows)
    temp_dev[0] = np.random.normal(0.0, 2.0)
    wind_kts[0] = np.random.normal(0.0, 10.0)
    turbulence[0] = np.clip(np.random.normal(0.2, 0.1), 0.0, 1.0)
    regime[0] = np.random.choice([0.0, 1.0], p=[0.9, 0.1])
    for i in range(1, num_rows):
        temp_dev[i] = np.clip(0.95 * temp_dev[i - 1] + np.random.normal(0.0, 0.3), -8.0, 8.0)
        wind_kts[i] = np.clip(0.90 * wind_kts[i - 1] + np.random.normal(0.0, 2.5), -60.0, 60.0)
        turbulence[i] = np.clip(0.2 + 0.85 * (turbulence[i - 1] - 0.2) + np.random.normal(0.0, 0.05), 0.0, 1.0)
        if np.random.rand() < 0.02:
            regime[i] = 1.0 - regime[i - 1]
        else:
            regime[i] = regime[i - 1]
    
    # FMC Mach: The target mach monitored by Flight Management Computer
    fmc_mach = mach + np.random.normal(0, 0.005, num_rows) # slightly different from actual
    
    # 3. Physics Simulation (Fuel Flow Approximation)
    fuel_flow_1 = np.zeros(num_rows)
    fuel_flow_2 = np.zeros(num_rows)
    tat = np.zeros(num_rows)
    cas = np.zeros(num_rows)

    for i in range(num_rows):
        temp_k, _, rho = isa_atmosphere(altitude[i] * 0.3048)
        temp_k = temp_k + temp_dev[i]
        tat[i] = tat_from_mach(temp_k, mach[i])
        tas = mach_to_tas(mach[i], temp_k)
        cas[i] = tas_to_cas(tas, rho)

        base_flow = fuel_flow_model(altitude[i], gross_weight[i], mach[i], temp_dev[i])
        noise = np.random.normal(0, 30)
        fuel_flow_1[i] = max(base_flow + noise + 25.0 * turbulence[i], 500.0)
        fuel_flow_2[i] = max(base_flow + np.random.normal(0, 30) + 25.0 * turbulence[i], 500.0)
    
    # 4. DataFrame Construction
    df = pd.DataFrame({
        'Month': dates.month,
        'Day': dates.day,
        'Year': dates.year,
        'altitude': altitude,
        'grossWeight': gross_weight,
        'totalAirTemperatureCelsius': tat,
        'CAS': cas,
        'mach': mach,
        'fmcMach': fmc_mach,
        'tempDevC': temp_dev,
        'windKts': wind_kts,
        'turbulence': turbulence,
        'regime': regime,
        'phase': phase,
        'targetAltitude': target_altitude,
        'tail': tail_number,
        'selectedFuelFlow1': fuel_flow_1,
        'selectedFuelFlow2': fuel_flow_2
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    tails = [f"Tail_{i:02d}" for i in range(1, 51)]
    
    for tail in tails:
        df = generate_qar_data(tail)
        output_path = f"data/{tail}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")

    # Also save a sample for the environment to use as a 'truth' model
    # (Since in real DRL we use the env, here we use data to regression-fit a model or just use the same formula)
