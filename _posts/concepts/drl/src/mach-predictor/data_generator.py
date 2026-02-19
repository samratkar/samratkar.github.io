import pandas as pd
import numpy as np
import os

def generate_qar_data(tail_number, num_rows=1000):
    """
    Generates synthetic QAR data for a specific aircraft tail.
    """
    print(f"Generating data for {tail_number}...")
    
    # 1. Date Simulation
    dates = pd.date_range(start='2024-01-01', periods=num_rows, freq='h')
    
    # 2. Flight Parameters
    # Altitude: Cruising altitude typically between 30,000 and 40,000 ft
    altitude = np.random.uniform(30000, 41000, num_rows)
    
    # Gross Weight: Decreases as fuel burns, but for snapshot data we verify range
    # e.g., Boeing 737-800 MTOW is approx 79,000 kg. Let's say 60k - 75k kg
    gross_weight = np.random.uniform(60000, 78000, num_rows)
    
    # Total Air Temperature (TAT): Approx -50C to -10C depending on alt/speed
    # Simple model: -2C per 1000ft lapse rate from 15C ground (approx standard atmosphere)
    # Stratosphere is constant approx -56C. 
    # Let's just create some variance around -40 to -10 C
    tat = np.random.uniform(-45, -5, num_rows)
    
    # Calibrated Air Speed (CAS): ~250-300 knots at cruise
    cas = np.random.uniform(250, 310, num_rows)
    
    # Mach Number: The control variable we are interested in.
    # Typically 0.74 to 0.82 for narrowbodies
    mach = np.random.uniform(0.74, 0.82, num_rows)
    
    # FMC Mach: The target mach monitored by Flight Management Computer
    fmc_mach = mach + np.random.normal(0, 0.005, num_rows) # slightly different from actual
    
    # 3. Physics Simulation (Fuel Flow Approximation)
    # Fuel Flow ~ Drag * Speed / Efficiency
    # Drag ~ C_D * rho * V^2 * S
    # For a given weight (Lift = Weight), Drag is minimized at (L/D)max.
    # Higher Mach -> Higher Drag (Wave drag onset). 
    # Heavier Weight -> Higher Induced Drag.
    # Higher Altitude -> Lower Density (rho).
    
    # Simplified Synthetic Fuel Model:
    # Base flow + Weight Penalty + Drag Penalty (Speed away from optimal)
    
    # Optimal Mach approx 0.78. 
    # Curve is quadratic around optimal.
    
    optimal_mach = 0.78
    mach_penalty = 10000 * (mach - optimal_mach)**2
    
    weight_factor = (gross_weight - 60000) * 0.05
    
    # Altitude efficiency: Higher is better (less drag for same TAS usually, up to a point)
    # until engine efficiency drops.
    alt_factor = - (altitude - 30000) * 0.05
    
    base_flow_per_engine = 1200 # kg/hr
    
    noise = np.random.normal(0, 50, num_rows)
    
    fuel_flow_1 = base_flow_per_engine + (mach_penalty * 0.5) + (weight_factor * 0.5) + alt_factor + noise
    fuel_flow_2 = base_flow_per_engine + (mach_penalty * 0.5) + (weight_factor * 0.5) + alt_factor + np.random.normal(0, 50, num_rows)
    
    # Ensure positive
    fuel_flow_1 = np.maximum(fuel_flow_1, 500)
    fuel_flow_2 = np.maximum(fuel_flow_2, 500)
    
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
        'selectedFuelFlow1': fuel_flow_1,
        'selectedFuelFlow2': fuel_flow_2
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    tails = ["Tail_X1", "Tail_Y2"]
    
    for tail in tails:
        df = generate_qar_data(tail)
        output_path = f"data/{tail}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")

    # Also save a sample for the environment to use as a 'truth' model
    # (Since in real DRL we use the env, here we use data to regression-fit a model or just use the same formula)
