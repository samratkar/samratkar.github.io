id: my-first-codelab
summary: Build a sample app with X
categories: web, beginner
tags: web
status: draft
authors: Samrat Kar

# Aircraft Mach Optimization using Deep Reinforcement Learning

This application demonstrates how to use Deep Reinforcement Learning (DRL) to optimize the Mach number for aircraft tails to minimize fuel consumption.

## Problem Statement
Aircraft fuel efficiency depends significantly on the Mach number (speed) flown relative to the aircraft's weight, altitude, and temperature. By analyzing historical QAR (Quick Access Recorder) data, we can learn a policy to recommend the optimal Mach number for specific flight conditions.

## Solution Approach
1.  **Data Generation**: We generate synthetic QAR data simulating realistic flight physics (Fuel Flow vs. Mach/Weight/Alt).
2.  **Environment**: A custom Gymnasium environment (`AircraftEnv`) simulates the aircraft fuel burn based on the synthetic physics model.
3.  **RL Agent**: A PPO (Proximal Policy Optimization) agent is trained to select the Mach number that minimizes fuel flow (maximizes negative fuel burn).
4.  **Prediction**: The trained agent can be used to predict the optimal Mach number for new flight conditions.

## Project Structure
- `src/data_generator.py`: Generates synthetic flight data CSVs in `data/`.
- `src/aircraft_env.py`: Custom Gym environment.
- `src/train_agent.py`: Training script using PyTorch PPO.
- `src/predict_mach.py`: Inference script to query the trained model.
- `requirements.txt`: Python dependencies.

## Setup
1.  **Install Dependencies**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

### 1. Generate Data
Generate synthetic data for tails (e.g., Tail_X1):
```bash
python src/data_generator.py
```
This creates `data/Tail_X1.csv`.

### 2. Train the Agent
Train the DRL agent on the generated data/environment:
```bash
python src/train_agent.py
```
This will train for 100,000 timesteps and save the model to `models/tail_policy.pt`.

### 3. Predict Optimal Mach
Use the trained model to predict the optimal Mach for specific conditions:
```bash
# Usage: python src/predict_mach.py <Altitude> <Weight> <TAT> <CAS>
python src/predict_mach.py 36000 70000 -45 280
```
Output:
```
Predicted Optimal Mach: 0.7802
```

## How it Works
- The **Environment** calculates a "Fuel Penalty" based on how far the chosen Mach is from the theoretical optimal (approx 0.78 in our simulation) and other factors like weight and altitude.
- The **Agent** observes `[Altitude, Weight, TAT, CAS]` and outputs a continuous action mapped to Mach `[0.70, 0.86]`.
- Over time, the agent learns to output Mach numbers close to 0.78 (or the optimal for the condition), minimizing the fuel flow.
