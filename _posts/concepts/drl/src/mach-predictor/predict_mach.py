import torch
import numpy as np
import sys
from train_agent import ActorCritic # Import model structure
from aircraft_env import AircraftEnv
import pandas as pd

def predict_optimal_mach(model_path, altitude, weight, tat, cas):
    # Load Model
    # We need to know input/output dims from env or hardcode if we know them
    env = AircraftEnv() 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = ActorCritic(state_dim, action_dim)
    try:
        policy.load_state_dict(torch.load(model_path))
        policy.eval()
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Please train first.")
        return None

    # Prepare Input
    state = np.array([altitude, weight, tat, cas], dtype=np.float32)
    state_tensor = torch.FloatTensor(state)
    
    # Inference
    with torch.no_grad():
        action_mean, _, _, _ = policy.get_action_and_value(state_tensor)
        
    # Convert Action [-1, 1] to Mach [0.70, 0.86]
    # Logic must match env.step()
    optimal_mach = 0.78 + (action_mean.item() * 0.08)
    
    return optimal_mach

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python predict_mach.py <Altitude> <Weight> <TAT> <CAS>")
        print("Example: python predict_mach.py 35000 70000 -40 280")
        
        # Run a demo prediction
        print("\nRunning Demo Prediction:")
        alt, weight, tat, cas = 36000, 72000, -45, 290
        mach = predict_optimal_mach("models/tail_policy.pt", alt, weight, tat, cas)
        if mach:
            print(f"Inputs: Alt={alt}, Wt={weight}, TAT={tat}, CAS={cas}")
            print(f"Predicted Optimal Mach: {mach:.4f}")
    else:
        alt = float(sys.argv[1])
        weight = float(sys.argv[2])
        tat = float(sys.argv[3])
        cas = float(sys.argv[4])
        
        mach = predict_optimal_mach("models/tail_policy.pt", alt, weight, tat, cas)
        if mach:
             print(f"Predicted Optimal Mach: {mach:.4f}")
