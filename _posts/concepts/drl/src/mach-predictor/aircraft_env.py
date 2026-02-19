import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class AircraftEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    The goal is to find the optimal Mach number to minimize fuel flow given flight conditions.
    """
    metadata = {'render_mode': ['human']}

    def __init__(self, data_path=None):
        super(AircraftEnv, self).__init__()
        
        # Load data to sample realistic initial states
        self.df = None
        if data_path:
            self.df = pd.read_csv(data_path)
        
        # Action Space: Continuous Mach number selection [0.70, 0.86]
        # We normalize action to [-1, 1] for stable learning in PPO/SAC
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation Space:
        # [Altitude, GrossWeight, TAT, CAS]
        # We also normalize these roughly or let the NN handle it (normalization is better)
        # Low/High limits are theoretical maxes
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -100, 0]), 
            high=np.array([60000, 100000, 50, 500]), 
            dtype=np.float32
        )
        
        self.state = None
        self.current_step = 0
        
        # Physics constants (approximations matching the data generator)
        self.optimal_mach = 0.78
        self.base_flow = 1200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize state with a random sample from our "historical data" or random realistic values
        if self.df is not None:
             sample = self.df.sample(1).iloc[0]
             self.altitude = sample['altitude']
             self.weight = sample['grossWeight']
             self.tat = sample['totalAirTemperatureCelsius']
             self.cas = sample['CAS']
        else:
             self.altitude = np.random.uniform(30000, 40000)
             self.weight = np.random.uniform(60000, 78000)
             self.tat = np.random.uniform(-50, -20)
             self.cas = np.random.uniform(250, 320)
             
        self.state = np.array([self.altitude, self.weight, self.tat, self.cas], dtype=np.float32)
        self.current_step = 0
        
        return self.state, {}

    def step(self, action):
        # Convert normalized action [-1, 1] back to Mach [0.70, 0.86]
        mach = 0.78 + (action[0] * 0.08) 
        
        # Calculate Reward (Minimize Fuel Flow)
        # We use the same physics logic as the generator to simulate "Truth"
        
        # Fuel Flow Model
        mach_penalty = 10000 * (mach - self.optimal_mach)**2
        weight_factor = (self.weight - 60000) * 0.05
        alt_factor = - (self.altitude - 30000) * 0.05
        
        fuel_flow = self.base_flow + (mach_penalty * 0.5) + (weight_factor * 0.5) + alt_factor
        
        # Reward is negative fuel flow (divided by 1000 to keep scale reasonable, approx -1.0 to -2.0)
        reward = -fuel_flow / 1000.0
        
        # Episode termination
        # In this context, it's a "one-step" bandit problem or short horizon. 
        # We can treat each step as a new condition or run for a short duration.
        # Let's say one episode is 1 step for "Instantaneous Optimization" or we can simulate a cruise segment.
        # For simplicity: One Step Episode (Bandit style) is easiest to learn for this specific mapping.
        terminated = True 
        truncated = False
        
        info = {
            "mach": mach,
            "fuel_flow": fuel_flow,
            "optimal_mach": self.optimal_mach # For debugging/rendering
        }
        
        return self.state, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
