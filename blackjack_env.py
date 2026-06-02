import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class BlackjackEnv(gym.Env):
    def __init__(self):
        super(BlackjackEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: Stand, 1: Hit
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # Player's hand value (0-31)
            spaces.Discrete(11),  # Dealer's visible card (1-10)
            spaces.Discrete(2)    # Usable Ace (0 or 1)
        ))
        
        self.reset()

    