# utils/env_utils.py
import gymnasium as gym
from env.blackjack_env import BlackjackEnv

def make_blackjack_env(seed: int, n_episodes: int, num_decks: int = 6, penetration: float = 0.75, stand_on_soft_17: bool = True):
    """Zentrale Factory-Funktion für die Blackjack-Umgebung."""
    env = BlackjackEnv(
        num_decks=num_decks,
        penetration=penetration,
        stand_on_soft_17=stand_on_soft_17,
    )
    env = gym.wrappers.RecordEpisodeStatistics(
        env,
        buffer_length=n_episodes,
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env