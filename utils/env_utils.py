# utils/env_utils.py
import gymnasium as gym
from env.blackjack_env import BlackjackEnv

def make_blackjack_env(seed: int, n_episodes: int, num_decks: int = 6, penetration: float = 0.75, stand_on_soft_17: bool = True, statistics_buffer_length: int = 100_000):
    """Zentrale Factory-Funktion für die Blackjack-Umgebung."""
    base_env = BlackjackEnv(
        num_decks=num_decks,
        penetration=penetration,
        stand_on_soft_17=stand_on_soft_17,
    )
    env = gym.wrappers.RecordEpisodeStatistics(
        base_env,
        buffer_length=min(n_episodes, statistics_buffer_length),
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
