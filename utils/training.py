import gymnasium as gym
import numpy as np
from pathlib import Path
from env.blackjack_env import BlackjackEnv  


class ProgressWrapper(gym.Wrapper):
    def __init__(self, env, agent_name, progress_dict, start_episode):
        super().__init__(env)
        self.agent_name = agent_name
        self.progress_dict = progress_dict
        self.episode_count = start_episode
        self.progress_dict[self.agent_name] = self.episode_count

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Wenn eine Runde Blackjack vorbei ist
        if terminated or truncated:
            self.episode_count += 1
            # Nicht bei jedem Schritt funken (Performance!), sondern alle 5000 Runden
            if self.episode_count % 5000 == 0:
                self.progress_dict[self.agent_name] = self.episode_count
        return obs, reward, terminated, truncated, info


def train_single_agent(
    name,
    agent,
    agent_type,
    seed,
    selected_artifact,
    episodes_per_seed,
    checkpoint_interval,
    checkpoint_dir,
    run_id,
    progress_dict=None, # <--- NEU: Das geteilte Gedächtnis annehmen
):
    """Diese Funktion läuft isoliert in einem eigenen CPU-Prozess."""
    env = BlackjackEnv(
        num_decks=6,
        penetration=0.75,
        stand_on_soft_17=True,
    )

    env = gym.wrappers.RecordEpisodeStatistics(
        env,
        buffer_length=episodes_per_seed,
    )

    start_episode = 0
    if selected_artifact is not None:
        loaded_artifact = agent.load(selected_artifact)
        start_episode = int(loaded_artifact.get("episode") or 0)

    # NEU: Schalte den Episoden-Zähler dazwischen
    if progress_dict is not None:
        env = ProgressWrapper(env, name, progress_dict, start_episode)

    agent.env = env

    np.random.seed(seed)
    agent.env.reset(seed=seed)
    agent.env.action_space.seed(seed)

    episodes_to_train = max(0, episodes_per_seed - start_episode)
    if episodes_to_train == 0:
        if progress_dict is not None:
            progress_dict[name] = episodes_per_seed
        return name, agent

    agent.train(
        n_episodes=episodes_to_train,
        base_seed=seed,
        start_episode=start_episode,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        checkpoint_label=f"{run_id}_{name}_agent",
        checkpoint_metadata={
            "agent_name": name,
            "agent_type": agent_type,
            "run_id": run_id,
            "seed": seed,
            "start_episode": start_episode,
            "target_episode": episodes_per_seed,
        },
    )
    
    # Am Ende finalisieren
    if progress_dict is not None:
        progress_dict[name] = episodes_per_seed

    return name, agent
