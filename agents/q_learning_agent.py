from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import cast
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
import pickle

from pathlib import Path

class QValueFactory:

    def __init__(self, action_shape: int):
        self.action_shape = action_shape

    def __call__(self):
        """Wird aufgerufen, wenn ein neuer State im defaultdict abgefragt wird."""
        return np.zeros(self.action_shape, dtype=np.float32)



class QLearningBlackjackAgent:
    """
    Generic Q-learning agent for discrete-action Gym environments.
    """
    
    def __init__(
        self,
        env: gym.Env,
        state_encoder,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
    ):
        self.env = env
        self.state_encoder = state_encoder

        action_space = cast(Discrete, env.action_space)
        self.action_shape = int(action_space.n)

        self.q_values: defaultdict[object, np.ndarray] = defaultdict(QValueFactory(self.action_shape))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error: list[float] | deque[float] = []
        self.episode_rewards: list[float] | deque[float] = []
        self.checkpoint_paths: list[Path] = []

    # ---------------------------------------------------------
    # Core RL methods
    # ---------------------------------------------------------

    def get_action(self, obs) -> int:
        state = self.state_encoder(obs)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        return int(np.argmax(self.q_values[state]))

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ) -> float:
        state = self.state_encoder(obs)
        next_state = self.state_encoder(next_obs)

        best_next_q = (
            0.0
            if terminated
            else np.max(self.q_values[next_state])
        )

        td_target = reward + self.discount_factor * best_next_q
        td_error = td_target - self.q_values[state][action]

        self.q_values[state][action] += self.lr * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.final_epsilon,
            self.epsilon - self.epsilon_decay,
        )

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    
    def train(
        self,
        n_episodes: int,
        base_seed: int = 42,
        show_progress: bool = True,
        start_episode: int = 0,
        checkpoint_interval: int | None = None,
        checkpoint_dir: Path | str | None = None,
        checkpoint_label: str | None = None,
        checkpoint_metadata: dict | None = None,
        checkpoint_include_history: bool = False,
        history_limit: int | None = None,
    ) -> list[float]:
        """
        Train agent and return episode rewards.
        """

        if history_limit is not None and history_limit <= 0:
            raise ValueError("history_limit must be positive.")

        self.episode_rewards = deque[float](maxlen=history_limit) if history_limit is not None else list[float]()
        self.training_error = deque[float](maxlen=history_limit) if history_limit is not None else list[float]()
        self.checkpoint_paths.clear()

        if start_episode < 0:
            raise ValueError("start_episode must not be negative.")
        if n_episodes < 0:
            raise ValueError("n_episodes must not be negative.")

        end_episode = start_episode + n_episodes

        if checkpoint_interval is not None:
            if checkpoint_interval <= 0:
                raise ValueError("checkpoint_interval must be positive.")
            if checkpoint_dir is None:
                raise ValueError("checkpoint_dir is required when checkpoint_interval is set.")
            resolved_checkpoint_dir = Path(checkpoint_dir)
            resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            resolved_checkpoint_dir = None

        label = checkpoint_label or self.__class__.__name__
        episodes = range(start_episode, end_episode)

        if show_progress:
            episodes = tqdm(
                episodes,
                desc=self.__class__.__name__,
                leave=False,
            )

        for episode in episodes:
            obs, _ = self.env.reset(seed=base_seed + episode)

            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_td_errors = []

            while not (terminated or truncated):
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                reward = float(reward)

                td_error = self.update(
                    obs,
                    action,
                    reward,
                    terminated,
                    next_obs,
                )

                episode_td_errors.append(td_error)
                episode_reward += reward
                obs = next_obs

            self.episode_rewards.append(episode_reward)
            self.training_error.append(float(np.mean(np.abs(episode_td_errors))))
            self.decay_epsilon()

            episode_number = episode + 1
            if checkpoint_interval is not None and episode_number % checkpoint_interval == 0:
                assert resolved_checkpoint_dir is not None
                checkpoint_path = resolved_checkpoint_dir / f"{label}_episode_{episode_number}.pkl"
                self.save(
                    checkpoint_path,
                    label=label,
                    artifact_type="checkpoint",
                    episode=episode_number,
                    n_episodes=end_episode,
                    base_seed=base_seed,
                    metadata=checkpoint_metadata,
                    include_history=checkpoint_include_history,
                )
                self.checkpoint_paths.append(checkpoint_path)

        return list(self.episode_rewards)

    # ---------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------

    def save(
        self,
        artifact_path: Path,
        label: str,
        env=None,
        artifact_type: str = "final",
        episode: int | None = None,
        n_episodes: int | None = None,
        base_seed: int | None = None,
        metadata: dict | None = None,
        include_history: bool = True,
    ) -> None:
        """
        Speichert den Zustand des Agenten sauber als Pickle-Datei.
        Konvertiert das defaultdict in ein normales dict, um Serialisierungsfehler zu vermeiden.
        """
        artifact_path = Path(artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        clean_q_values = {k: np.array(v, dtype=np.float32) for k, v in self.q_values.items()}
        saved_episode = episode if episode is not None else len(self.episode_rewards)
        saved_n_episodes = n_episodes if n_episodes is not None else saved_episode

        artifact = {
            "label": label,
            "artifact_type": artifact_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episode": saved_episode,
            "n_episodes": saved_n_episodes,
            "base_seed": base_seed,
            "q_values": clean_q_values,
            "q_state_count": len(clean_q_values),
            "q_states": len(clean_q_values),
            "training_error": list(self.training_error) if include_history else [],
            "episode_rewards": list(self.episode_rewards) if include_history else [],
            "training_error_tail": list(self.training_error)[-1000:],
            "episode_rewards_tail": list(self.episode_rewards)[-1000:],
            "learning_rate": self.lr,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon": self.final_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "metadata": metadata or {},
        }

        if env is not None:
            if hasattr(env, "return_queue") and env.return_queue:
                artifact["episode_returns"] = list(env.return_queue)
            if hasattr(env, "length_queue") and env.length_queue:
                artifact["episode_lengths"] = list(env.length_queue)

        with artifact_path.open("wb") as f:
            pickle.dump(artifact, f)

    def load(self, artifact_path: Path | str) -> dict:
        """
        Laedt den gespeicherten Zustand des Agenten aus einem Artefakt
        und stellt das defaultdict fuer das weitere Training wieder her.
        """
        artifact_path = Path(artifact_path)

        if not artifact_path.exists():
            raise FileNotFoundError(f"Kein Artefakt unter {artifact_path} gefunden.")

        with artifact_path.open("rb") as f:
            artifact = pickle.load(f)

        self.loaded_artifact_metadata = {
            key: value
            for key, value in artifact.items()
            if key != "q_values"
        }

        self.lr = artifact["learning_rate"]
        self.discount_factor = artifact["discount_factor"]
        self.initial_epsilon = artifact.get("initial_epsilon", self.initial_epsilon)
        self.epsilon = artifact["epsilon"]
        self.final_epsilon = artifact.get("final_epsilon", self.final_epsilon)
        self.epsilon_decay = artifact.get("epsilon_decay", self.epsilon_decay)
        self.training_error = artifact.get("training_error") or artifact.get("training_error_tail", [])
        self.episode_rewards = artifact.get("episode_rewards") or artifact.get("episode_rewards_tail", [])

        self.q_values = defaultdict(
            lambda: np.zeros(self.action_shape, dtype=np.float32)
        )
        
        for k, v in artifact["q_values"].items():
            self.q_values[k] = np.array(v, dtype=np.float32)

        return artifact
