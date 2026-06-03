from collections import defaultdict
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import pickle

import sys
from pathlib import Path

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
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.state_encoder = state_encoder

        self.q_values = defaultdict(
            lambda: np.zeros(env.action_space.n, dtype=np.float32)
        )

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error: list[float] = []
        self.episode_rewards: list[float] = []

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
    ) -> None:
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

        self.training_error.append(td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.final_epsilon,
            self.epsilon - self.epsilon_decay,
        )


    # ---------------------------------------------------------
    # Training loop (removes notebook duplication)
    # ---------------------------------------------------------
    
    def train(
        self,
        n_episodes: int,
        base_seed: int = 42,
        show_progress: bool = True,
    ) -> list[float]:
        """
        Train agent and return episode rewards.
        """

        self.episode_rewards.clear()

        episodes = range(n_episodes)

        if show_progress:
            # leave=False sorgt dafür, dass sich der Ladebalken nach dem 
            # erfolgreichen Training selbst aufräumt und die Ausgabe nicht flutet.
            episodes = tqdm(
                episodes,
                desc=self.__class__.__name__,
                leave=False,
            )

        for episode in episodes:
            obs, _ = self.env.reset(
                seed=base_seed + episode
            )

            terminated = False
            truncated = False

            episode_reward = 0.0

            while not (terminated or truncated):
                action = self.get_action(obs)

                next_obs, reward, terminated, truncated, _ = (
                    self.env.step(action)
                )

                self.update(
                    obs,
                    action,
                    reward,
                    terminated,
                    next_obs,
                )

                episode_reward += reward
                obs = next_obs

            self.episode_rewards.append(episode_reward)
            self.decay_epsilon()

        return self.episode_rewards

    # ---------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------

    def save(self, artifact_path: Path, label: str, env=None) -> None:
        """
        Speichert den Zustand des Agenten sauber als Pickle-Datei.
        Konvertiert das defaultdict in ein normales dict, um Serialisierungsfehler zu vermeiden.
        """
        # Ordnerstruktur (z.B. ./models/) automatisch erstellen, falls nicht vorhanden
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        # WICHTIG: defaultdict in ein Standard-dict konvertieren.
        # Das entfernt die nicht-picklbare lambda-Funktion!
        clean_q_values = {k: np.array(v, dtype=np.float32) for k, v in self.q_values.items()}

        artifact = {
            "label": label,
            "q_values": clean_q_values,
            "training_error": list(self.training_error),
            "episode_rewards": list(self.episode_rewards),
            "learning_rate": self.lr,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
        }

        # Falls eine Umgebung mit Statistik-Wrapper übergeben wurde, diese Queues sichern
        if env is not None:
            if hasattr(env, "return_queue") and env.return_queue:
                artifact["episode_returns"] = list(env.return_queue)
            if hasattr(env, "length_queue") and env.length_queue:
                artifact["episode_lengths"] = list(env.length_queue)

        with artifact_path.open("wb") as f:
            pickle.dump(artifact, f)

    def load(self, artifact_path: Path) -> None:
        """
        Lädt den gespeicherten Zustand des Agenten aus einem Artefakt
        und stellt das defaultdict für das weitere Training wieder her.
        """
        if not artifact_path.exists():
            raise FileNotFoundError(f"Kein Artefakt unter {artifact_path} gefunden.")

        with artifact_path.open("rb") as f:
            artifact = pickle.load(f)

        # Hyperparameter und Historie wiederherstellen
        self.lr = artifact["learning_rate"]
        self.discount_factor = artifact["discount_factor"]
        self.epsilon = artifact["epsilon"]
        self.training_error = artifact["training_error"]
        self.episode_rewards = artifact["episode_rewards"]

        # Q-Tabelle als defaultdict mit der ursprünglichen Lambda-Logik neu aufbauen
        self.q_values = defaultdict(
            lambda: np.zeros(self.env.action_space.n, dtype=np.float32)
        )
        
        # Gelernte Werte in das defaultdict übertragen
        for k, v in artifact["q_values"].items():
            self.q_values[k] = np.array(v, dtype=np.float32)