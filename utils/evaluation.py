# utils/evaluation.py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_evaluation_cis(greedy_eval_df: pd.DataFrame, z_score: float = 1.96) -> pd.DataFrame:
    """
    Berechnet die 95% Konfidenzintervalle für alle Metriken (Win, Loss, Push, Avg Reward)
    pro Seed auf Basis der Simulationsgröße.
    """
    df = greedy_eval_df.copy()
    for col in ["episodes", "win_rate", "loss_rate", "push_rate", "average_reward", "std_reward"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    # CIs für die Raten (Wald-Intervall für Proportionen)
    for rate_col, label in [("win_rate", "Win Rate"), ("loss_rate", "Loss Rate"), ("push_rate", "Push Rate")]:
        se = np.sqrt(df[rate_col] * (1 - df[rate_col]) / df["episodes"])
        margin = z_score * se
        df[f"{label} 95% CI"] = df.apply(lambda r: f"[{r[rate_col] - margin[r.name]:.4f}, {r[rate_col] + margin[r.name]:.4f}]", axis=1)

    # CI für Average Reward (Zentraler Grenzwertsatz)
    avg_reward_se = df["std_reward"] / np.sqrt(df["episodes"])
    avg_reward_margin = z_score * avg_reward_se
    df["Avg Reward 95% CI"] = df.apply(lambda r: f"[{r['average_reward'] - avg_reward_margin[r.name]:.4f}, {r['average_reward'] + avg_reward_margin[r.name]:.4f}]", axis=1)

    return df[["win_rate", "Win Rate 95% CI", "loss_rate", "Loss Rate 95% CI", 
               "push_rate", "Push Rate 95% CI", "average_reward", "Avg Reward 95% CI"]]


def plot_evaluation_metrics_with_cis(
    greedy_eval_df: pd.DataFrame, 
    agent_styles: dict, 
    split_agent_name_func, 
    save_fig_func=None, 
    z_score: float = 1.96
):
    """
    Erstellt ein 2x2 Grid für Win-, Loss-, Push-Rate und Average Reward
    mit echten statistischen Konfidenzintervallen (gepoolte Daten über Seeds).
    """
    df = greedy_eval_df.copy().reset_index().rename(columns={"index": "agent_name"})
    name_parts = df["agent_name"].apply(lambda n: pd.Series(split_agent_name_func(n), index=["agent_type", "seed"]))
    df = pd.concat([df, name_parts], axis=1)

    metric_columns = ["win_rate", "loss_rate", "push_rate", "average_reward", "std_reward", "episodes"]
    for column in metric_columns:
        df[column] = pd.to_numeric(df[column])

    # Aggregation über die Seeds (Gewichtetes Pooling)
    grouped = df.groupby("agent_type").agg({
        "episodes": "sum",
        "win_rate": "mean",
        "loss_rate": "mean",
        "push_rate": "mean",
        "average_reward": "mean",
        "std_reward": "mean"
    }).reset_index()

    # Fehlerbalken-Margen berechnen
    grouped["win_ci"] = z_score * np.sqrt(grouped["win_rate"] * (1 - grouped["win_rate"]) / grouped["episodes"])
    grouped["loss_ci"] = z_score * np.sqrt(grouped["loss_rate"] * (1 - grouped["loss_rate"]) / grouped["episodes"])
    grouped["push_ci"] = z_score * np.sqrt(grouped["push_rate"] * (1 - grouped["push_rate"]) / grouped["episodes"])
    grouped["avg_reward_ci"] = z_score * (grouped["std_reward"] / np.sqrt(grouped["episodes"]))

    # 2x2 Plot Setup
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.ravel()

    metric_plot_config = [
        ("win_rate", "win_ci", "Win Rate", (0, 0.6)),
        ("loss_rate", "loss_ci", "Loss Rate", (0, 0.7)),
        ("push_rate", "push_ci", "Push Rate", (0, 0.2)),
        ("average_reward", "avg_reward_ci", "Average Reward", None),
    ]

    agent_types = [at for at in ["baseline", "counting"] if at in grouped["agent_type"].unique()]
    x = np.arange(len(agent_types))
    colors = [agent_styles[at]["color"] for at in agent_types]
    labels = [agent_styles[at]["label"] for at in agent_types]

    for ax, (metric, ci_col, title, ylim) in zip(axs, metric_plot_config):
        means = [grouped.loc[grouped["agent_type"] == at, metric].values[0] for at in agent_types]
        cis = [grouped.loc[grouped["agent_type"] == at, ci_col].values[0] for at in agent_types]

        # Balken mit statistischem Fehlerbalken (CI) statt simpler Standardabweichung der Seeds
        ax.bar(x, means, yerr=cis, color=colors, width=0.45, capsize=8, alpha=0.85, edgecolor='black')

        for idx, at in enumerate(agent_types):
            values = df.loc[df["agent_type"] == at, metric].to_numpy()
            jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(np.full(len(values), idx) + jitter, values, color="black", edgecolor="white", s=45, zorder=3, alpha=0.8)

        ax.set_title(f"{title} (Greedy Evaluation mit 95% CI)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        if ylim is not None:
            ax.set_ylim(*ylim)

        for idx, value in enumerate(means):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    if save_fig_func:
        save_fig_func("final_evaluation_metrics_with_cis")
    plt.show()

import numpy as np
import pandas as pd
from utils.env_utils import make_blackjack_env

def evaluate_single_agent_parallel(agent_name, q_values, state_encoder, source_label, n_episodes, base_seed, progress_dict=None):
    # Absolut sauberer Import, kein sys.modules-Hack unter Windows nötig!
    env = make_blackjack_env(seed=base_seed, n_episodes=n_episodes)
    
    wins = losses = pushes = 0
    action_counts = {0: 0, 1: 0}
    sum_rewards = 0.0
    sum_sq_rewards = 0.0

    # Dynamisches Update-Intervall für den Fortschrittsbalken berechnen (z.B. alle 5%)
    update_interval = max(1, n_episodes // 20)

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            state = state_encoder(obs)
            values = q_values.get(state)
            action = int(np.argmax(values)) if values is not None else 0
            
            action_counts[action] = action_counts.get(action, 0) + 1
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            pushes += 1

        sum_rewards += episode_reward
        sum_sq_rewards += episode_reward ** 2
        
        # Jetzt funktioniert der Balken sowohl bei 1.000 als auch bei 100.000 Episoden flüssig!
        if progress_dict is not None and episode % update_interval == 0:
            progress_dict[agent_name] = episode

    total = wins + losses + pushes
    mean_reward = sum_rewards / total if total else 0.0
    var_reward = (sum_sq_rewards / total) - (mean_reward ** 2) if total else 0.0
    std_reward = np.sqrt(max(0.0, var_reward))

    if progress_dict is not None:
        progress_dict[agent_name] = n_episodes

    return agent_name, {
        "source": source_label,
        "episodes": total,
        "win_rate": wins / total if total else 0.0,
        "loss_rate": losses / total if total else 0.0,
        "push_rate": pushes / total if total else 0.0,
        "average_reward": mean_reward,
        "std_reward": std_reward,
        "action_distribution": {"stand": action_counts.get(0, 0), "hit": action_counts.get(1, 0)},
    }