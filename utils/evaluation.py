import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.env_utils import make_blackjack_env

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
    Erstellt 4 separate Plots für Win-, Loss-, Push-Rate und Average Reward
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

    # Konfiguration für die 4 separaten Plots
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

    def grouped_value(agent_type, column):
        return float(grouped[grouped["agent_type"].eq(agent_type)][column].iloc[0])

    # Schleife erstellt nun für jede Metrik ein eigenes Figure-Objekt
    for metric, ci_col, title, ylim in metric_plot_config:
        fig, ax = plt.subplots(figsize=(8, 6))  # Einzelner Plot
        
        means = [grouped_value(at, metric) for at in agent_types]
        cis = [grouped_value(at, ci_col) for at in agent_types]

        # Balken mit statistischem Fehlerbalken (CI)
        ax.bar(x, means, yerr=cis, color=colors, width=0.45, capsize=8, alpha=0.85, edgecolor='black')

        # Jitter für die Einzeldaten (Seeds)
        for idx, at in enumerate(agent_types):
            values = df[df["agent_type"].eq(at)][metric].astype(float).tolist()
            jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(np.full(len(values), idx) + jitter, np.asarray(values), color="black", edgecolor="white", s=45, zorder=3, alpha=0.8)

        ax.set_title(f"{title} (Greedy Evaluation mit 95% CI)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        if ylim is not None:
            ax.set_ylim(*ylim)

        for idx, value in enumerate(means):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        
        # Speichern mit dynamischem Namen pro Metrik
        if save_fig_func:
            save_fig_func(f"final_evaluation_{metric}_with_ci")
            
        plt.show()
        

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


def _true_count_bucket(true_count, buckets):
    for label, lower, upper in buckets:
        if (lower is None or true_count >= lower) and (upper is None or true_count <= upper):
            return label
    return None


def evaluate_true_count_buckets_parallel(
    agent_name,
    q_values,
    state_encoder,
    source_label,
    n_episodes,
    base_seed,
    buckets,
    progress_dict=None,
):
    env = make_blackjack_env(seed=base_seed, n_episodes=n_episodes)
    bucket_stats = {
        label: {
            "episodes": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "sum_rewards": 0.0,
            "stand_actions": 0,
            "hit_actions": 0,
        }
        for label, _, _ in buckets
    }

    update_interval = max(1, n_episodes // 20)

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_bucket = _true_count_bucket(int(obs[4]), buckets)

        while not (terminated or truncated):
            current_bucket = _true_count_bucket(int(obs[4]), buckets)
            state = state_encoder(obs)
            values = q_values.get(state)
            action = int(np.argmax(values)) if values is not None else 0

            if current_bucket is not None:
                action_key = "hit_actions" if action == 1 else "stand_actions"
                bucket_stats[current_bucket][action_key] += 1

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        if episode_bucket is not None:
            stats = bucket_stats[episode_bucket]
            stats["episodes"] += 1
            stats["sum_rewards"] += episode_reward
            if episode_reward > 0:
                stats["wins"] += 1
            elif episode_reward < 0:
                stats["losses"] += 1
            else:
                stats["pushes"] += 1

        if progress_dict is not None and episode % update_interval == 0:
            progress_dict[agent_name] = episode

    if progress_dict is not None:
        progress_dict[agent_name] = n_episodes

    rows = []
    for label, _, _ in buckets:
        stats = bucket_stats[label]
        total = stats["episodes"]
        actions = stats["stand_actions"] + stats["hit_actions"]
        rows.append({
            "source": source_label,
            "bucket": label,
            "episodes": total,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "pushes": stats["pushes"],
            "reward_sum": stats["sum_rewards"],
            "win_rate": stats["wins"] / total if total else 0.0,
            "loss_rate": stats["losses"] / total if total else 0.0,
            "push_rate": stats["pushes"] / total if total else 0.0,
            "average_reward": stats["sum_rewards"] / total if total else 0.0,
            "stand_actions": stats["stand_actions"],
            "hit_actions": stats["hit_actions"],
            "stand_rate": stats["stand_actions"] / actions if actions else 0.0,
            "hit_rate": stats["hit_actions"] / actions if actions else 0.0,
        })

    return agent_name, rows


TRUE_COUNT_BUCKET_METRICS = [
    "average_reward",
    "win_rate",
    "loss_rate",
    "push_rate",
    "hit_rate",
    "stand_rate",
]

TRUE_COUNT_BUCKET_SUMMARY_COLUMNS = [
    "aggregation",
    "comparison_id",
    "bucket",
    "episodes",
    "average_reward",
    "win_rate",
    "loss_rate",
    "push_rate",
    "hit_rate",
    "stand_rate",
]


def build_true_count_bucket_tasks(
    agents,
    selected_eval_agent_func,
    split_agent_name_func,
    eval_task_name_func,
    baseline_state_key_func,
    counting_state_key_func,
    eval_seeds,
    eval_episodes,
    buckets,
):
    tasks = []
    agent_names = []

    for name in agents:
        eval_agent, source_label = selected_eval_agent_func(name)
        agent_type, _ = split_agent_name_func(name)
        encoder_func = baseline_state_key_func if agent_type == "baseline" else counting_state_key_func

        for eval_seed in eval_seeds:
            task_name = eval_task_name_func(name, eval_seed)
            agent_names.append(task_name)
            tasks.append((
                task_name,
                eval_agent.q_values,
                encoder_func,
                source_label,
                eval_episodes,
                eval_seed,
                buckets,
            ))

    return tasks, agent_names


def true_count_bucket_results_to_dataframe(bucket_results, split_eval_task_name_func, split_agent_name_func):
    bucket_rows = []
    for task_name, rows in bucket_results:
        agent_name, eval_seed = split_eval_task_name_func(task_name)
        agent_type, train_seed = split_agent_name_func(agent_name)
        for row in rows:
            bucket_rows.append({
                **row,
                "agent_name": agent_name,
                "agent_type": agent_type,
                "train_seed": train_seed,
                "eval_seed": eval_seed,
            })

    raw_df = pd.DataFrame(bucket_rows)
    if raw_df.empty:
        return raw_df

    raw_df["source"] = raw_df["source"].fillna("in_memory")
    raw_df["comparison_id"] = raw_df["agent_name"] + " | " + raw_df["source"]
    return raw_df


def weighted_bucket_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + TRUE_COUNT_BUCKET_SUMMARY_COLUMNS)

    summary = df.groupby(group_cols, sort=False).agg({
        "episodes": "sum",
        "wins": "sum",
        "losses": "sum",
        "pushes": "sum",
        "reward_sum": "sum",
        "stand_actions": "sum",
        "hit_actions": "sum",
    }).reset_index()

    total_actions = summary["stand_actions"] + summary["hit_actions"]
    safe_episodes = summary["episodes"].replace(0, np.nan)
    safe_actions = total_actions.replace(0, np.nan)

    summary["average_reward"] = summary["reward_sum"] / safe_episodes
    summary["win_rate"] = summary["wins"] / safe_episodes
    summary["loss_rate"] = summary["losses"] / safe_episodes
    summary["push_rate"] = summary["pushes"] / safe_episodes
    summary["stand_rate"] = summary["stand_actions"] / safe_actions
    summary["hit_rate"] = summary["hit_actions"] / safe_actions
    return summary.fillna(0.0)


def summarize_true_count_buckets(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_df = weighted_bucket_summary(
        raw_df,
        ["agent_name", "agent_type", "train_seed", "source", "comparison_id", "bucket"],
    )
    if seed_df.empty:
        return seed_df, seed_df

    weighted_agent_summary = weighted_bucket_summary(seed_df, ["agent_type", "bucket"])
    weighted_agent_summary["comparison_id"] = weighted_agent_summary["agent_type"] + " (all seeds)"
    weighted_agent_summary["aggregation"] = "weighted"

    seed_metric_aggs = {
        "episodes": "sum",
        "average_reward": ["mean", "median"],
        "win_rate": ["mean", "median"],
        "loss_rate": ["mean", "median"],
        "push_rate": ["mean", "median"],
        "stand_rate": ["mean", "median"],
        "hit_rate": ["mean", "median"],
        "stand_actions": "sum",
        "hit_actions": "sum",
    }
    seed_aggregates = seed_df.groupby(["agent_type", "bucket"], sort=False).agg(seed_metric_aggs)

    def seed_summary_from_aggregate(aggregate_name: str) -> pd.DataFrame:
        summary = pd.DataFrame({
            "agent_type": seed_aggregates.index.get_level_values("agent_type"),
            "bucket": seed_aggregates.index.get_level_values("bucket"),
            "episodes": seed_aggregates[("episodes", "sum")].to_numpy(),
            "average_reward": seed_aggregates[("average_reward", aggregate_name)].to_numpy(),
            "win_rate": seed_aggregates[("win_rate", aggregate_name)].to_numpy(),
            "loss_rate": seed_aggregates[("loss_rate", aggregate_name)].to_numpy(),
            "push_rate": seed_aggregates[("push_rate", aggregate_name)].to_numpy(),
            "stand_rate": seed_aggregates[("stand_rate", aggregate_name)].to_numpy(),
            "hit_rate": seed_aggregates[("hit_rate", aggregate_name)].to_numpy(),
            "stand_actions": seed_aggregates[("stand_actions", "sum")].to_numpy(),
            "hit_actions": seed_aggregates[("hit_actions", "sum")].to_numpy(),
        })
        summary["comparison_id"] = summary["agent_type"] + " (all seeds)"
        summary["aggregation"] = aggregate_name
        return summary

    summary_df = pd.concat(
        [
            seed_summary_from_aggregate("mean"),
            seed_summary_from_aggregate("median"),
            weighted_agent_summary,
        ],
        ignore_index=True,
    )
    return summary_df, seed_df


def save_true_count_bucket_reports(evaluation_dir, raw_df, seed_df, summary_df):
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(evaluation_dir / "true_count_bucket_metrics_raw.csv", index=False)
    seed_df.to_csv(evaluation_dir / "true_count_bucket_metrics_by_seed.csv", index=False)
    summary_df.to_csv(evaluation_dir / "true_count_bucket_metrics.csv", index=False)
    summary_df.to_json(evaluation_dir / "true_count_bucket_metrics.json", orient="records", indent=2)


def prepare_true_count_bucket_reports(
    bucket_results,
    split_eval_task_name_func,
    split_agent_name_func,
    evaluation_dir,
):
    raw_df = true_count_bucket_results_to_dataframe(
        bucket_results,
        split_eval_task_name_func=split_eval_task_name_func,
        split_agent_name_func=split_agent_name_func,
    )
    summary_df, seed_df = summarize_true_count_buckets(raw_df)
    save_true_count_bucket_reports(evaluation_dir, raw_df, seed_df, summary_df)
    return raw_df, summary_df, seed_df


def true_count_bucket_comparison_options(summary_df, seed_df):
    aggregate_options = sorted(summary_df["comparison_id"].dropna().unique()) if not summary_df.empty else []
    individual_options = sorted(seed_df["comparison_id"].dropna().unique()) if not seed_df.empty else []
    return aggregate_options + individual_options


def true_count_bucket_comparison_rows(summary_df, seed_df, selected_comparisons, aggregation: str) -> pd.DataFrame:
    frames = []
    selected_comparisons = list(selected_comparisons)
    aggregate_names = set(summary_df["comparison_id"].unique()) if not summary_df.empty else set()

    selected_aggregates = [item for item in selected_comparisons if item in aggregate_names]
    if selected_aggregates:
        aggregate_df = summary_df[
            (summary_df["aggregation"] == aggregation)
            & (summary_df["comparison_id"].isin(selected_aggregates))
        ]
        frames.append(aggregate_df)

    selected_individuals = [item for item in selected_comparisons if item not in aggregate_names]
    if selected_individuals:
        individual_df = weighted_bucket_summary(
            seed_df[seed_df["comparison_id"].isin(selected_individuals)],
            ["comparison_id", "bucket"],
        )
        individual_df["aggregation"] = "selected"
        frames.append(individual_df)

    if not frames:
        return pd.DataFrame(columns=TRUE_COUNT_BUCKET_SUMMARY_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def plot_true_count_bucket_comparison(
    summary_df,
    seed_df,
    buckets,
    metric,
    aggregation,
    selected_comparisons,
    summary_columns=None,
):
    summary_columns = summary_columns or TRUE_COUNT_BUCKET_SUMMARY_COLUMNS
    df = true_count_bucket_comparison_rows(summary_df, seed_df, selected_comparisons, aggregation)
    if df.empty:
        print("Keine Auswahl getroffen.")
        return pd.DataFrame(columns=summary_columns)

    bucket_order = [label for label, _, _ in buckets]
    df = df.copy()
    df["bucket"] = pd.Categorical(df["bucket"], categories=bucket_order, ordered=True)
    df = df.sort_values(["bucket", "comparison_id"])

    pivot = df.pivot(index="bucket", columns="comparison_id", values=metric).reindex(bucket_order)
    ax = pivot.plot(kind="bar", figsize=(12, 6), width=0.82)
    ax.set_title(f"{metric} nach True-Count-Bucket")
    ax.set_xlabel("True-Count-Bucket")
    ax.set_ylabel(metric)
    ax.legend(title="Agent / Checkpoint", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return df[summary_columns]


def create_true_count_bucket_widget(summary_df, seed_df, buckets):
    import ipywidgets as widgets
    from IPython.display import display

    comparison_options = true_count_bucket_comparison_options(summary_df, seed_df)
    default_comparisons = tuple(
        item for item in ("baseline (all seeds)", "counting (all seeds)")
        if item in comparison_options
    )
    if not default_comparisons:
        default_comparisons = tuple(comparison_options[:2])

    metric_selector = widgets.Dropdown(
        options=TRUE_COUNT_BUCKET_METRICS,
        value="average_reward",
        description="Metrik:",
    )
    aggregation_selector = widgets.Dropdown(
        options=[("Mittelwert über Seeds", "mean"), ("Median über Seeds", "median"), ("Gewichtet", "weighted")],
        value="mean",
        description="Aggregation:",
    )
    comparison_selector = widgets.SelectMultiple(
        options=comparison_options,
        value=default_comparisons,
        description="Vergleich:",
        layout=widgets.Layout(width="760px", height="180px"),
    )

    def plot_bucket_comparison(metric, aggregation, selected_comparisons):
        comparison_df = plot_true_count_bucket_comparison(
            summary_df=summary_df,
            seed_df=seed_df,
            buckets=buckets,
            metric=metric,
            aggregation=aggregation,
            selected_comparisons=selected_comparisons,
            summary_columns=TRUE_COUNT_BUCKET_SUMMARY_COLUMNS,
        )
        display(comparison_df)

    return widgets.interactive(
        plot_bucket_comparison,
        metric=metric_selector,
        aggregation=aggregation_selector,
        selected_comparisons=comparison_selector,
    )
