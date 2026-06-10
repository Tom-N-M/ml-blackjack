from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.evaluation import (
    TRUE_COUNT_BUCKET_METRICS,
    TRUE_COUNT_BUCKET_SUMMARY_COLUMNS,
    true_count_bucket_comparison_options,
    true_count_bucket_comparison_rows,
)


AGENT_STYLES = {
    "baseline": {"color": "#6b7c93", "label": "Baseline"},
    "counting": {"color": "#2a9d8f", "label": "Counting"},
}

DEFAULT_FIGSIZE = (8, 6)
WIDE_FIGSIZE = (10, 5)


def setup_plot_style() -> None:
    """Setzt das globale Matplotlib-Design fuer alle Projektplots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "figure.autolayout": True,
            "figure.dpi": 120,
        }
    )


def _create_figure(figsize=DEFAULT_FIGSIZE):
    setup_plot_style()
    return plt.subplots(figsize=figsize)


def save_figure(
    fig_id: str,
    output_dir: Path | str,
    fig=None,
    tight_layout: bool = True,
    fig_extension: str = "png",
    resolution: int = 300,
) -> Path:
    """Speichert die aktive oder uebergebene Figure einheitlich im Projektformat."""
    output_dir = Path(output_dir)
    fig = fig or plt.gcf()

    if tight_layout:
        fig.tight_layout()

    path = output_dir / f"{fig_id}.{fig_extension}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=fig_extension, dpi=resolution, bbox_inches="tight")
    return path


def make_figure_saver(output_dir: Path | str):
    """Erzeugt einen Notebook-freundlichen Speicher-Callback."""

    def _save_fig(
        fig_id: str,
        tight_layout: bool = True,
        fig_extension: str = "png",
        resolution: int = 300,
        fig=None,
    ) -> Path:
        return save_figure(
            fig_id=fig_id,
            output_dir=output_dir,
            fig=fig,
            tight_layout=tight_layout,
            fig_extension=fig_extension,
            resolution=resolution,
        )

    return _save_fig


def _finish_plot(fig, fig_id: str | None, save_fig_func=None, show: bool = False, close: bool = True):
    fig.tight_layout()
    saved_path = None
    if save_fig_func and fig_id:
        try:
            saved_path = save_fig_func(fig_id, fig=fig)
        except TypeError:
            plt.figure(fig.number)
            saved_path = save_fig_func(fig_id)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return saved_path


def display_saved_images(image_paths, title: str | None = None, max_images: int | None = None) -> None:
    """Zeigt gespeicherte PNGs im Notebook, ohne Matplotlib-Figures offen zu halten."""
    from IPython.display import Image, Markdown, display

    paths = [Path(path) for path in image_paths if path is not None]
    if max_images is not None:
        paths = paths[:max_images]

    if title:
        display(Markdown(f"### {title}"))

    for path in paths:
        display(Image(filename=str(path)))


def get_moving_stats(values, window: int):
    values = np.asarray(values).flatten()
    if len(values) == 0:
        return values, values, np.arange(0)

    min_periods = max(1, window // 10)
    series = pd.Series(values)
    means = series.rolling(window=window, min_periods=min_periods).mean().to_numpy()
    stds = series.rolling(window=window, min_periods=min_periods).std().to_numpy()
    return means, np.nan_to_num(stds), np.arange(len(values))


def _group_agents_by_type(agents: dict, split_agent_name_func) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for name, agent in agents.items():
        agent_type, seed = split_agent_name_func(name)
        grouped.setdefault(agent_type, []).append((seed, agent))
    return grouped


def _plot_training_curve(
    agents: dict,
    split_agent_name_func,
    value_attr: str,
    window: int,
    title: str,
    ylabel: str,
    fig_id: str,
    agent_styles: dict | None = None,
    save_fig_func=None,
    show: bool = False,
    close: bool = True,
):
    agent_styles = agent_styles or AGENT_STYLES
    grouped = _group_agents_by_type(agents, split_agent_name_func)
    fig, ax = _create_figure()

    for agent_type, agent_list in grouped.items():
        style = agent_styles[agent_type]
        curves = []

        for _, agent in agent_list:
            values = np.asarray(getattr(agent, value_attr))
            means, _, _ = get_moving_stats(values, window)
            if len(means) > 0:
                curves.append(means)

        if not curves:
            continue

        min_len = min(len(curve) for curve in curves)
        aligned_curves = np.array([curve[:min_len] for curve in curves])
        curve_mean = aligned_curves.mean(axis=0)
        curve_std = aligned_curves.std(axis=0)
        x_values = np.arange(len(curve_mean))

        ax.plot(
            x_values,
            curve_mean,
            label=style["label"],
            color=style["color"],
            linewidth=2.5,
        )
        ax.fill_between(
            x_values,
            curve_mean - curve_std,
            curve_mean + curve_std,
            color=style["color"],
            alpha=0.2,
        )

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend()

    return _finish_plot(fig, fig_id, save_fig_func=save_fig_func, show=show, close=close)


def plot_training_rewards(
    agents: dict,
    split_agent_name_func,
    window: int = 1000,
    agent_styles: dict | None = None,
    save_fig_func=None,
    show: bool = False,
    close: bool = True,
):
    return _plot_training_curve(
        agents=agents,
        split_agent_name_func=split_agent_name_func,
        value_attr="episode_rewards",
        window=window,
        title="Episode Rewards (Mittelwert ueber Seeds)",
        ylabel="Reward",
        fig_id="training/training_results_rewards",
        agent_styles=agent_styles,
        save_fig_func=save_fig_func,
        show=show,
        close=close,
    )


def plot_training_td_error(
    agents: dict,
    split_agent_name_func,
    window: int = 5000,
    agent_styles: dict | None = None,
    save_fig_func=None,
    show: bool = False,
    close: bool = True,
):
    return _plot_training_curve(
        agents=agents,
        split_agent_name_func=split_agent_name_func,
        value_attr="training_error",
        window=window,
        title="TD Error (Mittelwert ueber Seeds)",
        ylabel="TD Error",
        fig_id="training/training_results_td_error",
        agent_styles=agent_styles,
        save_fig_func=save_fig_func,
        show=show,
        close=close,
    )


def plot_evaluation_metrics_with_cis(
    greedy_eval_df: pd.DataFrame,
    agent_styles: dict,
    split_agent_name_func,
    save_fig_func=None,
    z_score: float = 1.96,
    show: bool = False,
    close: bool = True,
):
    """
    Erstellt je Metrik eine eigene Figure fuer Win-, Loss-, Push-Rate und Average Reward.
    """
    df = greedy_eval_df.copy().reset_index().rename(columns={"index": "agent_name"})
    name_parts = df["agent_name"].apply(
        lambda name: pd.Series(split_agent_name_func(name), index=["agent_type", "seed"])
    )
    df = pd.concat([df, name_parts], axis=1)

    metric_columns = ["win_rate", "loss_rate", "push_rate", "average_reward", "std_reward", "episodes"]
    for column in metric_columns:
        df[column] = pd.to_numeric(df[column])

    grouped = (
        df.groupby("agent_type")
        .agg(
            {
                "episodes": "sum",
                "win_rate": "mean",
                "loss_rate": "mean",
                "push_rate": "mean",
                "average_reward": "mean",
                "std_reward": "mean",
            }
        )
        .reset_index()
    )

    grouped["win_ci"] = z_score * np.sqrt(grouped["win_rate"] * (1 - grouped["win_rate"]) / grouped["episodes"])
    grouped["loss_ci"] = z_score * np.sqrt(grouped["loss_rate"] * (1 - grouped["loss_rate"]) / grouped["episodes"])
    grouped["push_ci"] = z_score * np.sqrt(grouped["push_rate"] * (1 - grouped["push_rate"]) / grouped["episodes"])
    grouped["avg_reward_ci"] = z_score * (grouped["std_reward"] / np.sqrt(grouped["episodes"]))

    metric_plot_config = [
        ("win_rate", "win_ci", "Win Rate", (0, 0.6)),
        ("loss_rate", "loss_ci", "Loss Rate", (0, 0.7)),
        ("push_rate", "push_ci", "Push Rate", (0, 0.2)),
        ("average_reward", "avg_reward_ci", "Average Reward", None),
    ]

    agent_types = [agent_type for agent_type in ["baseline", "counting"] if agent_type in grouped["agent_type"].unique()]
    x_values = np.arange(len(agent_types))
    colors = [agent_styles[agent_type]["color"] for agent_type in agent_types]
    labels = [agent_styles[agent_type]["label"] for agent_type in agent_types]
    plot_paths = {}

    def grouped_value(agent_type, column):
        return float(grouped[grouped["agent_type"].eq(agent_type)][column].iloc[0])

    for metric, ci_col, title, ylim in metric_plot_config:
        fig, ax = _create_figure()
        means = [grouped_value(agent_type, metric) for agent_type in agent_types]
        cis = [grouped_value(agent_type, ci_col) for agent_type in agent_types]

        ax.bar(
            x_values,
            means,
            yerr=cis,
            color=colors,
            width=0.45,
            capsize=8,
            alpha=0.85,
            edgecolor="black",
        )

        for index, agent_type in enumerate(agent_types):
            values = df[df["agent_type"].eq(agent_type)][metric].astype(float).tolist()
            jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(values), index) + jitter,
                np.asarray(values),
                color="black",
                edgecolor="white",
                s=45,
                zorder=3,
                alpha=0.8,
            )

        ax.set_title(f"{title} (Greedy Evaluation mit 95% CI)")
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        if ylim is not None:
            ax.set_ylim(*ylim)

        for index, value in enumerate(means):
            ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontweight="bold")

        saved_path = _finish_plot(
            fig,
            f"evaluation/final_evaluation_{metric}_with_ci",
            save_fig_func=save_fig_func,
            show=show,
            close=close,
        )
        plot_paths[metric] = saved_path

    return plot_paths


def plot_true_count_bucket_comparison(
    summary_df,
    seed_df,
    buckets,
    metric,
    aggregation,
    selected_comparisons,
    summary_columns=None,
    save_fig_func=None,
    show: bool = False,
    close: bool = True,
):
    summary_columns = summary_columns or TRUE_COUNT_BUCKET_SUMMARY_COLUMNS
    df = true_count_bucket_comparison_rows(summary_df, seed_df, selected_comparisons, aggregation)
    if df.empty:
        print(f"Keine Auswahl getroffen fuer Metrik: {metric}.")
        return pd.DataFrame(columns=summary_columns)

    bucket_order = [label for label, _, _ in buckets]
    df = df.copy()
    df["bucket"] = pd.Categorical(df["bucket"], categories=bucket_order, ordered=True)
    df = df.sort_values(["bucket", "comparison_id"])
    pivot = df.pivot(index="bucket", columns="comparison_id", values=metric).reindex(bucket_order)

    fig, ax = _create_figure(WIDE_FIGSIZE)
    pivot.plot(kind="bar", width=0.82, ax=ax)

    ax.set_title(f"{metric} nach True-Count-Bucket ({aggregation})", fontsize=12, fontweight="bold")
    ax.set_xlabel("True-Count-Bucket")
    ax.set_ylabel(metric)
    ax.legend(title="Agent / Checkpoint", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    saved_path = _finish_plot(
        fig,
        f"true_count/{aggregation}/{metric}",
        save_fig_func=save_fig_func,
        show=show,
        close=close,
    )
    return {"data": df[summary_columns], "path": saved_path}


def plot_all_true_count_bucket_metrics(
    summary_df,
    seed_df,
    buckets,
    aggregation="mean",
    selected_comparisons=None,
    metrics_to_plot=None,
    save_fig_func=None,
    show: bool = False,
    close: bool = True,
):
    """
    Erstellt fuer jede True-Count-Bucket-Metrik eine eigene Figure.
    """
    if selected_comparisons is None:
        comparison_options = true_count_bucket_comparison_options(summary_df, seed_df)
        selected_comparisons = [
            item
            for item in ("baseline (all seeds)", "counting (all seeds)")
            if item in comparison_options
        ]
        if not selected_comparisons and comparison_options:
            selected_comparisons = comparison_options[:2]

    if metrics_to_plot is None:
        metrics_to_plot = TRUE_COUNT_BUCKET_METRICS

    metric_dfs = {}
    print(f"Generiere Plots fuer Aggregations-Typ: '{aggregation}'")
    print(f"Verglichene Agenten/Checkpoints: {selected_comparisons}\n" + "-" * 50)

    for metric in metrics_to_plot:
        metric_dfs[metric] = plot_true_count_bucket_comparison(
            summary_df=summary_df,
            seed_df=seed_df,
            buckets=buckets,
            metric=metric,
            aggregation=aggregation,
            selected_comparisons=selected_comparisons,
            summary_columns=TRUE_COUNT_BUCKET_SUMMARY_COLUMNS,
            save_fig_func=save_fig_func,
            show=show,
            close=close,
        )

    return metric_dfs


def get_basic_strategy_action(player_total: int, dealer_upcard: int, usable_ace: int) -> int:
    strategy_dealer_value = 11 if dealer_upcard == 1 else dealer_upcard

    if usable_ace:
        if player_total >= 19:
            return 0
        if player_total == 18:
            return 0 if strategy_dealer_value <= 8 else 1
        return 1

    if player_total >= 17:
        return 0
    if 13 <= player_total <= 16:
        return 0 if strategy_dealer_value <= 6 else 1
    if player_total == 12:
        return 0 if 4 <= strategy_dealer_value <= 6 else 1
    return 1


def extract_agent_data(
    agent,
    agent_type: str,
    player_total: int,
    dealer_upcard: int,
    usable_ace: int,
    true_count: int = 0,
    counting_index=None,
):
    q_table = getattr(agent, "q_values", getattr(agent, "Q", None))
    if q_table is None:
        raise ValueError("Der Agent besitzt keine Q-Tabelle.")

    if agent_type == "baseline":
        state = (player_total, dealer_upcard, usable_ace)
        values = q_table.get(state)
        if values is None:
            return 0, np.array([np.nan, np.nan])
        return int(np.argmax(values)), np.asarray(values, dtype=float)

    if counting_index is None:
        counting_index = _build_counting_q_index(q_table)

    values = counting_index.get((player_total, dealer_upcard, usable_ace, true_count))
    if values is None:
        available_true_counts = [
            key[3]
            for key in counting_index
            if key[:3] == (player_total, dealer_upcard, usable_ace)
        ]
        if not available_true_counts:
            return 0, np.array([np.nan, np.nan])
        nearest_true_count = min(available_true_counts, key=lambda value: abs(value - true_count))
        values = counting_index[(player_total, dealer_upcard, usable_ace, nearest_true_count)]

    return int(np.argmax(values)), values


def _build_counting_q_index(q_table) -> dict:
    """Aggregiert Counting-Q-Werte ueber Running Count und Deck-Bucket."""
    grouped_values = {}
    for state, values in q_table.items():
        if len(state) != 6:
            continue
        player_total, dealer_upcard, usable_ace, _, true_count, _ = state
        key = (player_total, dealer_upcard, usable_ace, true_count)
        grouped_values.setdefault(key, []).append(np.asarray(values, dtype=float))

    return {
        key: np.mean(values, axis=0)
        for key, values in grouped_values.items()
    }


def _policy_matrices(agent, agent_type: str, true_count: int):
    player_totals = np.arange(12, 22)
    dealer_state_values = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 1])
    dealer_plot_positions = np.arange(2, 12)
    shape = (len(player_totals), len(dealer_state_values))
    matrices = {
        "policy_hard": np.zeros(shape),
        "policy_soft": np.zeros(shape),
        "q_diff_hard": np.full(shape, np.nan),
        "q_diff_soft": np.full(shape, np.nan),
        "match_basic_hard": np.zeros(shape),
        "match_basic_soft": np.zeros(shape),
    }

    q_table = getattr(agent, "q_values", getattr(agent, "Q", None))
    if not q_table:
        raise ValueError("Die Q-Tabelle ist leer. Bitte den Agenten zuerst trainieren oder laden.")
    counting_index = _build_counting_q_index(q_table) if agent_type == "counting" else None

    for i, player_total in enumerate(player_totals):
        for j, dealer_upcard in enumerate(dealer_state_values):
            act_hard, val_hard = extract_agent_data(
                agent,
                agent_type,
                player_total,
                int(dealer_upcard),
                0,
                true_count,
                counting_index=counting_index,
            )
            matrices["policy_hard"][i, j] = act_hard
            matrices["q_diff_hard"][i, j] = val_hard[1] - val_hard[0]
            matrices["match_basic_hard"][i, j] = (
                1 if act_hard == get_basic_strategy_action(player_total, dealer_upcard, 0) else 0
            )

            act_soft, val_soft = extract_agent_data(
                agent,
                agent_type,
                player_total,
                int(dealer_upcard),
                1,
                true_count,
                counting_index=counting_index,
            )
            matrices["policy_soft"][i, j] = act_soft
            matrices["q_diff_soft"][i, j] = val_soft[1] - val_soft[0]
            matrices["match_basic_soft"][i, j] = (
                1 if act_soft == get_basic_strategy_action(player_total, dealer_upcard, 1) else 0
            )

    finite_q_values = np.isfinite(matrices["q_diff_hard"]).sum() + np.isfinite(matrices["q_diff_soft"]).sum()
    if finite_q_values == 0:
        raise ValueError(
            f"Keine passenden Q-Zustaende fuer Agenttyp '{agent_type}' und True Count {true_count} gefunden."
        )

    return player_totals, dealer_plot_positions, matrices


def _setup_blackjack_matrix_axis(ax, player_totals, dealer_cards) -> None:
    ax.set_xticks(dealer_cards)
    ax.set_xticklabels([str(card) if card < 11 else "A" for card in dealer_cards])
    ax.set_yticks(player_totals)
    ax.set_ylabel("Player Hand Total")
    ax.set_xlabel("Dealer Upcard")
    ax.grid(False)


def _plot_policy_map(data, title, player_totals, dealer_cards, legend_handles, fig_id, save_fig_func, show, close):
    fig, ax = _create_figure()
    extent = [1.5, 11.5, 11.5, 21.5]
    cmap_policy = mcolors.ListedColormap(["#e74c3c", "#2ecc71"])
    ax.imshow(data, cmap=cmap_policy, origin="lower", extent=extent, aspect="auto")

    for i, player_total in enumerate(player_totals):
        for j, dealer_upcard in enumerate(dealer_cards):
            label = "H" if data[i, j] == 1 else "S"
            ax.text(dealer_upcard, player_total, label, ha="center", va="center", color="white", weight="bold")

    ax.set_title(title)
    _setup_blackjack_matrix_axis(ax, player_totals, dealer_cards)
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    return _finish_plot(fig, fig_id, save_fig_func=save_fig_func, show=show, close=close)


def _plot_heatmap(data, title, player_totals, dealer_cards, cmap, legend_handles, fig_id, save_fig_func, show, close):
    fig, ax = _create_figure()
    extent = [1.5, 11.5, 11.5, 21.5]
    image = ax.imshow(data, cmap=cmap, origin="lower", extent=extent, aspect="auto")
    ax.set_title(title)
    _setup_blackjack_matrix_axis(ax, player_totals, dealer_cards)

    if legend_handles is None:
        fig.colorbar(image, ax=ax)
    else:
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

    return _finish_plot(fig, fig_id, save_fig_func=save_fig_func, show=show, close=close)


def plot_policy_and_q_values(
    agent,
    agent_name: str,
    split_agent_name_func,
    save_fig_func=None,
    true_count: int = 0,
    show: bool = False,
    close: bool = True,
):
    """
    Erstellt Policy-, Q-Value- und Basic-Strategy-Vergleiche als einzelne Figures.
    """
    agent_type, _ = split_agent_name_func(agent_name)
    player_totals, dealer_cards, matrices = _policy_matrices(agent, agent_type, true_count)
    base_id = f"policy/{agent_name}"
    title_suffix = ""
    if agent_type == "counting":
        base_id = f"{base_id}/tc_{true_count}"
        title_suffix = f" | True Count: {true_count}"

    action_legend = [
        mpatches.Patch(color="#e74c3c", label="Stick / Halten (S)"),
        mpatches.Patch(color="#2ecc71", label="Hit / Ziehen (H)"),
    ]
    match_legend = [
        mpatches.Patch(color="#d35400", label="Abweichung"),
        mpatches.Patch(color="#27ae60", label="Konform"),
    ]
    cmap_match = mcolors.ListedColormap(["#d35400", "#27ae60"])

    plot_paths = {
        "policy_hard": _plot_policy_map(
            matrices["policy_hard"],
            f"Policy Map - Hard Totals\nAgent: {agent_name}{title_suffix}",
            player_totals,
            dealer_cards,
            action_legend,
            f"{base_id}/policy_hard",
            save_fig_func,
            show,
            close,
        ),
        "policy_soft": _plot_policy_map(
            matrices["policy_soft"],
            f"Policy Map - Soft Totals\nAgent: {agent_name}{title_suffix}",
            player_totals,
            dealer_cards,
            action_legend,
            f"{base_id}/policy_soft",
            save_fig_func,
            show,
            close,
        ),
        "q_diff_hard": _plot_heatmap(
            matrices["q_diff_hard"],
            f"Q-Value Confidence ($Q(Hit) - Q(Stand)$) - Hard\nAgent: {agent_name}{title_suffix}",
            player_totals,
            dealer_cards,
            "RdYlGn",
            None,
            f"{base_id}/q_diff_hard",
            save_fig_func,
            show,
            close,
        ),
        "q_diff_soft": _plot_heatmap(
            matrices["q_diff_soft"],
            f"Q-Value Confidence ($Q(Hit) - Q(Stand)$) - Soft\nAgent: {agent_name}{title_suffix}",
            player_totals,
            dealer_cards,
            "RdYlGn",
            None,
            f"{base_id}/q_diff_soft",
            save_fig_func,
            show,
            close,
        ),
        "basic_strategy_hard": _plot_heatmap(
            matrices["match_basic_hard"],
            f"Vergleich mit Basic Strategy (Hard)\nAgent: {agent_name}{title_suffix}",
            player_totals,
            dealer_cards,
            cmap_match,
            match_legend,
            f"{base_id}/basic_strategy_hard",
            save_fig_func,
            show,
            close,
        ),
        "basic_strategy_soft": _plot_heatmap(
            matrices["match_basic_soft"],
            f"Vergleich mit Basic Strategy (Soft)\nAgent: {agent_name}{title_suffix}",
            player_totals,
            dealer_cards,
            cmap_match,
            match_legend,
            f"{base_id}/basic_strategy_soft",
            save_fig_func,
            show,
            close,
        ),
    }

    return plot_paths


def plot_all_agent_policy_and_q_values(
    agents: dict,
    split_agent_name_func,
    save_fig_func,
    agent_names=None,
    counting_true_counts=(-3, 0, 3),
    baseline_true_count: int = 0,
    show: bool = False,
    close: bool = True,
) -> dict:
    """
    Speichert Policy-, Q-Value- und Basic-Strategy-Plots fuer ausgewaehlte Agenten.

    Baseline-Agenten haben keinen True-Count-State und werden deshalb genau einmal
    geplottet. Counting-Agenten werden fuer jeden uebergebenen True Count geplottet.
    Ohne ``agent_names`` werden alle Agenten verarbeitet.
    """
    all_paths = {}
    selected_names = list(agent_names) if agent_names is not None else list(agents)

    missing_names = [name for name in selected_names if name not in agents]
    if missing_names:
        raise KeyError(f"Unbekannte Agenten fuer Policy-Plots: {missing_names}")

    for agent_name in selected_names:
        agent = agents[agent_name]
        agent_type, _ = split_agent_name_func(agent_name)
        true_counts = [baseline_true_count] if agent_type == "baseline" else list(counting_true_counts)

        for true_count in true_counts:
            key = (agent_name, true_count)
            all_paths[key] = plot_policy_and_q_values(
                agent=agent,
                agent_name=agent_name,
                split_agent_name_func=split_agent_name_func,
                save_fig_func=save_fig_func,
                true_count=true_count,
                show=show,
                close=close,
            )

    return all_paths


def select_policy_preview_paths(
    policy_paths: dict,
    preferred_true_count: int = 0,
    plot_types=("policy_hard", "q_diff_hard"),
) -> list[Path]:
    """Waehlt eine kleine Vorschau, waehrend alle Policy-Plots gespeichert bleiben."""
    selected_paths = []
    selected_agent_types = set()

    for (agent_name, true_count), paths_by_type in policy_paths.items():
        agent_type = agent_name.rsplit("-", 1)[0]
        if agent_type in selected_agent_types:
            continue
        if agent_type == "counting" and true_count != preferred_true_count:
            continue

        selected_paths.extend(
            paths_by_type[plot_type]
            for plot_type in plot_types
            if paths_by_type.get(plot_type) is not None
        )
        selected_agent_types.add(agent_type)

    return selected_paths
