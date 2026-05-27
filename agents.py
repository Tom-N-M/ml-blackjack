import collections
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager
import os
import pickle
import queue
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from tqdm.auto import tqdm

from blackjack import (
    ACTION_DOUBLE,
    ACTION_HIT,
    ACTION_SPLIT,
    ACTION_STAND,
    ACTION_SURRENDER,
    BlackjackEnv,
)


ACTIONS = (
    ACTION_HIT,
    ACTION_STAND,
    ACTION_DOUBLE,
    ACTION_SPLIT,
    ACTION_SURRENDER,
)
ACTION_NAMES = ("Hit", "Stand", "Double", "Split", "Surrender")
ACTION_SHORT_NAMES = ("H", "S", "D", "P", "R")
ACTION_COLORS = ("#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e")
CARD_RANKS = ['ace', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'jack', 'queen', 'king']
BANKROLL_BUCKET_LIMITS = (1, 2, 3, 5, 10, 20, 50, 100)
AGENT_COLORS = {
    "strategy": "#4d4d4d",
    "basic": "#1f77b4",
    "counting": "#d55e00",
    "extended": "#d55e00",
}
FALLBACK_COLORS = ("#009e73", "#cc79a7", "#0072b2", "#f0e442")
MAX_LINE_POINTS = 2_500
MAX_TREND_POINTS = 350
TRAINING_EXPORT_VERSION = 2
BASIC_STRATEGY_OVERRIDE_MARGIN = 0.75
PROGRESS_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}]"
)


@dataclass(frozen=True)
class ExperimentConfig:
  base_seed: int = 20260526
  num_decks: int = 6
  max_training_rounds: int = 5_000_000
  eval_rounds: int = 1_000_000
  alpha: float = 0.05
  gamma: float = 1.0
  epsilon_start: float = 1.0
  epsilon_min: float = 0.05
  epsilon_decay: float = 0.9995
  bet_unit: int = 10
  max_bet_units: int = 10
  training_bankroll: int = 250
  training_wallets: Optional[int] = None
  evaluation_bankroll: int = 250
  evaluation_wallets: Optional[int] = None
  bankruptcy_penalty: int = -50
  bet_policy_override_margin: float = 5.0
  blackjack_payout: float = 1.5
  stand_on_soft_17: bool = True
  allow_double: bool = True
  allow_split: bool = True
  allow_surrender: bool = True
  double_after_split: bool = True
  max_split_hands: int = 4
  hit_split_aces: bool = False
  resplit_aces: bool = False
  dealer_peek: bool = True
  shoe_penetration: float = 0.75
  basic_strategy_override_margin: float = BASIC_STRATEGY_OVERRIDE_MARGIN
  learning_curve_window: int = 2_000
  bankruptcy_rate_window: int = 2_000

  @property
  def train_seed_basic(self):
    return self.base_seed + 1

  @property
  def train_seed_extended(self):
    return self.base_seed + 2

  @property
  def test_deck_seed(self):
    return self.base_seed + 100

  @property
  def min_bet(self):
    return self.bet_unit

  @property
  def max_bet(self):
    return self.bet_unit * self.max_bet_units

  @property
  def bet_amounts(self):
    return tuple(self.bet_unit * units for units in range(1, self.max_bet_units + 1))

  def __post_init__(self):
    if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
      raise ValueError("base_seed must be an integer.")
    if self.base_seed < 0:
      raise ValueError("base_seed must be non-negative.")

    positive_int_fields = (
        "num_decks",
        "max_training_rounds",
        "eval_rounds",
        "bet_unit",
        "max_bet_units",
        "training_bankroll",
        "evaluation_bankroll",
        "max_split_hands",
        "learning_curve_window",
        "bankruptcy_rate_window",
    )
    for field_name in positive_int_fields:
      value = getattr(self, field_name)
      if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")

    for field_name in ("training_wallets", "evaluation_wallets"):
      value = getattr(self, field_name)
      if value is not None and (
          isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError(f"{field_name} must be None or a positive integer.")

    if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
      raise ValueError("epsilon_min and epsilon_start must satisfy 0 <= min <= start <= 1.")
    if not 0 < self.epsilon_decay <= 1:
      raise ValueError("epsilon_decay must be in the interval (0, 1].")
    if not 0 < self.alpha <= 1:
      raise ValueError("alpha must be in the interval (0, 1].")
    if self.gamma < 0:
      raise ValueError("gamma must be non-negative.")
    if self.bankruptcy_penalty > 0:
      raise ValueError("bankruptcy_penalty should be zero or negative.")
    if self.bet_policy_override_margin < 0:
      raise ValueError("bet_policy_override_margin must be non-negative.")
    if self.blackjack_payout <= 0:
      raise ValueError("blackjack_payout must be positive.")
    if self.basic_strategy_override_margin < 0:
      raise ValueError("basic_strategy_override_margin must be non-negative.")
    bool_fields = (
        "stand_on_soft_17",
        "allow_double",
        "allow_split",
        "allow_surrender",
        "double_after_split",
        "hit_split_aces",
        "resplit_aces",
        "dealer_peek",
    )
    for field_name in bool_fields:
      if not isinstance(getattr(self, field_name), bool):
        raise ValueError(f"{field_name} must be a boolean.")
    if not 0 < self.shoe_penetration <= 1:
      raise ValueError("shoe_penetration must be in the interval (0, 1].")


@dataclass
class TrainingResult:
  state_mode: str
  play_state_mode: str
  q_table: DefaultDict[Any, np.ndarray] = field(repr=False)
  bet_q_table: DefaultDict[Any, np.ndarray] = field(repr=False)
  money_reward: np.ndarray = field(repr=False)
  learning_reward: np.ndarray = field(repr=False)
  bet_amount: np.ndarray = field(repr=False)
  bet_units: np.ndarray = field(repr=False)
  bankruptcy_event: np.ndarray = field(repr=False)
  active_bankroll: np.ndarray = field(repr=False)
  wallet_id: np.ndarray = field(repr=False)
  epsilon: np.ndarray = field(repr=False)
  moving_avg_money_reward: np.ndarray = field(repr=False)
  moving_avg_learning_reward: np.ndarray = field(repr=False)
  moving_avg_bet: np.ndarray = field(repr=False)
  moving_bankruptcy_rate: np.ndarray = field(repr=False)
  cumulative_avg_learning_reward: np.ndarray = field(repr=False)
  rounds_played: int
  wallets_used: int
  wallet_count: Optional[int]
  wallet_bankruptcies: int
  current_budget: float
  avg_bet: float
  avg_money_reward: float
  avg_learning_reward: float
  bankruptcy_rate: float
  final_epsilon: float
  exhausted_wallets: bool


@dataclass
class EvaluationResult:
  state_mode: str
  money_reward: np.ndarray = field(repr=False)
  penalized_score: np.ndarray = field(repr=False)
  bet_amount: np.ndarray = field(repr=False)
  bet_units: np.ndarray = field(repr=False)
  bankruptcy_event: np.ndarray = field(repr=False)
  active_bankroll: np.ndarray = field(repr=False)
  total_cash: np.ndarray = field(repr=False)
  net_money_profit: np.ndarray = field(repr=False)
  wallet_id: np.ndarray = field(repr=False)
  outcomes: collections.Counter = field(repr=False)
  rounds_played: int
  wins: int
  losses: int
  draws: int
  winrate: float
  avg_bet: float
  avg_money_reward: float
  avg_penalized_score: float
  wallet_count: Optional[int]
  wallets_used: int
  wallet_bankruptcies: int
  bankruptcy_rate: float
  current_budget: float
  inactive_cash: float
  total_cash_remaining: float
  capital_injected_total: float
  net_profit_total: float
  penalty_total: float
  penalized_score_total: float
  exhausted_wallets: bool


def default_action_values():
  return np.zeros(len(ACTIONS), dtype=float)


class BetValueFactory:
  def __init__(self, action_count):
    self.action_count = action_count

  def __call__(self):
    return np.zeros(self.action_count, dtype=float)


def make_q_table():
  return collections.defaultdict(default_action_values)


def make_bet_q_table(config):
  return collections.defaultdict(BetValueFactory(config.max_bet_units))


def env_config(config):
  return {
      "num_decks": config.num_decks,
      "stand_on_soft_17": config.stand_on_soft_17,
      "min_bet": config.min_bet,
      "max_bet": config.max_bet,
      "blackjack_payout": config.blackjack_payout,
      "allow_double": config.allow_double,
      "allow_split": config.allow_split,
      "allow_surrender": config.allow_surrender,
      "double_after_split": config.double_after_split,
      "max_split_hands": config.max_split_hands,
      "hit_split_aces": config.hit_split_aces,
      "resplit_aces": config.resplit_aces,
      "dealer_peek": config.dealer_peek,
      "shoe_penetration": config.shoe_penetration,
  }


CONFIG_TABLE_ROWS = (
    ("Run size", "max_training_rounds", "Training rounds",
     "Number of Q-learning hands per agent."),
    ("Run size", "eval_rounds", "Evaluation rounds",
     "Number of deterministic test hands per agent."),
    ("Run size", "base_seed", "Base seed",
     "Root seed for training and shared evaluation shoes."),
    ("Bankroll", "training_bankroll", "Training wallet",
     "Money in one training wallet before a ruin reset."),
    ("Bankroll", "evaluation_bankroll", "Evaluation wallet",
     "Money in one test wallet before a ruin reset."),
    ("Bankroll", "training_wallets", "Training wallet limit",
     "None means unlimited training wallets."),
    ("Bankroll", "evaluation_wallets", "Evaluation wallet limit",
     "None means unlimited evaluation wallets."),
    ("Betting", "bet_unit", "Minimum bet / bet unit",
     "Smallest bet and step size for the learned bet policy."),
    ("Betting", "max_bet_units", "Maximum bet units",
     "Maximum learned bet is bet_unit * max_bet_units."),
    ("Learning", "alpha", "Learning rate",
     "Q-learning update step size."),
    ("Learning", "gamma", "Discount factor",
     "Future-reward weight inside one hand."),
    ("Learning", "epsilon_start", "Initial epsilon",
     "Exploration probability at the beginning of training."),
    ("Learning", "epsilon_min", "Minimum epsilon",
     "Exploration floor after decay."),
    ("Learning", "epsilon_decay", "Epsilon decay",
     "Multiplicative epsilon decay after each hand."),
    ("Learning", "bankruptcy_penalty", "Bankruptcy penalty",
     "Additional training/evaluation penalty when a wallet is ruined."),
    ("Learning", "bet_policy_override_margin", "Bet-policy guard",
     "Money edge required before learned bet sizing may override the baseline spread."),
    ("Learning", "basic_strategy_override_margin", "Basic-strategy guard",
     "Q-value edge required before a learned play action may override basic strategy."),
    ("Casino rules", "num_decks", "Decks per shoe",
     "Number of decks in one casino shoe."),
    ("Casino rules", "shoe_penetration", "Shoe penetration",
     "Fraction of a shoe dealt before shuffling."),
    ("Casino rules", "blackjack_payout", "Blackjack payout",
     "Natural blackjack payout multiplier."),
    ("Casino rules", "stand_on_soft_17", "Dealer stands soft 17",
     "True means S17, False means H17."),
    ("Casino rules", "dealer_peek", "Dealer peek",
     "Dealer checks blackjack with ace or 10 upcard."),
    ("Casino rules", "allow_double", "Double enabled",
     "Allow double down where legal."),
    ("Casino rules", "allow_split", "Split enabled",
     "Allow pair splitting where legal."),
    ("Casino rules", "allow_surrender", "Late surrender enabled",
     "Allow surrender before other player actions."),
    ("Casino rules", "double_after_split", "Double after split",
     "Allow doubling split hands."),
    ("Casino rules", "max_split_hands", "Max split hands",
     "Maximum player hands after splitting."),
    ("Casino rules", "hit_split_aces", "Hit split aces",
     "Allow more hits after splitting aces."),
    ("Casino rules", "resplit_aces", "Resplit aces",
     "Allow splitting aces again after a split."),
    ("Plotting", "learning_curve_window", "Learning curve window",
     "Rolling window for reward and bet plots."),
    ("Plotting", "bankruptcy_rate_window", "Ruin-rate window",
     "Rolling window for bankruptcy-rate plots."),
)


def _config_display_value(value):
  if value is None:
    return "unlimited"
  if isinstance(value, bool):
    return "yes" if value else "no"
  if isinstance(value, float):
    return f"{value:g}"
  if isinstance(value, int):
    return f"{value:,}"
  return str(value)


def config_table(config):
  rows = []
  for group, field_name, label, description in CONFIG_TABLE_ROWS:
    value = getattr(config, field_name)
    rows.append({
        "Group": group,
        "Parameter": field_name,
        "Setting": label,
        "Value": _config_display_value(value),
        "Description": description,
    })

  try:
    import pandas as pd

    frame = pd.DataFrame(rows)
    return frame[["Group", "Setting", "Parameter", "Value", "Description"]]
  except ImportError:
    return rows


def progress_bar(iterable=None, total=None, desc="", disable=False):
  return tqdm(
      iterable,
      total=total,
      desc=desc,
      disable=disable,
      dynamic_ncols=True,
      mininterval=0.2,
      maxinterval=1.0,
      smoothing=0.05,
      bar_format=PROGRESS_BAR_FORMAT,
  )


def progress_interval(total):
  return max(1, int(total) // 500)


def report_progress(progress_callback, progress_delta):
  if progress_callback is not None and progress_delta > 0:
    progress_callback(progress_delta)


def drain_progress_queue(progress_queue, bar):
  if progress_queue is None:
    return
  while True:
    try:
      bar.update(progress_queue.get_nowait())
    except queue.Empty:
      break


def make_env(state_mode, config, seed=None, bankroll=None):
  return BlackjackEnv(
      state_mode=state_mode,
      seed=seed,
      bankroll=bankroll,
      **env_config(config),
  )


def make_wallet_env(state_mode, config, base_seed, wallet_index, bankroll):
  wallet_seed = base_seed + wallet_index * 100_000
  return make_env(play_state_mode(state_mode), config, seed=wallet_seed,
                  bankroll=bankroll)


def training_seed(config, state_mode):
  if state_mode == "basic":
    return config.train_seed_basic
  if state_mode == "extended":
    return config.train_seed_extended
  raise ValueError(f"Unknown state_mode: {state_mode}")


def agent_label_from_state_mode(state_mode):
  if state_mode == "basic":
    return "Basic"
  if state_mode == "extended":
    return "Counting"
  return state_mode


def play_state_mode(state_mode):
  if state_mode in ("basic", "extended"):
    return "basic"
  raise ValueError(f"Unknown state_mode: {state_mode}")


def wallet_limit_label(wallet_count):
  return "unlimited" if wallet_count is None else str(wallet_count)


def has_wallet_available(wallet_index, wallet_count):
  return wallet_count is None or wallet_index < wallet_count


def can_afford_next_round(env, config):
  return env.bankroll is None or env.bankroll >= config.min_bet


def bankruptcy_penalty(config, is_bankrupt):
  return config.bankruptcy_penalty if is_bankrupt else 0


def learning_reward(config, money_reward, is_bankrupt):
  return money_reward + bankruptcy_penalty(config, is_bankrupt)


def choose_action(
    state,
    q_table,
    epsilon,
    rng,
    override_margin=BASIC_STRATEGY_OVERRIDE_MARGIN,
):
  legal_actions = legal_play_actions(state)
  if epsilon > 0 and rng.random() < epsilon:
    return rng.choice(legal_actions)
  return choose_greedy_action(state, q_table, override_margin=override_margin)


def choose_greedy_action(
    state,
    q_table,
    override_margin=BASIC_STRATEGY_OVERRIDE_MARGIN,
):
  legal_actions = legal_play_actions(state)
  values = q_table[state]
  learned_action = greedy_legal_index(values, legal_actions)
  strategy_action = basic_strategy_action_from_state(state)

  if strategy_action not in legal_actions:
    return learned_action
  if learned_action == strategy_action:
    return learned_action

  learned_edge = values[learned_action] - values[strategy_action]
  if learned_edge >= override_margin:
    return learned_action
  return strategy_action


def legal_play_actions(state):
  if state == ("terminal",):
    return ()
  if state[0] >= 21:
    return (ACTION_STAND,)

  if len(state) >= 6:
    _, _, _, can_double, can_split, can_surrender = state[:6]
    actions = [ACTION_HIT, ACTION_STAND]
    if can_double:
      actions.append(ACTION_DOUBLE)
    if can_split:
      actions.append(ACTION_SPLIT)
    if can_surrender:
      actions.append(ACTION_SURRENDER)
    return tuple(actions)

  return (ACTION_HIT, ACTION_STAND)


def _hard_fallback_action(player_total, dealer_card):
  if player_total <= 11:
    return ACTION_HIT
  if player_total == 12:
    return ACTION_STAND if 4 <= dealer_card <= 6 else ACTION_HIT
  if 13 <= player_total <= 16:
    return ACTION_STAND if 2 <= dealer_card <= 6 else ACTION_HIT
  return ACTION_STAND


def _double_or_fallback(action, legal_actions, player_total, usable_ace):
  if action in legal_actions:
    return action
  if usable_ace:
    if player_total <= 17:
      return ACTION_HIT if ACTION_HIT in legal_actions else ACTION_STAND
    return ACTION_STAND
  return ACTION_HIT if ACTION_HIT in legal_actions else ACTION_STAND


def _pair_value_from_state(player_total, usable_ace, can_split):
  if not can_split:
    return None
  if usable_ace and player_total == 12:
    return 11
  if player_total % 2 == 0:
    return player_total // 2
  return None


def basic_strategy_action_from_state(state):
  """Return a conservative S17/DAS/late-surrender basic-strategy action."""
  legal_actions = legal_play_actions(state)
  if not legal_actions:
    raise ValueError("No legal actions available.")

  player_total, dealer_card, usable_ace = state[:3]
  can_double = len(state) >= 6 and state[3]
  can_split = len(state) >= 6 and state[4]
  can_surrender = len(state) >= 6 and state[5]
  pair_value = _pair_value_from_state(player_total, usable_ace, can_split)

  if ACTION_SPLIT in legal_actions and pair_value is not None:
    if pair_value in (8, 11):
      return ACTION_SPLIT
    if pair_value in (2, 3, 7) and 2 <= dealer_card <= 7:
      return ACTION_SPLIT
    if pair_value == 6 and 2 <= dealer_card <= 6:
      return ACTION_SPLIT
    if pair_value == 9 and dealer_card in (2, 3, 4, 5, 6, 8, 9):
      return ACTION_SPLIT
    if pair_value == 4 and dealer_card in (5, 6):
      return ACTION_SPLIT

  if can_surrender and ACTION_SURRENDER in legal_actions and not usable_ace:
    if player_total == 16 and dealer_card in (9, 10, 11):
      return ACTION_SURRENDER
    if player_total == 15 and dealer_card == 10:
      return ACTION_SURRENDER

  if usable_ace:
    if can_double and ACTION_DOUBLE in legal_actions:
      if player_total in (13, 14) and dealer_card in (5, 6):
        return ACTION_DOUBLE
      if player_total in (15, 16) and dealer_card in (4, 5, 6):
        return ACTION_DOUBLE
      if player_total == 17 and dealer_card in (3, 4, 5, 6):
        return ACTION_DOUBLE
      if player_total == 18 and dealer_card in (3, 4, 5, 6):
        return ACTION_DOUBLE
    if player_total <= 17:
      return ACTION_HIT if ACTION_HIT in legal_actions else ACTION_STAND
    if player_total == 18 and dealer_card in (9, 10, 11):
      return ACTION_HIT if ACTION_HIT in legal_actions else ACTION_STAND
    return ACTION_STAND

  if can_double and ACTION_DOUBLE in legal_actions:
    if player_total == 11 and dealer_card != 11:
      return ACTION_DOUBLE
    if player_total == 10 and 2 <= dealer_card <= 9:
      return ACTION_DOUBLE
    if player_total == 9 and 3 <= dealer_card <= 6:
      return ACTION_DOUBLE

  action = _hard_fallback_action(player_total, dealer_card)
  if action in legal_actions:
    return action
  return ACTION_STAND


def bankroll_units(env, config):
  if env.bankroll is None:
    return config.max_bet_units
  return int(env.bankroll // config.bet_unit)


def bankroll_bucket_from_units(units):
  for limit in BANKROLL_BUCKET_LIMITS:
    if units <= limit:
      return limit
  return BANKROLL_BUCKET_LIMITS[-1]


def bankroll_bucket_label(bucket):
  previous = 0
  for limit in BANKROLL_BUCKET_LIMITS:
    if bucket == limit:
      if limit <= 3:
        return f"{limit}u"
      return f"{previous + 1}-{limit}u"
    previous = limit
  return f"{bucket}u"


def betting_state(env, config, state_mode):
  state = (bankroll_bucket_from_units(bankroll_units(env, config)),)
  if state_mode == "extended":
    state = state + (env._get_true_count_bucket(),)
  return state


def available_bet_indices(env, config):
  max_units = config.max_bet_units
  if env.bankroll is not None:
    max_units = min(max_units, bankroll_units(env, config))
  return tuple(range(max(0, max_units)))


def available_bet_amounts(env, config):
  return tuple(bet_amount_from_index(index, config)
               for index in available_bet_indices(env, config))


def bet_amount_from_index(index, config):
  return (index + 1) * config.bet_unit


def count_spread_units(true_count, max_bet_units):
  if true_count <= 1:
    return 1
  if true_count == 2:
    return min(2, max_bet_units)
  if true_count == 3:
    return min(4, max_bet_units)
  if true_count == 4:
    return min(6, max_bet_units)
  return min(8, max_bet_units)


def baseline_bet_index(state, config, legal_indices, state_mode):
  if state_mode == "extended" and len(state) >= 2:
    target_units = count_spread_units(state[1], config.max_bet_units)
  else:
    target_units = 1

  target_index = target_units - 1
  affordable_indices = [index for index in legal_indices if index <= target_index]
  if affordable_indices:
    return max(affordable_indices)
  return min(legal_indices)


def greedy_legal_index(values, legal_indices):
  if not legal_indices:
    raise ValueError("No legal actions available.")
  return max(legal_indices, key=lambda index: (values[index], -index))


def choose_legal_index(values, legal_indices, epsilon, rng):
  if not legal_indices:
    raise ValueError("No legal actions available.")
  if epsilon > 0 and rng.random() < epsilon:
    return rng.choice(legal_indices)
  return greedy_legal_index(values, legal_indices)


def choose_bet_index(
    state,
    bet_q_table,
    legal_indices,
    epsilon,
    rng,
    config=None,
    state_mode="basic",
):
  if epsilon > 0 and rng.random() < epsilon:
    return rng.choice(legal_indices)
  return choose_greedy_bet_index(
      state,
      bet_q_table,
      legal_indices,
      config=config,
      state_mode=state_mode,
  )


def choose_greedy_bet_index(
    state,
    bet_q_table,
    legal_indices,
    config=None,
    state_mode="basic",
):
  values = bet_q_table[state]
  learned_index = greedy_legal_index(values, legal_indices)
  if config is None:
    return learned_index

  baseline_index = baseline_bet_index(
      state,
      config,
      legal_indices,
      state_mode,
  )
  learned_edge = values[learned_index] - values[baseline_index]
  if learned_edge >= config.bet_policy_override_margin:
    return learned_index
  return baseline_index


def moving_average(values, window):
  values = np.asarray(values, dtype=float)
  if len(values) == 0:
    return np.array([], dtype=np.float32)

  window = max(1, min(int(window), len(values)))
  cumsum = np.cumsum(np.insert(values, 0, 0.0))
  return ((cumsum[window:] - cumsum[:-window]) / window).astype(np.float32)


def moving_average_rounds(values, window):
  if len(values) == 0:
    return np.array([], dtype=int)
  effective_window = max(1, min(int(window), len(values)))
  return np.arange(effective_window, len(values) + 1)


def agent_color(label, index=0):
  normalized = str(label).lower()
  for key, color in AGENT_COLORS.items():
    if key in normalized:
      return color
  return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def downsample_series(x_values, y_values, max_points=MAX_LINE_POINTS):
  x_values = np.asarray(x_values, dtype=float)
  y_values = np.asarray(y_values, dtype=float)
  mask = np.isfinite(x_values) & np.isfinite(y_values)
  x = x_values[mask]
  y = y_values[mask]
  if len(x) <= max_points:
    return x, y

  indices = np.unique(np.linspace(0, len(x) - 1, max_points, dtype=int))
  return x[indices], y[indices]


def binned_mean_series(x_values, y_values, max_points=MAX_TREND_POINTS):
  x_values = np.asarray(x_values, dtype=float)
  y_values = np.asarray(y_values, dtype=float)
  mask = np.isfinite(x_values) & np.isfinite(y_values)
  x = x_values[mask]
  y = y_values[mask]
  if len(x) <= max_points:
    return x, y

  edges = np.linspace(0, len(x), max_points + 1, dtype=int)
  binned_x = []
  binned_y = []
  for start, end in zip(edges[:-1], edges[1:]):
    if start == end:
      continue
    binned_x.append(float(np.mean(x[start:end])))
    binned_y.append(float(np.mean(y[start:end])))
  return np.asarray(binned_x), np.asarray(binned_y)


def smoothed_trend_series(x_values, y_values, max_points=MAX_TREND_POINTS):
  x, y = binned_mean_series(x_values, y_values, max_points=max_points)
  if len(x) < 3:
    return x, y

  window = max(3, min(len(y), len(y) // 20))
  if window % 2 == 0:
    window += 1
  trend_y = moving_average(y, window)
  trend_x = moving_average(x, window)
  return trend_x, trend_y


def plot_metric_series(
    ax,
    x_values,
    y_values,
    label,
    color,
    linewidth=1.7,
    alpha=0.95,
    show_trend=True,
):
  x_plot, y_plot = downsample_series(x_values, y_values)
  ax.plot(
      x_plot,
      y_plot,
      color=color,
      linewidth=linewidth,
      alpha=alpha,
      label=label,
  )
  if show_trend:
    trend_x, trend_y = smoothed_trend_series(x_values, y_values)
    if len(trend_x) >= 3:
      ax.plot(
          trend_x,
          trend_y,
          color=color,
          linestyle="--",
          linewidth=2.4,
          alpha=0.8,
          label="_nolegend_",
      )


def build_shuffled_shoe(rng, num_decks):
  shoe = CARD_RANKS * 4 * num_decks
  rng.shuffle(shoe)
  return shoe


class SharedTestShoes:
  def __init__(self, config):
    self.rounds = config.eval_rounds
    self.num_decks = config.num_decks
    self.seed = config.test_deck_seed

  def __len__(self):
    return self.rounds

  def __bool__(self):
    return self.rounds > 0

  def __getitem__(self, index):
    if index < 0:
      index += self.rounds
    if index < 0 or index >= self.rounds:
      raise IndexError(index)

    rng = random.Random(self.seed + index)
    return build_shuffled_shoe(rng, self.num_decks)

  def __eq__(self, other):
    return (
        isinstance(other, SharedTestShoes) and
        self.rounds == other.rounds and
        self.num_decks == other.num_decks and
        self.seed == other.seed
    )


def build_test_decks(config):
  return SharedTestShoes(config)


def load_test_deck(env, deck):
  env.deck = list(deck)
  env.played_cards = []
  env.cards_dealt = 0


def update_q_value(config, q_table, state, action, next_state, reward, done):
  if done:
    next_value = 0
  else:
    next_action = choose_greedy_action(
        next_state,
        q_table,
        override_margin=config.basic_strategy_override_margin,
    )
    next_value = q_table[next_state][next_action]
  td_target = reward + config.gamma * next_value
  q_table[state][action] += config.alpha * (td_target - q_table[state][action])


def update_immediate_q_value(config, q_table, state, action, reward):
  q_table[state][action] += config.alpha * (reward - q_table[state][action])


def play_training_round(
    env,
    config,
    q_table,
    bet_q_table,
    epsilon,
    policy_rng,
    state_mode,
    train_play_policy=True,
):
  bet_state = betting_state(env, config, state_mode)
  legal_bets = available_bet_indices(env, config)
  bet_index = choose_bet_index(
      bet_state,
      bet_q_table,
      legal_bets,
      epsilon,
      policy_rng,
      config=config,
      state_mode=state_mode,
  )
  bet_amount = bet_amount_from_index(bet_index, config)
  state = env.reset(bet=bet_amount)
  done = env.done
  money_reward = env.pending_reward if done else 0

  while not done:
    if train_play_policy:
      action = choose_action(
          state,
          q_table,
          epsilon,
          policy_rng,
          override_margin=config.basic_strategy_override_margin,
      )
    else:
      action = choose_greedy_action(
          state,
          q_table,
          override_margin=config.basic_strategy_override_margin,
      )
    next_state, money_reward, done = env.step(action)
    reward_per_unit = money_reward / bet_amount
    if train_play_policy:
      update_q_value(
          config,
          q_table,
          state,
          action,
          next_state,
          reward_per_unit,
          done,
      )
    state = next_state

  is_bankrupt = not can_afford_next_round(env, config)
  round_learning_reward = learning_reward(config, money_reward, is_bankrupt)
  update_immediate_q_value(config, bet_q_table, bet_state, bet_index,
                           round_learning_reward)
  return {
      "money_reward": money_reward,
      "learning_reward": round_learning_reward,
      "bet_amount": bet_amount,
      "bet_units": bet_index + 1,
      "is_bankrupt": is_bankrupt,
      "bankroll": env.bankroll,
  }


def _new_training_buffers(rounds):
  return {
      "money_reward": np.full(rounds, np.nan, dtype=np.float32),
      "learning_reward": np.full(rounds, np.nan, dtype=np.float32),
      "bet_amount": np.full(rounds, np.nan, dtype=np.float32),
      "bet_units": np.full(rounds, -1, dtype=np.int32),
      "bankruptcy_event": np.zeros(rounds, dtype=bool),
      "active_bankroll": np.full(rounds, np.nan, dtype=np.float32),
      "wallet_id": np.full(rounds, -1, dtype=np.int32),
      "epsilon": np.full(rounds, np.nan, dtype=np.float32),
  }


def _finalize_training_result(
    state_mode,
    config,
    q_table,
    bet_q_table,
    buffers,
    rounds_played,
    wallet_index,
    bankruptcies,
    current_budget,
    exhausted_wallets,
):
  money_rewards = buffers["money_reward"][:rounds_played]
  learning_rewards = buffers["learning_reward"][:rounds_played]
  bet_amounts = buffers["bet_amount"][:rounds_played]
  bankruptcy_events = buffers["bankruptcy_event"][:rounds_played]

  if rounds_played:
    avg_bet = float(np.mean(bet_amounts))
    avg_money_reward = float(np.mean(money_rewards))
    avg_learning_reward = float(np.mean(learning_rewards))
    bankruptcy_rate = float(np.mean(bankruptcy_events))
    cumulative_avg_learning_reward = (
        np.cumsum(learning_rewards, dtype=np.float64) /
        np.arange(1, rounds_played + 1)
    ).astype(np.float32)
    final_epsilon = float(buffers["epsilon"][rounds_played - 1])
  else:
    avg_bet = 0.0
    avg_money_reward = 0.0
    avg_learning_reward = 0.0
    bankruptcy_rate = 0.0
    cumulative_avg_learning_reward = np.array([], dtype=np.float32)
    final_epsilon = config.epsilon_start

  return TrainingResult(
      state_mode=state_mode,
      play_state_mode=play_state_mode(state_mode),
      q_table=q_table,
      bet_q_table=bet_q_table,
      money_reward=money_rewards,
      learning_reward=learning_rewards,
      bet_amount=bet_amounts,
      bet_units=buffers["bet_units"][:rounds_played],
      bankruptcy_event=bankruptcy_events,
      active_bankroll=buffers["active_bankroll"][:rounds_played],
      wallet_id=buffers["wallet_id"][:rounds_played],
      epsilon=buffers["epsilon"][:rounds_played],
      moving_avg_money_reward=moving_average(
          money_rewards, config.learning_curve_window),
      moving_avg_learning_reward=moving_average(
          learning_rewards, config.learning_curve_window),
      moving_avg_bet=moving_average(
          bet_amounts, config.learning_curve_window),
      moving_bankruptcy_rate=moving_average(
          bankruptcy_events.astype(float), config.bankruptcy_rate_window),
      cumulative_avg_learning_reward=cumulative_avg_learning_reward,
      rounds_played=rounds_played,
      wallets_used=wallet_index + 1,
      wallet_count=config.training_wallets,
      wallet_bankruptcies=bankruptcies,
      current_budget=current_budget,
      avg_bet=avg_bet,
      avg_money_reward=avg_money_reward,
      avg_learning_reward=avg_learning_reward,
      bankruptcy_rate=bankruptcy_rate,
      final_epsilon=final_epsilon,
      exhausted_wallets=exhausted_wallets,
  )


def train_agent(
    state_mode,
    config,
    show_progress=True,
    progress_callback=None,
    progress_report_interval=None,
    q_table=None,
    train_play_policy=True,
):
  seed = training_seed(config, state_mode)
  q_table = q_table if q_table is not None else make_q_table()
  bet_q_table = make_bet_q_table(config)
  policy_rng = random.Random(seed + 10_000)
  epsilon = config.epsilon_start
  wallet_index = 0
  bankruptcies = 0
  exhausted_wallets = False
  env = make_wallet_env(
      state_mode, config, seed, wallet_index, config.training_bankroll)
  buffers = _new_training_buffers(config.max_training_rounds)
  progress_report_interval = (
      progress_report_interval or progress_interval(config.max_training_rounds))
  iterator = progress_bar(
      range(config.max_training_rounds),
      desc=f"Training {agent_label_from_state_mode(state_mode)}",
      disable=not show_progress or progress_callback is not None,
  )

  rounds_played = 0
  last_progress_report = 0
  for round_index in iterator:
    if not can_afford_next_round(env, config):
      next_wallet_index = wallet_index + 1
      if not has_wallet_available(next_wallet_index, config.training_wallets):
        exhausted_wallets = True
        break
      wallet_index = next_wallet_index
      env = make_wallet_env(
          state_mode, config, seed, wallet_index, config.training_bankroll)

    result = play_training_round(
        env,
        config,
        q_table,
        bet_q_table,
        epsilon,
        policy_rng,
        state_mode,
        train_play_policy=train_play_policy,
    )
    if result["is_bankrupt"]:
      bankruptcies += 1

    buffers["money_reward"][round_index] = result["money_reward"]
    buffers["learning_reward"][round_index] = result["learning_reward"]
    buffers["bet_amount"][round_index] = result["bet_amount"]
    buffers["bet_units"][round_index] = result["bet_units"]
    buffers["bankruptcy_event"][round_index] = result["is_bankrupt"]
    buffers["active_bankroll"][round_index] = result["bankroll"]
    buffers["wallet_id"][round_index] = wallet_index + 1
    buffers["epsilon"][round_index] = epsilon
    rounds_played = round_index + 1

    epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

    if rounds_played - last_progress_report >= progress_report_interval:
      report_progress(
          progress_callback,
          rounds_played - last_progress_report,
      )
      last_progress_report = rounds_played

  report_progress(progress_callback, rounds_played - last_progress_report)

  return _finalize_training_result(
      state_mode=state_mode,
      config=config,
      q_table=q_table,
      bet_q_table=bet_q_table,
      buffers=buffers,
      rounds_played=rounds_played,
      wallet_index=wallet_index,
      bankruptcies=bankruptcies,
      current_budget=env.bankroll,
      exhausted_wallets=exhausted_wallets,
  )


def _train_agent_job(job):
  if len(job) == 3:
    label, state_mode, config = job
    progress_queue = None
    report_interval = None
  else:
    label, state_mode, config, progress_queue, report_interval = job
  progress_callback = progress_queue.put if progress_queue is not None else None
  return label, train_agent(
      state_mode,
      config,
      show_progress=False,
      progress_callback=progress_callback,
      progress_report_interval=report_interval,
  )


def train_agent_comparison(config, parallel=False, show_progress=True):
  report_interval = progress_interval(config.max_training_rounds)
  if show_progress:
    with progress_bar(
        total=config.max_training_rounds * 2,
        desc="Training agents",
    ) as bar:
      basic_result = train_agent(
          "basic",
          config,
          show_progress=False,
          progress_callback=bar.update,
          progress_report_interval=report_interval,
          train_play_policy=True,
      )
      counting_result = train_agent(
          "extended",
          config,
          show_progress=False,
          progress_callback=bar.update,
          progress_report_interval=report_interval,
          q_table=basic_result.q_table,
          train_play_policy=False,
      )
  else:
    basic_result = train_agent(
        "basic",
        config,
        show_progress=False,
        train_play_policy=True,
    )
    counting_result = train_agent(
        "extended",
        config,
        show_progress=False,
        q_table=basic_result.q_table,
        train_play_policy=False,
    )

  return {
      "Basic": basic_result,
      "Counting": counting_result,
  }


def save_training_export(training_results, config, path):
  export_path = Path(path)
  export_path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
      "version": TRAINING_EXPORT_VERSION,
      "config": asdict(config),
      "action_names": ACTION_NAMES,
      "training_results": training_results,
  }
  with export_path.open("wb") as export_file:
    pickle.dump(payload, export_file, protocol=pickle.HIGHEST_PROTOCOL)
  return export_path


def load_training_export(path):
  export_path = Path(path)
  with export_path.open("rb") as export_file:
    payload = pickle.load(export_file)

  if not isinstance(payload, dict):
    raise ValueError("Training export is not a valid payload.")
  if payload.get("version") != TRAINING_EXPORT_VERSION:
    raise ValueError("Training export version is not supported.")
  if payload.get("action_names") != ACTION_NAMES:
    raise ValueError("Training export action space does not match this code.")

  training_results = payload.get("training_results")
  if not isinstance(training_results, dict):
    raise ValueError("Training export does not contain training results.")

  config_payload = payload.get("config")
  loaded_config = (
      ExperimentConfig(**config_payload)
      if config_payload is not None
      else None
  )
  return training_results, loaded_config


def play_greedy_round(env, config, q_table, bet_q_table, state_mode):
  bet_state = betting_state(env, config, state_mode)
  legal_bets = available_bet_indices(env, config)
  bet_index = choose_greedy_bet_index(
      bet_state,
      bet_q_table,
      legal_bets,
      config=config,
      state_mode=state_mode,
  )
  bet_amount = bet_amount_from_index(bet_index, config)
  state = env.reset(bet=bet_amount)
  done = env.done
  money_reward = env.pending_reward if done else 0

  while not done:
    action = choose_greedy_action(
        state,
        q_table,
        override_margin=config.basic_strategy_override_margin,
    )
    state, money_reward, done = env.step(action)

  is_bankrupt = not can_afford_next_round(env, config)
  return {
      "money_reward": money_reward,
      "penalized_score": learning_reward(config, money_reward, is_bankrupt),
      "bet_amount": bet_amount,
      "bet_units": bet_index + 1,
      "is_bankrupt": is_bankrupt,
      "bankroll": env.bankroll,
      "outcome": env.last_outcome,
  }


def play_basic_strategy_round(env, config):
  bet_amount = min(config.min_bet, env.bankroll or config.min_bet)
  state = env.reset(bet=bet_amount)
  done = env.done
  money_reward = env.pending_reward if done else 0

  while not done:
    action = basic_strategy_action_from_state(state)
    state, money_reward, done = env.step(action)

  is_bankrupt = not can_afford_next_round(env, config)
  return {
      "money_reward": money_reward,
      "penalized_score": learning_reward(config, money_reward, is_bankrupt),
      "bet_amount": bet_amount,
      "bet_units": 1,
      "is_bankrupt": is_bankrupt,
      "bankroll": env.bankroll,
      "outcome": env.last_outcome,
  }


def _new_evaluation_buffers(rounds):
  return {
      "money_reward": np.full(rounds, np.nan, dtype=np.float32),
      "penalized_score": np.full(rounds, np.nan, dtype=np.float32),
      "bet_amount": np.full(rounds, np.nan, dtype=np.float32),
      "bet_units": np.full(rounds, -1, dtype=np.int32),
      "bankruptcy_event": np.zeros(rounds, dtype=bool),
      "active_bankroll": np.full(rounds, np.nan, dtype=np.float32),
      "total_cash": np.full(rounds, np.nan, dtype=np.float32),
      "net_money_profit": np.full(rounds, np.nan, dtype=np.float32),
      "wallet_id": np.full(rounds, -1, dtype=np.int32),
  }


def evaluate_agent(
    state_mode,
    q_table,
    bet_q_table,
    test_decks,
    config,
    show_progress=True,
    progress_callback=None,
    progress_report_interval=None,
):
  wallet_index = 0
  shoe_index = 0
  inactive_cash = 0
  bankruptcies = 0
  exhausted_wallets = False
  env = make_wallet_env(
      state_mode, config, config.test_deck_seed, wallet_index,
      config.evaluation_bankroll)
  if test_decks:
    load_test_deck(env, test_decks[shoe_index])

  buffers = _new_evaluation_buffers(len(test_decks))
  outcomes = collections.Counter()
  wins = losses = draws = 0
  progress_report_interval = (
      progress_report_interval or progress_interval(len(test_decks)))
  iterator = progress_bar(
      range(len(test_decks)),
      desc=f"Evaluating {agent_label_from_state_mode(state_mode)}",
      disable=not show_progress or progress_callback is not None,
  )

  rounds_played = 0
  last_progress_report = 0
  for round_index in iterator:
    if len(env.deck) <= (1 - config.shoe_penetration) * env.max_deck_size:
      shoe_index += 1
      if shoe_index >= len(test_decks):
        break
      load_test_deck(env, test_decks[shoe_index])

    if not can_afford_next_round(env, config):
      inactive_cash += env.bankroll
      next_wallet_index = wallet_index + 1
      if not has_wallet_available(next_wallet_index, config.evaluation_wallets):
        exhausted_wallets = True
        break
      wallet_index = next_wallet_index
      env.bankroll = config.evaluation_bankroll

    result = play_greedy_round(env, config, q_table, bet_q_table, state_mode)
    if result["is_bankrupt"]:
      bankruptcies += 1

    wallets_used = wallet_index + 1
    capital_injected = wallets_used * config.evaluation_bankroll
    total_cash = inactive_cash + env.bankroll
    net_money_profit = total_cash - capital_injected

    buffers["money_reward"][round_index] = result["money_reward"]
    buffers["penalized_score"][round_index] = result["penalized_score"]
    buffers["bet_amount"][round_index] = result["bet_amount"]
    buffers["bet_units"][round_index] = result["bet_units"]
    buffers["bankruptcy_event"][round_index] = result["is_bankrupt"]
    buffers["active_bankroll"][round_index] = env.bankroll
    buffers["total_cash"][round_index] = total_cash
    buffers["net_money_profit"][round_index] = net_money_profit
    buffers["wallet_id"][round_index] = wallets_used
    rounds_played = round_index + 1

    outcomes[result["outcome"]] += 1
    if result["money_reward"] > 0:
      wins += 1
    elif result["money_reward"] < 0:
      losses += 1
    else:
      draws += 1

    if rounds_played - last_progress_report >= progress_report_interval:
      report_progress(
          progress_callback,
          rounds_played - last_progress_report,
      )
      last_progress_report = rounds_played

  report_progress(progress_callback, rounds_played - last_progress_report)

  money_rewards = buffers["money_reward"][:rounds_played]
  penalized_scores = buffers["penalized_score"][:rounds_played]
  bet_amounts = buffers["bet_amount"][:rounds_played]
  bankruptcy_events = buffers["bankruptcy_event"][:rounds_played]
  wallets_used = wallet_index + 1
  capital_injected = wallets_used * config.evaluation_bankroll
  total_cash_remaining = inactive_cash + env.bankroll
  net_money_profit = total_cash_remaining - capital_injected
  penalty_total = bankruptcies * config.bankruptcy_penalty

  return EvaluationResult(
      state_mode=state_mode,
      money_reward=money_rewards,
      penalized_score=penalized_scores,
      bet_amount=bet_amounts,
      bet_units=buffers["bet_units"][:rounds_played],
      bankruptcy_event=bankruptcy_events,
      active_bankroll=buffers["active_bankroll"][:rounds_played],
      total_cash=buffers["total_cash"][:rounds_played],
      net_money_profit=buffers["net_money_profit"][:rounds_played],
      wallet_id=buffers["wallet_id"][:rounds_played],
      outcomes=outcomes,
      rounds_played=rounds_played,
      wins=wins,
      losses=losses,
      draws=draws,
      winrate=wins / rounds_played if rounds_played else 0,
      avg_bet=float(np.mean(bet_amounts)) if rounds_played else 0,
      avg_money_reward=float(np.mean(money_rewards)) if rounds_played else 0,
      avg_penalized_score=float(np.mean(penalized_scores)) if rounds_played else 0,
      wallet_count=config.evaluation_wallets,
      wallets_used=wallets_used,
      wallet_bankruptcies=bankruptcies,
      bankruptcy_rate=float(np.mean(bankruptcy_events)) if rounds_played else 0,
      current_budget=env.bankroll,
      inactive_cash=inactive_cash,
      total_cash_remaining=total_cash_remaining,
      capital_injected_total=capital_injected,
      net_profit_total=net_money_profit,
      penalty_total=penalty_total,
      penalized_score_total=net_money_profit + penalty_total,
      exhausted_wallets=exhausted_wallets,
  )


def evaluate_basic_strategy(
    test_decks,
    config,
    show_progress=True,
    progress_callback=None,
    progress_report_interval=None,
):
  wallet_index = 0
  shoe_index = 0
  inactive_cash = 0
  bankruptcies = 0
  exhausted_wallets = False
  env = make_wallet_env(
      "basic", config, config.test_deck_seed, wallet_index,
      config.evaluation_bankroll)
  if test_decks:
    load_test_deck(env, test_decks[shoe_index])

  buffers = _new_evaluation_buffers(len(test_decks))
  outcomes = collections.Counter()
  wins = losses = draws = 0
  progress_report_interval = (
      progress_report_interval or progress_interval(len(test_decks)))
  iterator = progress_bar(
      range(len(test_decks)),
      desc="Evaluating Basic Strategy",
      disable=not show_progress or progress_callback is not None,
  )

  rounds_played = 0
  last_progress_report = 0
  for round_index in iterator:
    if len(env.deck) <= (1 - config.shoe_penetration) * env.max_deck_size:
      shoe_index += 1
      if shoe_index >= len(test_decks):
        break
      load_test_deck(env, test_decks[shoe_index])

    if not can_afford_next_round(env, config):
      inactive_cash += env.bankroll
      next_wallet_index = wallet_index + 1
      if not has_wallet_available(next_wallet_index, config.evaluation_wallets):
        exhausted_wallets = True
        break
      wallet_index = next_wallet_index
      env.bankroll = config.evaluation_bankroll

    result = play_basic_strategy_round(env, config)
    if result["is_bankrupt"]:
      bankruptcies += 1

    wallets_used = wallet_index + 1
    capital_injected = wallets_used * config.evaluation_bankroll
    total_cash = inactive_cash + env.bankroll
    net_money_profit = total_cash - capital_injected

    buffers["money_reward"][round_index] = result["money_reward"]
    buffers["penalized_score"][round_index] = result["penalized_score"]
    buffers["bet_amount"][round_index] = result["bet_amount"]
    buffers["bet_units"][round_index] = result["bet_units"]
    buffers["bankruptcy_event"][round_index] = result["is_bankrupt"]
    buffers["active_bankroll"][round_index] = env.bankroll
    buffers["total_cash"][round_index] = total_cash
    buffers["net_money_profit"][round_index] = net_money_profit
    buffers["wallet_id"][round_index] = wallets_used
    rounds_played = round_index + 1

    outcomes[result["outcome"]] += 1
    if result["money_reward"] > 0:
      wins += 1
    elif result["money_reward"] < 0:
      losses += 1
    else:
      draws += 1

    if rounds_played - last_progress_report >= progress_report_interval:
      report_progress(
          progress_callback,
          rounds_played - last_progress_report,
      )
      last_progress_report = rounds_played

  report_progress(progress_callback, rounds_played - last_progress_report)

  money_rewards = buffers["money_reward"][:rounds_played]
  penalized_scores = buffers["penalized_score"][:rounds_played]
  bet_amounts = buffers["bet_amount"][:rounds_played]
  bankruptcy_events = buffers["bankruptcy_event"][:rounds_played]
  wallets_used = wallet_index + 1
  capital_injected = wallets_used * config.evaluation_bankroll
  total_cash_remaining = inactive_cash + env.bankroll
  net_money_profit = total_cash_remaining - capital_injected
  penalty_total = bankruptcies * config.bankruptcy_penalty

  return EvaluationResult(
      state_mode="basic_strategy",
      money_reward=money_rewards,
      penalized_score=penalized_scores,
      bet_amount=bet_amounts,
      bet_units=buffers["bet_units"][:rounds_played],
      bankruptcy_event=bankruptcy_events,
      active_bankroll=buffers["active_bankroll"][:rounds_played],
      total_cash=buffers["total_cash"][:rounds_played],
      net_money_profit=buffers["net_money_profit"][:rounds_played],
      wallet_id=buffers["wallet_id"][:rounds_played],
      outcomes=outcomes,
      rounds_played=rounds_played,
      wins=wins,
      losses=losses,
      draws=draws,
      winrate=wins / rounds_played if rounds_played else 0,
      avg_bet=float(np.mean(bet_amounts)) if rounds_played else 0,
      avg_money_reward=float(np.mean(money_rewards)) if rounds_played else 0,
      avg_penalized_score=float(np.mean(penalized_scores)) if rounds_played else 0,
      wallet_count=config.evaluation_wallets,
      wallets_used=wallets_used,
      wallet_bankruptcies=bankruptcies,
      bankruptcy_rate=float(np.mean(bankruptcy_events)) if rounds_played else 0,
      current_budget=env.bankroll,
      inactive_cash=inactive_cash,
      total_cash_remaining=total_cash_remaining,
      capital_injected_total=capital_injected,
      net_profit_total=net_money_profit,
      penalty_total=penalty_total,
      penalized_score_total=net_money_profit + penalty_total,
      exhausted_wallets=exhausted_wallets,
  )


def _evaluate_agent_job(job):
  if len(job) == 4:
    label, training_result, test_decks, config = job
    progress_queue = None
    report_interval = None
  else:
    label, training_result, test_decks, config, progress_queue, report_interval = job
  progress_callback = progress_queue.put if progress_queue is not None else None
  return label, evaluate_agent(
      training_result.state_mode,
      training_result.q_table,
      training_result.bet_q_table,
      test_decks,
      config,
      show_progress=False,
      progress_callback=progress_callback,
      progress_report_interval=report_interval,
  )


def evaluate_agent_comparison(training_results, test_decks, config,
                              parallel=False, show_progress=True):
  if not parallel:
    return {
        label: evaluate_agent(
            training_result.state_mode,
            training_result.q_table,
            training_result.bet_q_table,
            test_decks,
            config,
            show_progress=show_progress,
        )
        for label, training_result in training_results.items()
    }

  max_workers = min(len(training_results), os.cpu_count() or len(training_results))
  jobs = [
      (label, training_result)
      for label, training_result in training_results.items()
  ]
  if not show_progress:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
      return dict(executor.map(_evaluate_agent_job, [
          (label, training_result, test_decks, config, None, None)
          for label, training_result in jobs
      ]))

  results = {}
  total_rounds = len(test_decks) * len(jobs)
  report_interval = progress_interval(len(test_decks))
  with Manager() as manager:
    progress_queue = manager.Queue()
    with progress_bar(total=total_rounds, desc="Evaluating agents") as bar:
      with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_agent_job,
                (
                    label,
                    training_result,
                    test_decks,
                    config,
                    progress_queue,
                    report_interval,
                ),
            ): label
            for label, training_result in jobs
        }
        pending = set(futures)
        while pending:
          done, pending = wait(
              pending,
              timeout=0.2,
              return_when=FIRST_COMPLETED,
          )
          drain_progress_queue(progress_queue, bar)
          for future in done:
            label, result = future.result()
            results[label] = result
        drain_progress_queue(progress_queue, bar)

  return {
      label: results[label]
      for label, _ in jobs
  }


def print_config_summary(config):
  print("Experiment config")
  print(f"Base seed: {config.base_seed}")
  print(f"Basic training seed: {config.train_seed_basic}")
  print(f"Counting training seed: {config.train_seed_extended}")
  print(f"Shared test deck seed: {config.test_deck_seed}")
  print(f"Decks: {config.num_decks}")
  print(f"Training rounds: {config.max_training_rounds:,}")
  print(f"Evaluation rounds: {config.eval_rounds:,}")
  print(f"Bet unit: {config.bet_unit}")
  print(f"Bet range: {config.min_bet} to {config.max_bet}")
  print(f"Training wallet: {config.training_bankroll}")
  print(f"Training wallets: {wallet_limit_label(config.training_wallets)}")
  print(f"Evaluation wallet: {config.evaluation_bankroll}")
  print(f"Evaluation wallets: {wallet_limit_label(config.evaluation_wallets)}")
  print(f"Bankruptcy penalty: {config.bankruptcy_penalty}")
  print(f"Blackjack payout: {config.blackjack_payout}:1")
  print(f"Dealer stands on soft 17: {config.stand_on_soft_17}")
  print(f"Dealer peek: {config.dealer_peek}")
  print(f"Shoe penetration: {config.shoe_penetration:.0%}")
  print(f"Actions: {', '.join(ACTION_NAMES)}")
  print(f"Max split hands: {config.max_split_hands}")
  print(f"Double after split: {config.double_after_split}")
  print(f"Late surrender: {config.allow_surrender}")
  print(f"Alpha: {config.alpha}")
  print(f"Gamma: {config.gamma}")


def print_training_summary(name, result):
  status = "OUT OF WALLETS" if result.exhausted_wallets else "finished max rounds"
  print(f"\n{name}")
  print(f"Rounds played: {result.rounds_played:,}")
  print(f"Wallets used: {result.wallets_used} / {wallet_limit_label(result.wallet_count)}")
  print(f"Wallet bankruptcies: {result.wallet_bankruptcies}")
  print(f"Bankruptcy rate: {result.bankruptcy_rate * 100:.2f}%")
  print(f"Avg chosen bet: {result.avg_bet:.2f}")
  print(f"Avg money reward: {result.avg_money_reward:.4f}")
  print(f"Avg learning reward: {result.avg_learning_reward:.4f}")
  print(f"Current wallet budget: {result.current_budget:.2f}")
  print(f"Final epsilon: {result.final_epsilon:.4f}")
  print(f"Status: {status}")


def print_evaluation_summary(name, result):
  status = "OUT OF WALLETS" if result.exhausted_wallets else "finished test decks"
  print(f"\n{name}")
  print(f"Rounds played: {result.rounds_played:,}")
  print(f"Wallets used: {result.wallets_used} / {wallet_limit_label(result.wallet_count)}")
  print(f"Wallet bankruptcies: {result.wallet_bankruptcies}")
  print(f"Bankruptcy rate: {result.bankruptcy_rate * 100:.2f}%")
  print(f"Avg chosen bet: {result.avg_bet:.2f}")
  print(f"Total capital injected: {result.capital_injected_total:.2f}")
  print(f"Inactive leftover cash: {result.inactive_cash:.2f}")
  print(f"Total cash remaining: {result.total_cash_remaining:.2f}")
  print(f"Net money profit: {result.net_profit_total:.2f}")
  print(f"Penalty-adjusted score: {result.penalized_score_total:.2f}")
  print(f"Status: {status}")
  print(f"Wins: {result.wins} | Losses: {result.losses} | Draws: {result.draws}")
  print(f"Winrate: {result.winrate * 100:.2f}%")
  print(f"Avg money reward: {result.avg_money_reward:.4f}")
  print(f"Avg penalty-adjusted score: {result.avg_penalized_score:.4f}")
  print(f"Outcomes: {dict(result.outcomes)}")


def format_int(value):
  return f"{int(value):,}"


def format_float(value, digits=2):
  return f"{float(value):,.{digits}f}"


def format_percent(value, digits=2):
  return f"{float(value) * 100:.{digits}f}%"


def result_status(result, completed_label):
  return "OUT OF WALLETS" if result.exhausted_wallets else completed_label


def rounds_per_ruin(result):
  if result.wallet_bankruptcies == 0:
    return "n/a"
  return format_float(result.rounds_played / result.wallet_bankruptcies, 1)


def comparison_table(results, rows):
  data = {
      label: [formatter(result) for _, formatter in rows]
      for label, result in results.items()
  }
  index = [metric for metric, _ in rows]
  try:
    import pandas as pd

    frame = pd.DataFrame(data, index=index)
    frame.index.name = "Metric"
    return frame
  except ImportError:
    return [
        {"Metric": metric, **{label: data[label][row_index]
                              for label in data}}
        for row_index, metric in enumerate(index)
    ]


def training_comparison_table(results):
  rows = (
      ("Avg learning reward", lambda result: format_float(result.avg_learning_reward, 4)),
      ("Ruins (wallet bankruptcies)", lambda result: format_int(result.wallet_bankruptcies)),
      ("Ruin rate", lambda result: format_percent(result.bankruptcy_rate)),
      ("Rounds per ruin", rounds_per_ruin),
      ("Avg money reward", lambda result: format_float(result.avg_money_reward, 4)),
      ("Wallets used", lambda result: f"{format_int(result.wallets_used)} / {wallet_limit_label(result.wallet_count)}"),
      ("Avg chosen bet", lambda result: format_float(result.avg_bet)),
      ("Current wallet budget", lambda result: format_float(result.current_budget)),
      ("Final epsilon", lambda result: format_float(result.final_epsilon, 4)),
      ("Rounds played", lambda result: format_int(result.rounds_played)),
      ("Status", lambda result: result_status(result, "finished max rounds")),
  )
  return comparison_table(results, rows)


def evaluation_comparison_table(results):
  rows = (
      ("Penalty-adjusted score", lambda result: format_float(result.penalized_score_total)),
      ("Net money profit", lambda result: format_float(result.net_profit_total)),
      ("Ruins (wallet bankruptcies)", lambda result: format_int(result.wallet_bankruptcies)),
      ("Ruin rate", lambda result: format_percent(result.bankruptcy_rate)),
      ("Rounds per ruin", rounds_per_ruin),
      ("Total cash remaining", lambda result: format_float(result.total_cash_remaining)),
      ("Total capital injected", lambda result: format_float(result.capital_injected_total)),
      ("Wallets used", lambda result: f"{format_int(result.wallets_used)} / {wallet_limit_label(result.wallet_count)}"),
      ("Avg penalty-adjusted score", lambda result: format_float(result.avg_penalized_score, 4)),
      ("Avg money reward", lambda result: format_float(result.avg_money_reward, 4)),
      ("Avg chosen bet", lambda result: format_float(result.avg_bet)),
      ("Winrate", lambda result: format_percent(result.winrate)),
      ("Wins", lambda result: format_int(result.wins)),
      ("Losses", lambda result: format_int(result.losses)),
      ("Draws", lambda result: format_int(result.draws)),
      ("Rounds played", lambda result: format_int(result.rounds_played)),
      ("Status", lambda result: result_status(result, "finished test decks")),
  )
  return comparison_table(results, rows)


def representative_policy_states():
  dealer_cards = list(range(2, 12))
  hard_specs = [
      (f"Hard {total}", total, False, True, False, True, "Hard")
      for total in range(5, 22)
  ]
  soft_specs = [
      (f"Soft {total}", total, True, True, False, True, "Soft")
      for total in range(13, 21)
  ]
  pair_specs = [
      ("Pair A,A", 12, True, True, True, True, "Pair"),
      ("Pair 10,10", 20, False, True, True, True, "Pair"),
      ("Pair 9,9", 18, False, True, True, True, "Pair"),
      ("Pair 8,8", 16, False, True, True, True, "Pair"),
      ("Pair 7,7", 14, False, True, True, True, "Pair"),
      ("Pair 6,6", 12, False, True, True, True, "Pair"),
      ("Pair 5,5", 10, False, True, True, True, "Pair"),
      ("Pair 4,4", 8, False, True, True, True, "Pair"),
      ("Pair 3,3", 6, False, True, True, True, "Pair"),
      ("Pair 2,2", 4, False, True, True, True, "Pair"),
  ]
  states = []
  for label, total, soft, can_double, can_split, can_surrender, group in (
      hard_specs + soft_specs + pair_specs):
    for dealer in dealer_cards:
      if total >= 21:
        state = (total, dealer, soft, False, False, False)
      else:
        state = (total, dealer, soft, can_double, can_split, can_surrender)
      states.append((group, label, dealer, state))
  return states


def policy_alignment_table(training_results):
  states = representative_policy_states()
  rows = [
      ("Basic-strategy agreement", "all"),
      ("Hard-hand agreement", "Hard"),
      ("Soft-hand agreement", "Soft"),
      ("Pair-hand agreement", "Pair"),
      ("Different first-decision cells", "different"),
  ]
  data = {}
  for label, result in training_results.items():
    values = {}
    for metric, group in rows:
      if group == "different":
        differences = sum(
            choose_greedy_action(state, result.q_table) !=
            basic_strategy_action_from_state(state)
            for _, _, _, state in states
        )
        values[metric] = str(differences)
        continue

      group_states = [
          state for state_group, _, _, state in states
          if group == "all" or state_group == group
      ]
      matches = sum(
          choose_greedy_action(state, result.q_table) ==
          basic_strategy_action_from_state(state)
          for state in group_states
      )
      values[metric] = format_percent(matches / len(group_states), 1)
    data[label] = [values[metric] for metric, _ in rows]

  try:
    import pandas as pd

    frame = pd.DataFrame(data, index=[metric for metric, _ in rows])
    frame.index.name = "Metric"
    return frame
  except ImportError:
    return [
        {"Metric": metric, **{label: data[label][row_index]
                              for label in data}}
        for row_index, (metric, _) in enumerate(rows)
    ]


def plot_training_dashboard(results, config):
  fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
  ax_reward, ax_bet, ax_bankruptcy, ax_wallet, ax_epsilon, ax_distribution = axes.ravel()

  for index, (label, result) in enumerate(results.items()):
    color = agent_color(label, index)
    rounds = np.arange(1, result.rounds_played + 1)

    reward_rounds = moving_average_rounds(
        result.learning_reward, config.learning_curve_window)
    plot_metric_series(
        ax_reward, reward_rounds, result.moving_avg_learning_reward, label,
        color)

    bet_rounds = moving_average_rounds(
        result.bet_amount, config.learning_curve_window)
    plot_metric_series(
        ax_bet, bet_rounds, result.moving_avg_bet, label, color)

    bankruptcy_rounds = moving_average_rounds(
        result.bankruptcy_event, config.bankruptcy_rate_window)
    plot_metric_series(
        ax_bankruptcy, bankruptcy_rounds, result.moving_bankruptcy_rate * 100,
        label, color)

    plot_metric_series(
        ax_wallet, rounds, result.active_bankroll, label, color,
        linewidth=1.2, alpha=0.85)
    plot_metric_series(
        ax_epsilon, rounds, result.epsilon, label, color, linewidth=1.2,
        show_trend=False)
    ax_distribution.hist(
        result.bet_amount,
        bins=np.arange(config.min_bet, config.max_bet + config.bet_unit * 2,
                       config.bet_unit) - config.bet_unit / 2,
        alpha=0.45,
        color=color,
        label=label,
    )

  ax_reward.axhline(0, color="black", linewidth=1, linestyle=":")
  ax_reward.set_title("Rolling learning reward")
  ax_reward.set_xlabel("Round")
  ax_reward.set_ylabel("Reward per round")
  ax_reward.grid(True, alpha=0.25)
  ax_reward.legend(loc="upper right")

  ax_bet.set_title("Rolling chosen bet")
  ax_bet.set_xlabel("Round")
  ax_bet.set_ylabel("Bet")
  ax_bet.set_ylim(bottom=0)
  ax_bet.grid(True, alpha=0.25)
  ax_bet.legend(loc="upper right")

  ax_bankruptcy.set_title("Rolling bankruptcy rate")
  ax_bankruptcy.set_xlabel("Round")
  ax_bankruptcy.set_ylabel("Bankrupt rounds (%)")
  ax_bankruptcy.set_ylim(bottom=0)
  ax_bankruptcy.grid(True, alpha=0.25)
  ax_bankruptcy.legend(loc="upper right")

  ax_wallet.axhline(config.min_bet, color="black", linewidth=1,
                    linestyle=":", label="next bet needed")
  ax_wallet.set_title("Active wallet budget")
  ax_wallet.set_xlabel("Round")
  ax_wallet.set_ylabel("Budget")
  ax_wallet.grid(True, alpha=0.25)
  ax_wallet.legend(loc="upper right")

  ax_epsilon.set_title("Exploration rate")
  ax_epsilon.set_xlabel("Round")
  ax_epsilon.set_ylabel("Epsilon")
  ax_epsilon.set_ylim(0, 1.05)
  ax_epsilon.grid(True, alpha=0.25)
  ax_epsilon.legend(loc="upper right")

  ax_distribution.set_title("Training bet distribution")
  ax_distribution.set_xlabel("Bet")
  ax_distribution.set_ylabel("Rounds")
  ax_distribution.grid(True, axis="y", alpha=0.25)
  ax_distribution.legend(loc="upper right")

  plt.show()


def _policy_action(
    q_table,
    state_mode,
    player_total,
    dealer_card,
    usable_ace,
    can_double=True,
    can_split=False,
    can_surrender=True,
    true_count=0,
):
  if player_total >= 21:
    can_double = can_split = can_surrender = False

  state = (
      player_total,
      dealer_card,
      usable_ace,
      can_double,
      can_split,
      can_surrender,
  )
  if state_mode == "extended":
    state = state + (true_count,)
  return choose_greedy_action(state, q_table)


def _draw_policy_heatmap(ax, q_table, state_mode, row_specs, title,
                         true_count=0):
  dealer_cards = list(range(2, 12))
  policy = np.array([
      [
          _policy_action(
              q_table,
              state_mode,
              total,
              dealer,
              usable_ace,
              can_double=can_double,
              can_split=can_split,
              can_surrender=can_surrender,
              true_count=true_count,
          )
          for dealer in dealer_cards
      ]
      for _, total, usable_ace, can_double, can_split, can_surrender
      in row_specs
  ])

  cmap = ListedColormap(ACTION_COLORS)
  norm = BoundaryNorm(np.arange(-0.5, len(ACTIONS) + 0.5), cmap.N)
  image = ax.imshow(policy, cmap=cmap, norm=norm, aspect="auto")
  for row_index, _ in enumerate(row_specs):
    for col_index, dealer_card in enumerate(dealer_cards):
      action_index = int(policy[row_index, col_index])
      action_name = ACTION_SHORT_NAMES[action_index]
      ax.text(
          col_index,
          row_index,
          action_name,
          ha="center",
          va="center",
          color="white",
          fontsize=8,
          fontweight="bold",
      )

  ax.set_title(title)
  ax.set_xticks(np.arange(len(dealer_cards)))
  ax.set_xticklabels([str(card) if card != 11 else "A" for card in dealer_cards])
  ax.set_yticks(np.arange(len(row_specs)))
  ax.set_yticklabels([label for label, *_ in row_specs])
  ax.set_xlabel("Dealer upcard")
  ax.set_ylabel("Player hand")
  return image


def plot_policy_heatmaps(agent_specs, true_count=0):
  fig, axes = plt.subplots(3, len(agent_specs), figsize=(6 * len(agent_specs), 12),
                           constrained_layout=True)
  if len(agent_specs) == 1:
    axes = np.array(axes).reshape(3, 1)

  hard_specs = [
      (str(total), total, False, True, False, True)
      for total in range(21, 3, -1)
  ]
  soft_specs = [
      (f"A,{total - 11}", total, True, True, False, True)
      for total in range(20, 12, -1)
  ]
  pair_specs = [
      ("A,A", 12, True, True, True, True),
      ("10,10", 20, False, True, True, True),
      ("9,9", 18, False, True, True, True),
      ("8,8", 16, False, True, True, True),
      ("7,7", 14, False, True, True, True),
      ("6,6", 12, False, True, True, True),
      ("5,5", 10, False, True, True, True),
      ("4,4", 8, False, True, True, True),
      ("3,3", 6, False, True, True, True),
      ("2,2", 4, False, True, True, True),
  ]
  image = None
  for col, (label, spec) in enumerate(agent_specs.items()):
    state_mode, q_table = spec
    suffix = f", TC {true_count}" if state_mode == "extended" else ""
    image = _draw_policy_heatmap(
        axes[0, col], q_table, state_mode, hard_specs,
        f"{label}: hard first decisions{suffix}",
        true_count=true_count)
    _draw_policy_heatmap(
        axes[1, col], q_table, state_mode, soft_specs,
        f"{label}: soft first decisions{suffix}",
        true_count=true_count)
    _draw_policy_heatmap(
        axes[2, col], q_table, state_mode, pair_specs,
        f"{label}: pair first decisions{suffix}",
        true_count=true_count)

  colorbar = fig.colorbar(
      image,
      ax=axes.ravel().tolist(),
      ticks=list(ACTIONS),
      shrink=0.82,
  )
  colorbar.ax.set_yticklabels(ACTION_NAMES)
  plt.show()


def _representative_bankroll_buckets(config):
  max_wallet_units = max(
      1,
      config.training_bankroll // config.bet_unit,
      config.evaluation_bankroll // config.bet_unit,
  )
  buckets = [
      limit for limit in BANKROLL_BUCKET_LIMITS
      if limit < max_wallet_units
  ]
  for limit in BANKROLL_BUCKET_LIMITS:
    if limit >= max_wallet_units:
      buckets.append(limit)
      break
  else:
    buckets.append(BANKROLL_BUCKET_LIMITS[-1])
  return tuple(dict.fromkeys(buckets))


def _betting_policy_state(state_mode, bankroll_bucket, true_count):
  state = (bankroll_bucket,)
  if state_mode == "extended":
    state = state + (true_count,)
  return state


def _legal_indices_for_bucket(config, bankroll_bucket):
  max_units = min(config.max_bet_units, bankroll_bucket)
  return tuple(range(max_units))


def plot_betting_policy_heatmaps(agent_specs, config):
  fig, axes = plt.subplots(1, len(agent_specs), figsize=(6 * len(agent_specs), 5),
                           constrained_layout=True)
  if len(agent_specs) == 1:
    axes = np.array([axes])

  buckets = tuple(reversed(_representative_bankroll_buckets(config)))
  image = None
  for ax, (label, spec) in zip(axes, agent_specs.items()):
    state_mode, bet_q_table = spec
    true_counts = list(range(-5, 6)) if state_mode == "extended" else [0]
    policy = np.array([
        [
            bet_amount_from_index(
                choose_greedy_bet_index(
                    _betting_policy_state(state_mode, bucket, true_count),
                    bet_q_table,
                    _legal_indices_for_bucket(config, bucket),
                    config=config,
                    state_mode=state_mode,
                ),
                config,
            )
            for true_count in true_counts
        ]
        for bucket in buckets
    ])

    image = ax.imshow(policy, cmap="viridis", aspect="auto")
    for row_index, bucket in enumerate(buckets):
      for col_index, _ in enumerate(true_counts):
        ax.text(
            col_index,
            row_index,
            f"{policy[row_index, col_index]:.0f}",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title(f"{label}: betting policy")
    ax.set_xticks(np.arange(len(true_counts)))
    ax.set_xticklabels(true_counts if state_mode == "extended" else ["all"])
    ax.set_xlabel("True count" if state_mode == "extended" else "Count ignored")
    ax.set_yticks(np.arange(len(buckets)))
    ax.set_yticklabels([bankroll_bucket_label(bucket) for bucket in buckets])
    ax.set_ylabel("Wallet bucket")

  colorbar = fig.colorbar(image, ax=np.ravel(axes).tolist(), shrink=0.8)
  colorbar.set_label("Bet")
  plt.show()


def plot_evaluation_dashboard(results, config=None):
  fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
  ax_profit, ax_reward, ax_bet, ax_wallet, ax_distribution, ax_summary = axes.ravel()

  for index, (label, result) in enumerate(results.items()):
    color = agent_color(label, index)
    rounds = np.arange(1, result.rounds_played + 1)
    plot_metric_series(
        ax_profit, rounds, result.net_money_profit, label, color)
    bankrupt_rounds = np.flatnonzero(result.bankruptcy_event)
    if len(bankrupt_rounds):
      ax_profit.scatter(
          rounds[bankrupt_rounds],
          result.net_money_profit[bankrupt_rounds],
          marker="x",
          s=35,
          linewidths=1.5,
          color=color,
      )

    reward_window = min(500, max(1, result.rounds_played))
    reward_ma = moving_average(result.money_reward, reward_window)
    reward_rounds = moving_average_rounds(result.money_reward, reward_window)
    plot_metric_series(
        ax_reward, reward_rounds, reward_ma, label, color)

    plot_metric_series(
        ax_wallet, rounds, result.active_bankroll, label, color,
        linewidth=1.2)
    if len(bankrupt_rounds):
      ax_wallet.scatter(
          rounds[bankrupt_rounds],
          result.active_bankroll[bankrupt_rounds],
          marker="x",
          s=35,
          linewidths=1.5,
          color=color,
      )

    plot_metric_series(
        ax_bet, rounds, result.bet_amount, label, color, linewidth=1.2,
        alpha=0.9)
    if result.rounds_played:
      bet_step = config.bet_unit if config else 1
      min_bet = config.min_bet if config else float(result.bet_amount.min())
      max_bet = config.max_bet if config else float(result.bet_amount.max())
      ax_distribution.hist(
          result.bet_amount,
          bins=np.arange(min_bet, max_bet + bet_step * 2, bet_step) -
          bet_step / 2,
          alpha=0.45,
          color=color,
          label=label,
      )

  ax_profit.axhline(0, color="black", linewidth=1, linestyle=":")
  ax_profit.set_title("Evaluation net profit")
  ax_profit.set_xlabel("Round")
  ax_profit.set_ylabel("Money")
  ax_profit.grid(True, alpha=0.25)
  ax_profit.legend(loc="upper right")

  ax_reward.axhline(0, color="black", linewidth=1, linestyle=":")
  ax_reward.set_title("Rolling money reward")
  ax_reward.set_xlabel("Round")
  ax_reward.set_ylabel("Reward per round")
  ax_reward.grid(True, alpha=0.25)
  ax_reward.legend(loc="upper right")

  ax_bet.set_title("Chosen bet")
  ax_bet.set_xlabel("Round")
  ax_bet.set_ylabel("Bet")
  ax_bet.set_ylim(bottom=0)
  ax_bet.grid(True, alpha=0.25)
  ax_bet.legend(loc="upper right")

  if config is not None:
    ax_wallet.axhline(
        config.min_bet,
        color="black",
        linewidth=1,
        linestyle=":",
        label="next bet needed",
    )
  ax_wallet.set_title("Active wallet budget")
  ax_wallet.set_xlabel("Round")
  ax_wallet.set_ylabel("Budget")
  ax_wallet.grid(True, alpha=0.25)
  ax_wallet.legend(loc="upper right")

  ax_distribution.set_title("Evaluation bet distribution")
  ax_distribution.set_xlabel("Bet")
  ax_distribution.set_ylabel("Rounds")
  ax_distribution.grid(True, axis="y", alpha=0.25)
  ax_distribution.legend(loc="upper right")

  score_metrics = ("Net profit", "Penalty-adjusted score")
  x = np.arange(len(score_metrics))
  width = 0.8 / max(1, len(results))
  bar_groups = []
  for index, (label, result) in enumerate(results.items()):
    values = (result.net_profit_total, result.penalized_score_total)
    offset = (index - (len(results) - 1) / 2) * width
    bars = ax_summary.bar(
        x + offset,
        values,
        width,
        label=label,
        color=agent_color(label, index),
        alpha=0.85,
    )
    bar_groups.append((bars, values))

  ax_summary.axhline(0, color="black", linewidth=1, linestyle=":")
  ax_summary.set_title("Final evaluation score")
  ax_summary.set_xticks(x)
  ax_summary.set_xticklabels(score_metrics)
  ax_summary.set_ylabel("Money")
  ax_summary.grid(True, axis="y", alpha=0.25)
  ax_summary.legend(loc="upper right")

  for bars, values in bar_groups:
    for bar, value in zip(bars, values):
      label_y = value + (25 if value >= 0 else -25)
      va = "bottom" if value >= 0 else "top"
      ax_summary.text(
          bar.get_x() + bar.get_width() / 2,
          label_y,
          f"{value:.0f}",
          ha="center",
          va=va,
          fontsize=8,
      )

  plt.show()
