import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BlackjackEnv(gym.Env):
    """Blackjack environment with a finite shoe and count-tracking features."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, num_decks=6, penetration=0.75, stand_on_soft_17=True):
        super().__init__()

        if num_decks < 1:
            raise ValueError("num_decks must be at least 1")
        if not 0 < penetration < 1:
            raise ValueError("penetration must be between 0 and 1")

        self.num_decks = num_decks
        self.penetration = penetration
        self.stand_on_soft_17 = stand_on_soft_17

        self.action_space = spaces.Discrete(2)  # 0: Stand, 1: Hit
        self.observation_space = spaces.Box(
            low=np.array([0, 1, 0, -np.inf, -np.inf, 0], dtype=np.float32),
            high=np.array([31, 10, 1, np.inf, np.inf, 52 * num_decks], dtype=np.float32),
            dtype=np.float32,
        )

        self._shoe = []
        self.player_hand = []
        self.dealer_hand = []
        self.running_count = 0
        self._dealer_hole_revealed = False
        self._cards_dealt = 0
        self._cut_card = max(1, int((52 * self.num_decks) * self.penetration))
        self._rng = np.random.default_rng()
        self._reset_shoe()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._needs_reshuffle():
            self._reset_shoe()

        self._dealer_hole_revealed = False
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card(update_count=False)]

        observation = self._get_observation()
        info = {
            "running_count": self.running_count,
            "true_count": self._true_count(),
            "cards_remaining": len(self._shoe),
            "player_hand": list(self.player_hand),
            "dealer_upcard": self._card_value(self.dealer_hand[0]),
        }
        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        reward = 0.0
        terminated = False
        truncated = False

        if action == 1:
            self.player_hand.append(self._draw_card())
            if self._hand_value(self.player_hand) > 21:
                reward = -1.0
                terminated = True
        else:
            self._reveal_dealer_hole_card()
            while self._dealer_should_hit():
                self.dealer_hand.append(self._draw_card())

            player_value = self._hand_value(self.player_hand)
            dealer_value = self._hand_value(self.dealer_hand)

            if dealer_value > 21 or player_value > dealer_value:
                reward = 1.0
            elif player_value < dealer_value:
                reward = -1.0
            else:
                reward = 0.0
            terminated = True

        observation = self._get_observation()
        info = {
            "running_count": self.running_count,
            "true_count": self._true_count(),
            "cards_remaining": len(self._shoe),
            "player_hand": list(self.player_hand),
            "dealer_upcard": self._card_value(self.dealer_hand[0]),
        }
        return observation, reward, terminated, truncated, info

    def render(self):
        player_value = self._hand_value(self.player_hand)
        print(
            f"Player: {self.player_hand} (value={player_value}) | "
            f"Dealer: [{self.dealer_hand[0]}, ?] | "
            f"Running count: {self.running_count}"
        )

    def _reset_shoe(self):
        self._shoe = [rank for _ in range(self.num_decks) for rank in range(1, 14) for _ in range(4)]
        self._rng.shuffle(self._shoe)
        self.running_count = 0
        self._dealer_hole_revealed = False
        self._cards_dealt = 0

    def _needs_reshuffle(self):
        return len(self._shoe) <= self._cut_card

    def _draw_card(self, update_count=True):
        if not self._shoe:
            self._reset_shoe()

        card = self._shoe.pop()
        self._cards_dealt += 1
        if update_count:
            self.running_count += self._hi_lo_value(card)
        return card

    def _reveal_dealer_hole_card(self):
        if not self._dealer_hole_revealed and len(self.dealer_hand) > 1:
            self.running_count += self._hi_lo_value(self.dealer_hand[1])
            self._dealer_hole_revealed = True

    @staticmethod
    def _card_value(card_rank):
        return min(card_rank, 10)

    @staticmethod
    def _hi_lo_value(card_rank):
        card_value = min(card_rank, 10)
        if 2 <= card_value <= 6:
            return 1
        if 7 <= card_value <= 9:
            return 0
        return -1

    def _hand_value(self, hand):
        values = [self._card_value(card) for card in hand]
        total = sum(values)
        ace_count = sum(1 for card in hand if card == 1)

        while ace_count > 0 and total + 10 <= 21:
            total += 10
            ace_count -= 1

        return total

    def _usable_ace(self, hand):
        values = [self._card_value(card) for card in hand]
        total = sum(values)
        return any(card == 1 for card in hand) and total + 10 <= 21

    def _dealer_should_hit(self):
        dealer_value = self._hand_value(self.dealer_hand)
        if dealer_value < 17:
            return True
        if dealer_value > 17:
            return False
        return not self.stand_on_soft_17 and self._usable_ace(self.dealer_hand)

    def _true_count(self):
        decks_remaining = max(len(self._shoe) / 52.0, 0.25)
        return float(self.running_count / decks_remaining)

    def _get_observation(self):
        player_total = self._hand_value(self.player_hand)
        dealer_upcard = self._card_value(self.dealer_hand[0])
        usable_ace = 1.0 if self._usable_ace(self.player_hand) else 0.0
        running_count = float(self.running_count)
        true_count = self._true_count()
        cards_remaining = float(len(self._shoe))
        return np.array(
            [player_total, dealer_upcard, usable_ace, running_count, true_count, cards_remaining],
            dtype=np.float32,
        )
