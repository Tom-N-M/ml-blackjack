import collections
import random


ACTION_HIT = 0
ACTION_STAND = 1
ACTION_DOUBLE = 2
ACTION_SPLIT = 3
ACTION_SURRENDER = 4


class BlackjackEnv:
  def __init__(
      self,
      state_mode="basic",
      num_decks=6,
      stand_on_soft_17=True,
      min_bet=1,
      max_bet=None,
      bankroll=None,
      blackjack_payout=1.5,
      allow_double=True,
      allow_split=True,
      allow_surrender=True,
      double_after_split=True,
      max_split_hands=4,
      hit_split_aces=False,
      resplit_aces=False,
      dealer_peek=True,
      shoe_penetration=0.75,
      seed=None):
    self.state_mode = state_mode
    self.num_decks = self._validate_positive_int(num_decks, "num_decks")
    self.stand_on_soft_17 = self._validate_bool(
        stand_on_soft_17, "stand_on_soft_17")
    self.seed = seed
    self.rng = random.Random(seed)
    self.min_bet = self._validate_positive_number(min_bet, "min_bet")
    self.max_bet = self._validate_optional_positive_number(max_bet, "max_bet")
    self.bankroll = self._validate_optional_non_negative_number(
        bankroll, "bankroll")
    self.blackjack_payout = self._validate_positive_number(
        blackjack_payout, "blackjack_payout")
    self.allow_double = self._validate_bool(allow_double, "allow_double")
    self.allow_split = self._validate_bool(allow_split, "allow_split")
    self.allow_surrender = self._validate_bool(
        allow_surrender, "allow_surrender")
    self.double_after_split = self._validate_bool(
        double_after_split, "double_after_split")
    self.max_split_hands = self._validate_positive_int(
        max_split_hands, "max_split_hands")
    self.hit_split_aces = self._validate_bool(
        hit_split_aces, "hit_split_aces")
    self.resplit_aces = self._validate_bool(resplit_aces, "resplit_aces")
    self.dealer_peek = self._validate_bool(dealer_peek, "dealer_peek")
    self.shoe_penetration = self._validate_penetration(shoe_penetration)

    if self.max_bet is not None and self.max_bet < self.min_bet:
      raise ValueError("max_bet must be greater than or equal to min_bet.")

    self.cards = ['ace', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'jack', 'queen', 'king']
    self.max_deck_size = 52 * self.num_decks
    self.current_bet = self.min_bet
    self.round_start_bankroll = self.bankroll
    self.hands = []
    self.player_hand = []
    self.dealer_hand = []
    self.active_hand_index = 0
    self.done = True
    self.pending_reward = None
    self.last_outcome = None
    self.last_profit = None
    self.last_hand_outcomes = collections.Counter()
    self.dealer_hole_revealed = False
    self._shuffle_deck()

  def _validate_positive_int(self, value, name):
    if isinstance(value, bool) or not isinstance(value, int):
      raise ValueError(f"{name} must be a positive integer.")
    if value <= 0:
      raise ValueError(f"{name} must be greater than 0.")
    return value

  def _validate_bool(self, value, name):
    if not isinstance(value, bool):
      raise ValueError(f"{name} must be a boolean.")
    return value

  def _validate_positive_number(self, value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError(f"{name} must be a positive number.")
    if value <= 0:
      raise ValueError(f"{name} must be greater than 0.")
    return value

  def _validate_optional_positive_number(self, value, name):
    if value is None:
      return None
    return self._validate_positive_number(value, name)

  def _validate_optional_non_negative_number(self, value, name):
    if value is None:
      return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError(f"{name} must be a non-negative number.")
    if value < 0:
      raise ValueError(f"{name} must be greater than or equal to 0.")
    return value

  def _validate_penetration(self, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError("shoe_penetration must be a number in (0, 1].")
    if value <= 0 or value > 1:
      raise ValueError("shoe_penetration must be in the interval (0, 1].")
    return value

  def _validate_bet(self, bet):
    bet = self._validate_positive_number(bet, "bet")

    if bet < self.min_bet:
      raise ValueError("bet must be greater than or equal to min_bet.")
    if self.max_bet is not None and bet > self.max_bet:
      raise ValueError("bet must be less than or equal to max_bet.")
    if self.bankroll is not None and bet > self.bankroll:
      raise ValueError("bet cannot exceed the current bankroll.")

    return bet

  def _shuffle_deck(self):
    """Reset and shuffle the shoe."""
    self.deck = self.cards * 4 * self.num_decks
    self.rng.shuffle(self.deck)
    self.played_cards = []
    self.cards_dealt = 0

  def _draw_card(self):
    if not self.deck:
      raise RuntimeError("Shoe is empty. Shuffle before starting the next round.")

    self.cards_dealt += 1
    return self.deck.pop()

  def _mark_seen(self, card):
    self.played_cards.append(card)

  def _draw_visible_card(self):
    card = self._draw_card()
    self._mark_seen(card)
    return card

  def reset(self, bet=None):
    """Deal a new round and return the initial decision state."""
    next_bet = self._validate_bet(self.min_bet if bet is None else bet)

    if len(self.deck) <= (1 - self.shoe_penetration) * self.max_deck_size:
      self._shuffle_deck()

    self.current_bet = next_bet
    self.round_start_bankroll = self.bankroll
    self.done = False
    self.pending_reward = None
    self.last_outcome = None
    self.last_profit = None
    self.last_hand_outcomes = collections.Counter()
    self.dealer_hole_revealed = False

    player_first = self._draw_card()
    dealer_upcard = self._draw_card()
    player_second = self._draw_card()
    dealer_hole = self._draw_card()

    self.hands = [self._new_hand([player_first, player_second], next_bet)]
    self.active_hand_index = 0
    self.dealer_hand = [dealer_upcard, dealer_hole]
    self._sync_player_hand()

    for card in [player_first, dealer_upcard, player_second]:
      self._mark_seen(card)

    player_blackjack = self._is_natural_blackjack(self.player_hand)
    dealer_blackjack = self._is_natural_blackjack(self.dealer_hand)
    dealer_can_peek = self._get_card_value(dealer_upcard) in (10, 11)

    if self.dealer_peek and dealer_can_peek and dealer_blackjack:
      self._reveal_dealer_hole()
      if player_blackjack:
        self._finish_round("push", 0)
      else:
        self._finish_round("lose", -next_bet)
    elif player_blackjack:
      self._finish_round("blackjack", next_bet * self.blackjack_payout)

    return self._get_state()

  def _new_hand(self, cards, bet, from_split=False, split_from_aces=False):
    return {
        "cards": list(cards),
        "bet": bet,
        "done": False,
        "surrendered": False,
        "doubled": False,
        "from_split": from_split,
        "split_from_aces": split_from_aces,
    }

  def _sync_player_hand(self):
    if not self.hands:
      self.player_hand = []
      self.current_bet = self.min_bet
      return

    self.active_hand_index = min(self.active_hand_index, len(self.hands) - 1)
    active_hand = self.hands[self.active_hand_index]
    self.player_hand = active_hand["cards"]
    self.current_bet = active_hand["bet"]

  def _active_hand(self):
    if self.done or not self.hands:
      return None
    return self.hands[self.active_hand_index]

  def _get_card_value(self, card):
    """Return a card's blackjack value before ace reductions."""
    if card in ['jack', 'queen', 'king']:
      return 10
    elif card == 'ace':
      return 11
    else:
      return int(card)

  def _split_value(self, card):
    value = self._get_card_value(card)
    return 10 if value == 10 else value

  def _score_and_soft(self, hand):
    score = sum(self._get_card_value(card) for card in hand)
    aces_as_eleven = sum(1 for card in hand if card == 'ace')

    while score > 21 and aces_as_eleven:
      score -= 10
      aces_as_eleven -= 1

    return score, aces_as_eleven > 0

  def get_score(self, hand):
    """Calculate a hand's best blackjack score."""
    score, _ = self._score_and_soft(hand)
    return score

  def _has_usable_ace(self, hand):
    _, soft = self._score_and_soft(hand)
    return soft

  def _is_natural_blackjack(self, hand):
    return len(hand) == 2 and self.get_score(hand) == 21

  def _finish_round(self, outcome, profit):
    self.done = True
    self.pending_reward = profit
    self.last_outcome = outcome
    self.last_profit = profit

    if self.bankroll is not None:
      self.bankroll += profit

    self._sync_player_hand()
    return profit

  def _reveal_dealer_hole(self):
    if not self.dealer_hole_revealed:
      self._mark_seen(self.dealer_hand[1])
      self.dealer_hole_revealed = True

  def _hi_lo_value(self, card):
    value = self._get_card_value(card)
    if 2 <= value <= 6:
      return 1
    if value >= 10:
      return -1
    return 0

  def _get_true_count_bucket(self):
    running_count = sum(self._hi_lo_value(card) for card in self.played_cards)
    decks_remaining = max(len(self.deck) / 52, 0.25)
    true_count = round(running_count / decks_remaining)
    return max(-5, min(5, true_count))

  def _get_state(self):
    if getattr(self, "done", False):
      return ("terminal",)

    active_hand = self._active_hand()
    if active_hand is None:
      return ("terminal",)

    player_score = self.get_score(active_hand["cards"])
    dealer_card = self._get_card_value(self.dealer_hand[0])
    usable_ace = self._has_usable_ace(active_hand["cards"])

    base_state = (
        player_score,
        dealer_card,
        usable_ace,
        self._can_double(active_hand),
        self._can_split(active_hand),
        self._can_surrender(active_hand),
    )

    if self.state_mode == "basic":
      return base_state

    if self.state_mode == "extended":
      return base_state + (self._get_true_count_bucket(),)

    raise ValueError("Unknown state_mode")

  def _total_bet_at_risk(self):
    return sum(hand["bet"] for hand in self.hands)

  def _can_cover_extra_bet(self, amount):
    if self.bankroll is None:
      return True
    return self._total_bet_at_risk() + amount <= self.round_start_bankroll

  def _is_split_ace_locked(self, hand):
    return hand["split_from_aces"] and not self.hit_split_aces

  def _can_double(self, hand):
    if not self.allow_double or hand["done"] or self._is_split_ace_locked(hand):
      return False
    if len(hand["cards"]) != 2 or self.get_score(hand["cards"]) >= 21:
      return False
    if hand["from_split"] and not self.double_after_split:
      return False
    return self._can_cover_extra_bet(hand["bet"])

  def _can_split(self, hand):
    if not self.allow_split or hand["done"]:
      return False
    if len(hand["cards"]) != 2 or len(self.hands) >= self.max_split_hands:
      return False
    first, second = hand["cards"]
    if self._split_value(first) != self._split_value(second):
      return False
    if first == 'ace' and hand["from_split"] and not self.resplit_aces:
      return False
    return self._can_cover_extra_bet(hand["bet"])

  def _can_surrender(self, hand):
    if not self.allow_surrender or hand["done"]:
      return False
    if hand["from_split"] or len(hand["cards"]) != 2:
      return False
    return self.get_score(hand["cards"]) < 21

  def legal_actions(self):
    """Return the player actions that are legal for the active hand."""
    active_hand = self._active_hand()
    if active_hand is None:
      return ()

    score = self.get_score(active_hand["cards"])
    if score >= 21 or self._is_split_ace_locked(active_hand):
      return (ACTION_STAND,)

    actions = [ACTION_HIT, ACTION_STAND]
    if self._can_double(active_hand):
      actions.append(ACTION_DOUBLE)
    if self._can_split(active_hand):
      actions.append(ACTION_SPLIT)
    if self._can_surrender(active_hand):
      actions.append(ACTION_SURRENDER)
    return tuple(actions)

  def step(self, action):
    """Apply one player action and return state, reward, done."""
    if self.done:
      return self._get_state(), self.pending_reward, True

    if action not in self.legal_actions():
      raise ValueError("Action is not legal for the current hand.")

    if action == ACTION_HIT:
      return self.hit()
    if action == ACTION_STAND:
      return self.stand()
    if action == ACTION_DOUBLE:
      return self.double_down()
    if action == ACTION_SPLIT:
      return self.split()
    if action == ACTION_SURRENDER:
      return self.surrender()

    raise ValueError("Invalid action.")

  def hit(self):
    """Add one visible card to the active player hand."""
    if ACTION_HIT not in self.legal_actions():
      raise ValueError("Cannot hit the current hand.")

    active_hand = self._active_hand()
    active_hand["cards"].append(self._draw_visible_card())
    self._sync_player_hand()

    if self.get_score(active_hand["cards"]) > 21:
      active_hand["done"] = True
      return self._advance_hand_or_settle()

    return self._get_state(), 0, False

  def double_down(self):
    """Double the active bet, draw one card, then stand that hand."""
    if ACTION_DOUBLE not in self.legal_actions():
      raise ValueError("Cannot double the current hand.")

    active_hand = self._active_hand()
    active_hand["bet"] += active_hand["bet"]
    active_hand["doubled"] = True
    active_hand["cards"].append(self._draw_visible_card())
    active_hand["done"] = True
    self._sync_player_hand()
    return self._advance_hand_or_settle()

  def split(self):
    """Split a pair into two independently played hands."""
    if ACTION_SPLIT not in self.legal_actions():
      raise ValueError("Cannot split the current hand.")

    active_hand = self._active_hand()
    first_card, second_card = active_hand["cards"]
    bet = active_hand["bet"]
    split_from_aces = first_card == 'ace' and second_card == 'ace'

    first_hand = self._new_hand(
        [first_card, self._draw_visible_card()],
        bet,
        from_split=True,
        split_from_aces=split_from_aces,
    )
    second_hand = self._new_hand(
        [second_card, self._draw_visible_card()],
        bet,
        from_split=True,
        split_from_aces=split_from_aces,
    )

    if split_from_aces and not self.hit_split_aces:
      first_hand["done"] = True
      second_hand["done"] = True

    self.hands[self.active_hand_index:self.active_hand_index + 1] = [
        first_hand,
        second_hand,
    ]
    self._sync_player_hand()

    if first_hand["done"]:
      return self._advance_hand_or_settle()

    return self._get_state(), 0, False

  def surrender(self):
    """Late surrender the active hand for half the bet."""
    if ACTION_SURRENDER not in self.legal_actions():
      raise ValueError("Cannot surrender the current hand.")

    active_hand = self._active_hand()
    active_hand["surrendered"] = True
    active_hand["done"] = True
    return self._advance_hand_or_settle()

  def _dealer_should_hit(self):
    dealer_score, dealer_soft = self._score_and_soft(self.dealer_hand)
    if dealer_score < 17:
      return True
    return dealer_score == 17 and dealer_soft and not self.stand_on_soft_17

  def stand(self):
    """Stand the active hand and continue with the next hand or dealer."""
    if ACTION_STAND not in self.legal_actions():
      raise ValueError("Cannot stand the current hand.")

    active_hand = self._active_hand()
    active_hand["done"] = True
    return self._advance_hand_or_settle()

  def _advance_hand_or_settle(self):
    for next_index in range(self.active_hand_index + 1, len(self.hands)):
      if not self.hands[next_index]["done"]:
        self.active_hand_index = next_index
        self._sync_player_hand()
        return self._get_state(), 0, False

    return self._settle_round()

  def _settle_round(self):
    needs_dealer = any(
        not hand["surrendered"] and self.get_score(hand["cards"]) <= 21
        for hand in self.hands
    )
    dealer_blackjack = self._is_natural_blackjack(self.dealer_hand)

    if needs_dealer:
      self._reveal_dealer_hole()
      while self._dealer_should_hit():
        self.dealer_hand.append(self._draw_visible_card())
      dealer_blackjack = self._is_natural_blackjack(self.dealer_hand)

    dealer_score = self.get_score(self.dealer_hand)
    total_profit = 0
    hand_outcomes = collections.Counter()

    for hand in self.hands:
      outcome, profit = self._settle_hand(hand, dealer_score, dealer_blackjack)
      total_profit += profit
      hand_outcomes[outcome] += 1

    self.last_hand_outcomes = hand_outcomes
    if len(self.hands) == 1:
      outcome = next(iter(hand_outcomes))
    elif total_profit > 0:
      outcome = "win"
    elif total_profit < 0:
      outcome = "lose"
    else:
      outcome = "push"

    reward = self._finish_round(outcome, total_profit)
    return self._get_state(), reward, True

  def _settle_hand(self, hand, dealer_score, dealer_blackjack):
    player_score = self.get_score(hand["cards"])
    bet = hand["bet"]

    if hand["surrendered"]:
      return "surrender", -0.5 * bet
    if player_score > 21:
      return "bust", -bet

    player_blackjack = (
        self._is_natural_blackjack(hand["cards"]) and not hand["from_split"])
    if player_blackjack:
      if dealer_blackjack:
        return "push", 0
      return "blackjack", bet * self.blackjack_payout

    if dealer_blackjack:
      return "lose", -bet
    if dealer_score > 21 or player_score > dealer_score:
      return "win", bet
    if player_score < dealer_score:
      return "lose", -bet
    return "push", 0
