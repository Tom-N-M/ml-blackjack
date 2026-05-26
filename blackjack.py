import random


class BlackjackEnv:
  def __init__(self, state_mode="basic", num_decks=4, stand_on_soft_17=True):
    self.state_mode = state_mode
    self.num_decks = num_decks
    self.stand_on_soft_17 = stand_on_soft_17
    self.cards = ['ace', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'jack', 'queen', 'king']
    self.max_deck_size = 52 * self.num_decks
    self._shuffle_deck()

  def _shuffle_deck(self):
    """Reset and shuffle the shoe."""
    self.deck = self.cards * 4 * self.num_decks
    random.shuffle(self.deck)
    self.played_cards = []
    self.cards_dealt = 0

  def _draw_card(self):
    if not self.deck:
      raise RuntimeError("Shoe is empty. Shuffle before starting the next round.")

    self.cards_dealt += 1
    return self.deck.pop()

  def _mark_seen(self, card):
    self.played_cards.append(card)

  def reset(self):
    """Deal a new round and return the initial decision state."""
    if len(self.deck) <= 0.25 * self.max_deck_size:
      self._shuffle_deck()

    self.done = False
    self.pending_reward = None
    self.dealer_hole_revealed = False

    player_first = self._draw_card()
    dealer_upcard = self._draw_card()
    player_second = self._draw_card()
    dealer_hole = self._draw_card()

    self.player_hand = [player_first, player_second]
    self.dealer_hand = [dealer_upcard, dealer_hole]

    for card in [player_first, dealer_upcard, player_second]:
      self._mark_seen(card)

    player_blackjack = self._is_natural_blackjack(self.player_hand)
    dealer_blackjack = self._is_natural_blackjack(self.dealer_hand)
    dealer_peeks = self._get_card_value(dealer_upcard) in (10, 11)

    if dealer_peeks and dealer_blackjack:
      self._reveal_dealer_hole()
      self.done = True
      self.pending_reward = 0 if player_blackjack else -1
    elif player_blackjack:
      self.done = True
      self.pending_reward = 1

    return self._get_state()

  def _get_card_value(self, card):
    """Return a card's blackjack value before ace reductions."""
    if card in ['jack', 'queen', 'king']:
      return 10
    elif card == 'ace':
      return 11
    else:
      return int(card)

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

    player_score = self.get_score(self.player_hand)
    dealer_card = self._get_card_value(self.dealer_hand[0])
    usable_ace = self._has_usable_ace(self.player_hand)

    base_state = (player_score, dealer_card, usable_ace)

    if self.state_mode == "basic":
      return base_state

    if self.state_mode == "extended":
      return base_state + (self._get_true_count_bucket(),)

    raise ValueError("Unknown state_mode")

  def step(self, action):
    """Apply one player action and return state, reward, done."""
    if self.done:
      return self._get_state(), self.pending_reward, True

    if action == 0:  # Hit
      return self.hit()
    elif action == 1:  # Stand
      return self.stand()
    else:
      raise ValueError("Invalid action. Use 0 (Hit) or 1 (Stand).")

  def hit(self):
    """Add one visible card to the player's hand."""
    card = self._draw_card()
    self.player_hand.append(card)
    self._mark_seen(card)

    if self.get_score(self.player_hand) > 21:
      self.done = True
      self.pending_reward = -1
      return self._get_state(), -1, True

    return self._get_state(), 0, False

  def _dealer_should_hit(self):
    dealer_score, dealer_soft = self._score_and_soft(self.dealer_hand)
    if dealer_score < 17:
      return True
    return dealer_score == 17 and dealer_soft and not self.stand_on_soft_17

  def stand(self):
    """Resolve the dealer hand and score the round."""
    self._reveal_dealer_hole()

    while self._dealer_should_hit():
      card = self._draw_card()
      self.dealer_hand.append(card)
      self._mark_seen(card)

    player_score = self.get_score(self.player_hand)
    dealer_score = self.get_score(self.dealer_hand)

    if self.get_score(self.player_hand) > 21:
      reward = -1
    elif self._is_natural_blackjack(self.player_hand):
      reward = 1
    elif dealer_score > 21 or player_score > dealer_score:
      reward = 1
    elif player_score < dealer_score:
      reward = -1
    else:
      reward = 0

    self.done = True
    self.pending_reward = reward
    return self._get_state(), reward, True
