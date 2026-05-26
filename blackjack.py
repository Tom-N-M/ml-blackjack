import random

class BlackjackEnv:
  def __init__(self, state_mode="basic"):
    self.state_mode = state_mode
    self.cards = ['ace', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'jack', 'queen', 'king']
    self.deck = self.cards * 4 * 4  # 4 Decks
    random.shuffle(self.deck)
    self.played_cards = []

  def _shuffle_deck(self):
    """Mischt das Deck neu"""
    self.deck = self.cards * 4 * 4
    random.shuffle(self.deck)
    self.played_cards = []

  def reset(self):
    """Teilt die Karten aus und gibt den Anfangszustand zurück."""
    self.max_deck_size = 52 * 4
    if len(self.played_cards) > 0.75 * self.max_deck_size:
      self._shuffle_deck()

    self.player_hand = [self.deck.pop()]
    self.dealer_hand = [self.deck.pop()]
    self.player_hand.append(self.deck.pop())
    self.dealer_hand.append(self.deck.pop())

    # Nur die erste Karte des Dealers wird offen gespielt
    self.played_cards.extend(self.player_hand + self.dealer_hand[:1])  
    
    return self._get_state()

  def _get_card_value(self, card):
    """Gibt den Wert einer Karte zurück."""
    if card in ['jack', 'queen', 'king']:
      return 10
    elif card == 'ace':
      return 11
    else:
      return int(card)

  def get_score(self, hand):
    """Berechnet den Score einer Hand."""
    score = sum(self._get_card_value(card) for card in hand)
    aces = sum(1 for card in hand if card == 'ace')
    
    # Wenn score über 21 ist und es Asse gibt, zähle die Asse als 1 statt 11
    while score > 21 and aces:
      score -= 10
      aces -= 1
    return score

  def _has_usable_ace(self, hand):
    score = sum(self._get_card_value(card) for card in hand)
    return 'ace' in hand and score <= 21

  def _get_card_counts(self):
      high = sum(1 for c in self.played_cards if self._get_card_value(c) >= 10)
      low = sum(1 for c in self.played_cards if self._get_card_value(c) <= 6)
      return high, low

  def _get_state(self):
    player_score = self.get_score(self.player_hand)
    dealer_card = self._get_card_value(self.dealer_hand[0])
    usable_ace = self._has_usable_ace(self.player_hand)
    
    base_state = (player_score, dealer_card, usable_ace)

    if self.state_mode == "basic":
        return base_state

    elif self.state_mode == "extended":
        high, low = self._get_card_counts()
        return base_state + (high, low)

    else:
        raise ValueError("Unknown state_mode")

  def step(self, action):
    """Führt die Aktion des Spielers aus und gibt den neuen Zustand, die Belohnung und ob das Spiel vorbei ist zurück."""
    if action == 0:  # Hit
      return self.hit()
    elif action == 1:  # Stand
      return self.stand()
    else:
      raise ValueError("Ungültige Aktion. Aktion muss 0 (Hit) oder 1 (Stand) sein.")

  def hit(self):
    """Fügt eine Karte zur Hand des Spielers hinzu."""
    card = self.deck.pop()
    self.player_hand.append(card)
    self.played_cards.append(card)
    player_score = self.get_score(self.player_hand)
    if player_score > 21:
      return self._get_state(), -1, True
    else:
      return self._get_state(), 0, False

  def stand(self):    
    """Der Spieler bleibt stehen, der Dealer spielt seine Hand aus."""
    player_score = self.get_score(self.player_hand)
    
    # Die vorher verdeckte Karte des Dealers wird jetzt offen gelegt
    self.played_cards.append(self.dealer_hand[1])
    
    while self.get_score(self.dealer_hand) < 17:
        card = self.deck.pop()
        self.dealer_hand.append(card)
        self.played_cards.append(card)
    
    dealer_score = self.get_score(self.dealer_hand)
    
    if dealer_score > 21 or player_score > dealer_score:
      reward = 1
    elif player_score < dealer_score:
      reward = -1
    else:
      reward = 0

    return self._get_state(), reward, True