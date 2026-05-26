import random

class BlackjackEnv:
  def __init__(self):
    self.cards = ['ace', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'jack', 'queen', 'king']
    self.colors = ['hearts', 'diamonds', 'clubs', 'spades']
    self.reset()

  def reset(self):
    """Setzt das Spiel für eine neue Runde zurück."""
    self.deck = [(card, color) for color in self.colors for card in self.cards]
    random.shuffle(self.deck)
    
    self.player_hand = [self.deck.pop(), self.deck.pop()]
    self.dealer_hand = [self.deck.pop(), self.deck.pop()]
    
    return self._get_state()

  def _get_card_value(self, card):
    """Gibt den Wert einer Karte zurück."""
    if card[0] in ['jack', 'queen', 'king']:
      return 10
    elif card[0] == 'ace':
      return 11
    else:
      return int(card[0])

  def get_score(self, hand):
    """Berechnet den Score einer Hand."""
    score = sum(self._get_card_value(card) for card in hand)
    aces = sum(1 for card in hand if card[0] == 'ace')
    
    # Wenn score über 21 ist und es Asse gibt, zähle die Asse als 1 statt 11
    while score > 21 and aces:
      score -= 10
      aces -= 1
    return score

  def _get_state(self):
    """Gibt den aktuellen Zustand für den RL-Agenten zurück."""
    # Zustand: (Spieler-Score, Dealer zeigt Karte X, Hat der Spieler ein nutzbares As?)
    usable_ace = 1 if (sum(1 for c in self.player_hand if c[0] == 'ace') > 0 and self.get_score(self.player_hand) <= 21) else 0
    return (self.get_score(self.player_hand), self._get_card_value(self.dealer_hand[0]), usable_ace)

  def hit(self):
    """Fügt eine Karte zur Hand des Spielers hinzu."""
    self.player_hand.append(self.deck.pop())
    player_score = self.get_score(self.player_hand)
    if player_score > 21:
      return self._get_state(), True
    else:
      return self._get_state(), False

  def stand(self):    
    """Der Spieler bleibt stehen, der Dealer spielt seine Hand aus."""
    player_score = self.get_score(self.player_hand)
    if player_score > 21:
      return self._get_state(), True
    
    dealer_score = self.get_score(self.dealer_hand)
    
    # Dealer muss laut Casino-Regeln bis mindestens 17 ziehen
    while dealer_score < 17:
        self.dealer_hand.append(self.deck.pop())

    return self._get_state(), True
