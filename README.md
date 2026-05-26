# Blackjack

    $ git clone https://github.com/Tom-N-M/ml-blackjack.git
    $ cd ml-blackjack

Next, run the following commands:

    $ conda env create -f environment.yml
    $ conda activate ml-blackjack
    $ python -m ipykernel install --user --name=python3

Finally, start Jupyter:

    $ jupyter notebook

## Mathematischer Hintergrund: Q-Learning im Detail

Das Herzstück unseres Reinforcement Learning (RL) Agenten basiert auf **Tabular Q-Learning**, einem modellfreien, zeitlichen Differenzenverfahren (Temporal Difference Learning). Da der Zustandsraum beim Blackjack überschaubar ist, nutzt der Agent eine Tabelle (die sogenannte *Q-Tabelle*), um für jede Kombination aus Zustand und Aktion den erwarteten langfristigen Nutzen zu speichern.

### Die Bellman-Gleichung für Q-Learning

Nach jedem Schritt $t$, bei dem der Agent im Zustand $s$ die Aktion $a$ ausführt, eine Belohnung $r$ erhält und in den Folgezustand $s'$ übergeht, wird der entsprechende Eintrag in der Q-Tabelle nach folgender Update-Regel aktualisiert:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)$$

#### Erklärung der mathematischen Komponenten:

* **$Q(s, a)$**: Der aktuelle Qualitätswert (Q-Wert) für das Ausführen von Aktion $a$ im Zustand $s$. Er repräsentiert die erwartete kumulierte Belohnung bis zum Ende der Runde.
* **$\alpha$ (Alpha - Lernrate)**: Bestimmt, wie stark neue Informationen den alten Q-Wert überschreiben. 
    * Ein Wert von $\alpha = 0.01$ sorgt für ein stabiles, schrittweise Lernen über viele Episoden hinweg.
* **$r$ (Reward / Belohnung)**: Das unmittelbare Feedback der Umgebung auf die gewählte Aktion.
    * $+1.0$ bei einem Sieg
    * $-1.0$ bei einer Niederlage (oder beim Überkaufen)
    * $0.0$ bei einem Unentschieden (*Push*) oder wenn das Spiel nach einem *Hit* noch weiterläuft.
* **$\gamma$ (Gamma - Diskontierungsfaktor)**: Gewichtet die Bedeutung zukünftiger Belohnungen im Vergleich zu sofortigen Belohnungen. 
    * Da Blackjack ein episodisches Spiel mit sehr kurzen Runden ist und der Ausgang primär am Ende feststeht, setzen wir $\gamma = 1.0$ (keine Diskontierung zukünftiger Schritte).
* **$\max_{a'} Q(s', a')$**: Der maximale Q-Wert, der im nächsten Zustand $s'$ durch die theoretisch beste Folgeaktion $a'$ erreicht werden kann.
* **$\left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)$**: Der sogenannte **Temporal Difference Error (TD-Error)**. Er misst die Diskrepanz zwischen der neuen Schätzung des Nutzens und dem alten Q-Wert.

---

### Aktionsauswahl: Die $\epsilon$-Greedy-Strategie

Um die Balance zwischen dem Entdecken neuer Strategien (**Exploration**) und dem Nutzen bereits gelernter Pfade (**Exploitation**) zu meistern, verwendet der Agent eine Epsilon-Greedy-Policy:

$$\pi(a | s) = \begin{cases} 
\text{Zufällige Aktion } a \in \{0, 1\} & \text{mit Wahrscheinlichkeit } \epsilon \\ 
\arg\max_{a} Q(s, a) & \text{mit Wahrscheinlichkeit } 1 - \epsilon 
\end{cases}$$

* **Exploration ($\epsilon$)**: Mit einer Wahrscheinlichkeit von $\epsilon = 0.1$ wählt der Agent eine komplett zufällige Aktion (*Hit* oder *Stand*). Dies verhindert, dass sich der Agent zu früh auf eine suboptimale Strategie festlegt.
* **Exploitation ($1 - \epsilon$)**: Mit einer Wahrscheinlichkeit von $90\%$ wählt der Agent die Aktion, die laut Q-Tabelle aktuell den höchsten Erwartungswert hat.

---

### Der Zustandsraum (State Space)

Der Zustand $s$ wird in jeder Spielphase als 3-Tupel an den Agenten übergeben:

$$s = (\text{Score}_{\text{Spieler}}, \text{Karte}_{\text{Dealer}}, \text{Usable Ace})$$

1.  **Spieler-Score**: Eine diskrete Zahl von $4$ bis $21$.
2.  **Dealer-Karte**: Der sichtbare Wert der ersten Karte des Dealers ($2$ bis $11$).
3.  **Usable Ace**: Ein binärer Wert ($0$ oder $1$). Ein As ist "nutzbar" (*usable*), wenn es als $11$ gezählt werden kann, ohne dass der Spieler die $21$ überschreitet.