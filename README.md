# Blackjack

    $ git clone https://github.com/Tom-N-M/ml-blackjack.git
    $ cd ml-blackjack

Next, run the following commands:

    $ conda env create -f environment.yml
    $ conda activate ml-blackjack
    $ python -m ipykernel install --user --name=python3

Finally, start Jupyter:

    $ jupyter notebook

## Mathematical and Theoretical Background: Deep Dive into Q-Learning

At the core of our Reinforcement Learning (RL) agent lies **Tabular Q-Learning**, a model-free, Temporal Difference (TD) learning algorithm. Since the state space in Blackjack is relatively compact, the agent maintains a look-up table (the *Q-table*) to store the expected long-term utility for every unique combination of states and actions.

### The Bellman Optimality Equation for Q-Learning

After every time step $t$, where the agent transitions from state $s$ by executing action $a$, receives an immediate reward $r$, and moves into the subsequent state $s'$, the corresponding entry in the Q-table is updated using the following Bellman equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)$$

#### Breakdown of Mathematical Components:

* **$Q(s, a)$**: The current quality value (Q-value) for executing action $a$ in state $s$. It estimates the expected cumulative future reward from this point until the end of the episode.
* **$\alpha$ (Alpha - Learning Rate)**: Controls how much the newly acquired information overrides the old Q-value. 
    * Setting $\alpha = 0.01$ ensures stable, incremental learning across hundreds of thousands of training episodes.
* **$r$ (Reward)**: The immediate numerical scalar feedback returned by the environment as a direct consequence of the agent's action. Rewards are net profit relative to the current bet.
    * $+\text{bet}$ for a normal winning round.
    * $-\text{bet}$ for losing the round (or going *bust*).
    * $0.0$ for a tie (*push*) or when the game continues after a *hit*.
    * $+\text{bet} \times \text{blackjack_payout}$ for a natural Blackjack. The default payout multiplier is $1.5$.
* **$\gamma$ (Gamma - Discount Factor)**: Determines the present value of future rewards compared to immediate ones ($0 \le \gamma \le 1$). 
    * Because Blackjack is an episodic game with exceptionally short horizons, where the ultimate outcome is decided entirely at the final step, we set $\gamma = 1.0$ (no discounting of future steps).
* **$\max_{a'} Q(s', a')$**: The maximum estimated Q-value achievable in the next state $s'$ by selecting the theoretically optimal subsequent action $a'$.
* **$\left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)$**: The **Temporal Difference Error (TD-Error)**. It quantifies the difference between the updated target estimation of utility and the current value in the table.

---

### Action Selection: The $\epsilon$-Greedy Policy

To strike an optimal balance between discovering novel strategies (**Exploration**) and exploiting the agent's current knowledge base (**Exploitation**), an $\epsilon$-greedy policy is implemented:

$$\pi(a | s) = \begin{cases} 
\text{Random action } a \in \{0, 1\} & \text{with probability } \epsilon \\ 
\arg\max_{a} Q(s, a) & \text{with probability } 1 - \epsilon 
\end{cases}$$

* **Exploration ($\epsilon$)**: With a probability of $\epsilon = 0.1$, the agent chooses an action at random (*hit* or *stand*), regardless of historical performance. This prevents the policy from prematurely converging on a sub-optimal local minimum.
* **Exploitation ($1 - \epsilon$)**: With a probability of $90\%$, the agent greedily selects the action that yields the highest expected value according to the Q-table.

---

### The State Space

At each decision boundary, the current game configuration $s$ is formalized and exposed to the agent as a discrete 3-tuple:

$$s = (\text{Player's Score}, \text{Dealer's Showing Card}, \text{Usable Ace})$$

1.  **Player's Score**: An integer representing the current total card value ranging from $4$ to $21$.
2.  **Dealer's Showing Card**: The value of the single face-up card held by the dealer ($2$ to $11$, where an Ace counts as $11$).
3.  **Usable Ace**: A binary flag ($0$ or $1$). An Ace is considered "usable" if it can be valued at $11$ without causing the player's total score to exceed $21$.

### Bets and Bankroll

Each round can be started with a specific bet:

```python
env = BlackjackEnv(bankroll=100, min_bet=1, max_bet=20)
state = env.reset(bet=5)
next_state, reward, done = env.step(1)
```

If no bet is passed to `reset`, the environment uses `min_bet`. When `bankroll` is set, finished rounds update `env.bankroll` by the net reward. The latest result is also available as `env.last_outcome` and `env.last_profit`.
