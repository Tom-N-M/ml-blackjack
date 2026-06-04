import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches  

def get_basic_strategy_action(player_total: int, dealer_upcard: int, usable_ace: int) -> int:
    if usable_ace:
        if player_total >= 19: return 0
        if player_total == 18: return 0 if dealer_upcard <= 8 else 1
        return 1
    else:
        if player_total >= 17: return 0
        if 13 <= player_total <= 16: return 0 if dealer_upcard <= 6 else 1
        if player_total == 12: return 0 if 4 <= dealer_upcard <= 6 else 1
        return 1

def extract_agent_data(agent, agent_type: str, player_total: int, dealer_upcard: int, usable_ace: int, true_count: int = 0):
    if agent_type == "baseline":
        state = (player_total, dealer_upcard, usable_ace)
    else:
        cards_remaining_clipped = 4
        running_count = int(np.clip(true_count * cards_remaining_clipped, -20, 20))
        state = (player_total, dealer_upcard, usable_ace, running_count, true_count, cards_remaining_clipped)
    
    q_table = getattr(agent, "q_values", getattr(agent, "Q", None))
    if q_table is None: return 0, [0.0, 0.0]
    
    values = q_table.get(state)
    if values is None: return 0, [0.0, 0.0]
    return int(np.argmax(values)), values

def plot_policy_and_q_values(agent, agent_name: str, split_agent_name_func, save_fig_func, true_count: int = 0):
    agent_type, _ = split_agent_name_func(agent_name)
    player_totals = np.arange(12, 22)
    dealer_cards = np.arange(2, 12)
    shape = (len(player_totals), len(dealer_cards))
    
    policy_hard, policy_soft = np.zeros(shape), np.zeros(shape)
    q_diff_hard, q_diff_soft = np.zeros(shape), np.zeros(shape)
    match_basic_hard, match_basic_soft = np.zeros(shape), np.zeros(shape)
    
    for i, p in enumerate(player_totals):
        for j, d in enumerate(dealer_cards):
            act_h, val_h = extract_agent_data(agent, agent_type, p, d, 0, true_count)
            policy_hard[i, j] = act_h
            q_diff_hard[i, j] = val_h[1] - val_h[0]
            match_basic_hard[i, j] = 1 if act_h == get_basic_strategy_action(p, d, 0) else 0
            
            act_s, val_s = extract_agent_data(agent, agent_type, p, d, 1, true_count)
            policy_soft[i, j] = act_s
            q_diff_soft[i, j] = val_s[1] - val_s[0]
            match_basic_soft[i, j] = 1 if act_s == get_basic_strategy_action(p, d, 1) else 0

    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    extent = [1.5, 11.5, 11.5, 21.5]
    cmap_policy = mcolors.ListedColormap(['#e74c3c', '#2ecc71'])
    cmap_match = mcolors.ListedColormap(['#d35400', '#27ae60'])
    
    axs[0, 0].imshow(policy_hard, cmap=cmap_policy, origin='lower', extent=extent, aspect='auto')
    axs[0, 0].set_title(f"Policy Map - Hard Totals\nAgent: {agent_name}")
    axs[0, 1].imshow(policy_soft, cmap=cmap_policy, origin='lower', extent=extent, aspect='auto')
    axs[0, 1].set_title(f"Policy Map - Soft Totals\nAgent: {agent_name}")
    
    for i, p in enumerate(player_totals):
        for j, d in enumerate(dealer_cards):
            axs[0, 0].text(d, p, 'H' if policy_hard[i, j] == 1 else 'S', ha='center', va='center', color='white', weight='bold')
            axs[0, 1].text(d, p, 'H' if policy_soft[i, j] == 1 else 'S', ha='center', va='center', color='white', weight='bold')

    im_qh = axs[1, 0].imshow(q_diff_hard, cmap='RdYlGn', origin='lower', extent=extent, aspect='auto')
    axs[1, 0].set_title("Q-Value Confidence ($Q(Hit) - Q(Stand)$) - Hard")
    fig.colorbar(im_qh, ax=axs[1, 0])
    
    im_qs = axs[1, 1].imshow(q_diff_soft, cmap='RdYlGn', origin='lower', extent=extent, aspect='auto')
    axs[1, 1].set_title("Q-Value Confidence ($Q(Hit) - Q(Stand)$) - Soft")
    fig.colorbar(im_qs, ax=axs[1, 1])

    axs[2, 0].imshow(match_basic_hard, cmap=cmap_match, origin='lower', extent=extent, aspect='auto')
    axs[2, 0].set_title("Vergleich mit Basic Strategy (Hard)")
    axs[2, 1].imshow(match_basic_soft, cmap=cmap_match, origin='lower', extent=extent, aspect='auto')
    axs[2, 1].set_title("Vergleich mit Basic Strategy (Soft)")

    for row in axs:
        for ax in row:
            ax.set_xticks(dealer_cards)
            ax.set_xticklabels([str(x) if x < 11 else 'A' for x in dealer_cards])
            ax.set_yticks(player_totals)
            ax.set_ylabel("Player Hand Total")
            ax.set_xlabel("Dealer Upcard")
            ax.grid(False)

    patch_stick = mpatches.Patch(color='#e74c3c', label='Stick / Halten (S)')
    patch_hit = mpatches.Patch(color='#2ecc71', label='Hit / Ziehen (H)')
    patch_mismatch = mpatches.Patch(color='#d35400', label='Abweichung')
    patch_match = mpatches.Patch(color='#27ae60', label='Konform')
    
    plt.tight_layout()
    fig.subplots_adjust(
        hspace=0.15,
        wspace=0.05,
        bottom=0.08
    )

    axs[0, 0].legend(
        handles=[patch_stick, patch_hit],
        loc="upper center",
        bbox_to_anchor=(1.05, -0.12),
        ncol=2,
        fontsize=11,
        frameon=False,
        title="Aktionen"
    )

    axs[2, 0].legend(
        handles=[patch_mismatch, patch_match],
        loc="upper center",
        bbox_to_anchor=(1.05, -0.12),
        ncol=2,
        fontsize=11,
        frameon=False,
        title="Strategie-Vergleich"
    )

    tc_suffix = f"_tc_{true_count}" if agent_type == "counting" else ""
    save_fig_func(f"agent_policy_analysis_{agent_name}{tc_suffix}")
    plt.show()