# utils/training.py
import time
from datetime import timedelta
from pathlib import Path
import gymnasium as gym
import numpy as np
from multiprocess import Pool, Manager
import ipywidgets as widgets
from IPython.display import display
from env.blackjack_env import BlackjackEnv  
from utils.env_utils import make_blackjack_env 

class ProgressWrapper(gym.Wrapper):
    def __init__(self, env, agent_name, progress_dict, start_episode):
        super().__init__(env)
        self.agent_name = agent_name
        self.progress_dict = progress_dict
        self.episode_count = start_episode
        self.progress_dict[self.agent_name] = self.episode_count

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Wenn eine Runde Blackjack vorbei ist
        if terminated or truncated:
            self.episode_count += 1
            # Nicht bei jedem Schritt funken (Performance!), sondern alle 5000 Runden
            if self.episode_count % 5000 == 0:
                self.progress_dict[self.agent_name] = self.episode_count
        return obs, reward, terminated, truncated, info

def train_single_agent(
        name,
        agent,
        agent_type,
        seed,
        selected_artifact,
        episodes_per_seed,
        checkpoint_interval,
        checkpoint_dir,
        run_id,
        progress_dict=None
    ):
    # Nutzen jetzt die zentrale Factory-Funktion statt Hardcoding!
    env = make_blackjack_env(seed=seed, n_episodes=episodes_per_seed)

    start_episode = 0
    if selected_artifact is not None:
        loaded_artifact = agent.load(selected_artifact)
        start_episode = int(loaded_artifact.get("episode") or 0)

    if progress_dict is not None:
        env = ProgressWrapper(env, name, progress_dict, start_episode)

    agent.env = env
    # Die Seeding-Logik ist bereits in make_blackjack_env abgedeckt!
    
    episodes_to_train = max(0, episodes_per_seed - start_episode)
    if episodes_to_train == 0:
        if progress_dict is not None: progress_dict[name] = episodes_per_seed
        return name, agent

    agent.train(
        n_episodes=episodes_to_train,
        base_seed=seed,
        start_episode=start_episode,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        checkpoint_label=f"{run_id}_{name}_agent",
        checkpoint_metadata={"agent_name": name, "agent_type": agent_type, "run_id": run_id, "seed": seed, "start_episode": start_episode, "target_episode": episodes_per_seed},
    )
    
    if progress_dict is not None: progress_dict[name] = episodes_per_seed
    return name, agent

def run_parallel_with_dashboard(worker_func, base_tasks, agent_names, max_value_per_agent, title="Prozesse"):
    """
    Führt Aufgaben parallel aus und zeichnet ein Live-Dashboard im Jupyter Notebook.
    
    :param worker_func: Die Zielfunktion (z.B. train_single_agent)
    :param base_tasks: Liste von Tupeln mit Argumenten (OHNE das progress_dict)
    :param agent_names: Liste der Agenten-Namen für die Beschriftung
    :param max_value_per_agent: Ziel-Episodenanzahl pro Agent (z.B. EPISODES_PER_SEED)
    :param title: Name der Operation im Gesamtbalken
    """
    def format_time(secs):
        return "--:--" if secs < 0 else str(timedelta(seconds=int(secs)))

    print(f"Initialisiere paralleles {title}-Dashboard...")
    
    manager = Manager()
    progress_dict = manager.dict()
    for name in agent_names:
        progress_dict[name] = 0
        
    total_target = max_value_per_agent * len(agent_names)
        
    # Widgets bauen
    overall_bar = widgets.IntProgress(value=0, min=0, max=total_target, description=f"{title} Gesamt:", style={'description_width': '140px', 'bar_color': '#28a745'}, layout=widgets.Layout(width='400px'))
    overall_label = widgets.Label(value="Warte auf Start...")
    widget_rows = [widgets.HBox([overall_bar, overall_label]), widgets.HTML("<hr style='margin: 10px 0; border: 1px solid #ccc;'>")]
    
    bars, labels = {}, {}
    for name in agent_names:
        bars[name] = widgets.IntProgress(value=0, min=0, max=max_value_per_agent, description=f"{name}:", style={'description_width': '140px'}, layout=widgets.Layout(width='400px'))
        labels[name] = widgets.Label(value="Warte...")
        widget_rows.append(widgets.HBox([bars[name], labels[name]]))
        
    display(widgets.VBox(widget_rows))
    
    # Injektiere das progress_dict in alle Tasks am Ende des Tupels
    final_tasks = [tuple(list(task) + [progress_dict]) for task in base_tasks]

    start_time = time.time()
    with Pool() as pool:
        async_result = pool.starmap_async(worker_func, final_tasks)
        
        while not async_result.ready():
            elapsed = time.time() - start_time
            total_done = 0
            
            for name in agent_names:
                curr = progress_dict.get(name, 0)
                total_done += curr
                bars[name].value = curr
                pct = curr / max_value_per_agent
                eta = f"Restzeit: {format_time((elapsed / pct) - elapsed)}" if pct > 0 and curr < max_value_per_agent else "Fertig!" if curr >= max_value_per_agent else "Berechne..."
                labels[name].value = f"{pct*100:.1f}% ({curr:,}/{max_value_per_agent:,}) | {eta}"
            
            overall_bar.value = total_done
            o_pct = total_done / total_target
            o_eta = f"Gesamt-Restzeit: {format_time((elapsed / o_pct) - elapsed)}" if o_pct > 0 else "Berechne..."
            overall_label.value = f"{o_pct*100:.1f}% ({total_done:,}/{total_target:,}) | {o_eta}"
            
            time.sleep(1.0)
            
        # Finaler Clean-up der UI
        for name in agent_names:
            bars[name].value = max_value_per_agent
            labels[name].value = f"100.0% ({max_value_per_agent:,}) | Fertig!"
        overall_bar.value = total_target
        overall_label.value = f"100.0% ({total_target:,}) | Fertig!"
        
        results = async_result.get()
        
    print(f"\nProzess erfolgreich beendet! Gesamtdauer: {format_time(time.time() - start_time)}")
    return results