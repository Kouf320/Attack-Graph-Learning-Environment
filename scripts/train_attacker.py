import networkx as nx
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from environment.graph_env import GraphEnvironment
from agents.attacker import dqn_network as dqn
from memory.replay_buffer import ReplayMemory as rm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import networkx as nx

DATE_FORMAT = "%m-%d %H:%M:%S"

def state_to_tensor(state, env, device):
    state_index = env.nodes.index(state)
    state_tensor = torch.zeros(env.num_nodes, device=device)
    state_tensor[state_index] = 1
    return state_tensor

def select_action(state, model, epsilon, actions, env, device):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        with torch.no_grad():
            state_tensor = state_to_tensor(state, env, device).unsqueeze(0)
            q_values = model(state_tensor)
            return actions[torch.argmax(q_values[0, [env.nodes.index(a) for a in actions]]).item()]

def train_dqn(environment, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, location):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plt_y = []
    plt_x = []
    env = environment
    input_dim = env.num_nodes
    output_dim = env.num_nodes

    episode_rewards = []
    episode_lengths = []
    epsilon_values = []

    model = dqn.DQNetwork(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    target_model = copy.deepcopy(model)
    target_model.eval()

    memory_capacity = 100000
    memory = rm(memory_capacity)
    tau = 0.005
    
    recent_rewards = deque(maxlen=100)
    recent_losses = deque(maxlen=100) 
    recent_successes = deque(maxlen=100)
    recent_steps = deque(maxlen=100)
    
    best_reward = -999999
    
    epsilon = epsilon_start
    
    for episode in range(num_episodes):

        state = env.reset_recon(episode)
        
        DebugLogger.log_episode_start(episode, state)

        visited_nodes_in_episode = {state} 
        nodes = []
        episode_loss = 0
        episode_reward = 0
        
        termination_reason = "Max Steps Reached" 
        is_success = False 

        for t in range(200):  

            actions = env.get_actions(state)
            nodes.append(state)  
            
            action = None
            current_q_val = 0.0 
            
            if not actions:
                if state != env.end_node:
                    reward = -10
                    termination_reason = "Dead End (Penalty)"
                    DebugLogger.log_step(t, state, "NONE", reward, "NONE", 0.0, epsilon, True)
                else:
                    termination_reason = "Goal Reached"
                    is_success = True
                break
            else:
                if random.random() < epsilon:
                    action = random.choice(actions)
                    current_q_val = 0.0 
                else:
                    with torch.no_grad():
                        state_tensor = state_to_tensor(state, env, device).unsqueeze(0)
                        q_values = model(state_tensor)
                        action_idx = torch.argmax(q_values[0, [env.nodes.index(a) for a in actions]]).item()
                        action = actions[action_idx]
                        current_q_val = q_values[0, env.nodes.index(action)].item()

                next_state, reward, alerts, done = env.step(action)
                reward = (reward * 0.1) - t * 0.075 

            if next_state in visited_nodes_in_episode:
                reward = -5
            else:
                visited_nodes_in_episode.add(next_state)

            if env.graph.nodes[str(state)].get('name','').startswith('HostCompromise') and state != env.end_node:
                reward = 1
            
            DebugLogger.log_step(t, state, action, reward, next_state, current_q_val, epsilon, done)
            
            episode_reward += reward 
            memory.push(state, action, reward, next_state, done)
            
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                states, actions_b, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.stack([state_to_tensor(s, env, device) for s in states])
                actions_tensor = torch.tensor([env.nodes.index(a) for a in actions_b], dtype=torch.long, device=device).unsqueeze(1)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                dones_tensor = torch.tensor(dones ,dtype=torch.float32, device=device)
                next_states_tensor = torch.stack([state_to_tensor(ns, env, device) for ns in next_states])
                
                q_values = model(states_tensor).gather(1, actions_tensor)
                
                with torch.no_grad():
                    next_q_values_all = target_model(next_states_tensor).detach()                
                    
                    invalid_action_mask = torch.full_like(next_q_values_all, -float('inf'))

                    for i, ns in enumerate(next_states):
                        if dones[i]:
                            invalid_action_mask[i, :] = -float('inf') 
                        else:
                            valid_actions_b = env.get_actions(ns)
                            if valid_actions_b:
                                valid_indices = [env.nodes.index(a) for a in valid_actions_b]
                                invalid_action_mask[i, valid_indices] = 0.0

                    masked_next_q_values = next_q_values_all + invalid_action_mask
                    next_q_values = masked_next_q_values.max(1)[0]
                    next_q_values[next_q_values == -float('inf')] = 0.0

                target_q_values = rewards_tensor + (gamma * next_q_values) * (1- dones_tensor)

                loss = nn.functional.mse_loss(q_values, target_q_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                episode_loss += loss.item()
            
            state = next_state
            if done:
                termination_reason = "Goal Reached / Terminal State"
                is_success = True
                break
        
        DebugLogger.log_episode_summary(episode, episode_reward, t+1, episode_loss, termination_reason)

        episode_lengths.append(t + 1)  
        epsilon_values.append(epsilon)
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * episode / epsilon_decay)
        
        plt_x.append(episode+1)
        plt_y.append(episode_reward)
        episode_rewards.append(episode_reward)
        
        recent_rewards.append(episode_reward)
        recent_losses.append(episode_loss)
        recent_steps.append(t + 1)
        recent_successes.append(1 if is_success else 0)

        with torch.no_grad():
            for target_param, local_param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        if (episode + 1) % 100 == 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_loss = sum(recent_losses) / len(recent_losses)
            wins = sum(recent_successes)
            fails = len(recent_successes) - wins
            avg_steps = sum(recent_steps) / len(recent_steps)
            
            DebugLogger.log_period_summary(
                episode - 98, episode + 1, num_episodes, 
                avg_reward, avg_loss, epsilon, len(memory), memory_capacity,
                wins, fails, avg_steps
            )

            if avg_reward > best_reward and best_reward != -999999:
                print(f"{DebugLogger.WARNING} >>> New Best Reward: {avg_reward:.2f} (Saving Model) <<<{DebugLogger.ENDC}")
                torch.save(model, location+"_final")
                best_reward = avg_reward
            elif best_reward == -999999:
                 best_reward = avg_reward

    
    q_value_log = []
    state_to_track = env.reset()  
    state_tensor = state_to_tensor(state_to_track, env, device)
    q_values = model(state_tensor)
    q_value_log.append(q_values.detach().cpu().numpy())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 60))
    plot_full_diagnostics(
        episode_rewards,
        episode_lengths,
        epsilon_values,
        model,
        env,
        device,
        window_size=100, 
        save_path=f"plots/diagnostics.png"
    )
    
    return model

def plot_full_diagnostics(
    episode_rewards, 
    episode_lengths, 
    epsilon_values, 
    model, 
    env, 
    device, 
    window_size=100, 
    save_path="plots/full_diagnostics-attacker.png"
):
    
    print(f"Generating diagnostic plots... saving to {save_path}")


    rewards_series = pd.Series(episode_rewards)
    lengths_series = pd.Series(episode_lengths)
    
    smoothed_rewards = rewards_series.rolling(window=window_size, min_periods=window_size).mean()
    smoothed_lengths = lengths_series.rolling(window=window_size, min_periods=window_size).mean()
    
    episodes = list(range(len(episode_rewards)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 24))
    
    ax1.set_title(f'Training Performance (Smoothed over {window_size} episodes)', fontsize=16)
    ax1.set_xlabel('Episode', fontsize=12)

    color = 'tab:blue'
    ax1.set_ylabel('Smoothed Reward', color=color, fontsize=12)
    ax1.plot(episodes, smoothed_rewards, color=color, label='Smoothed Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax1_twin_len = ax1.twinx()
    color = 'tab:green'
    ax1_twin_len.set_ylabel('Smoothed Episode Length', color=color, fontsize=12)
    ax1_twin_len.plot(episodes, smoothed_lengths, color=color, label='Smoothed Length')
    ax1_twin_len.tick_params(axis='y', labelcolor=color)

    ax1_twin_eps = ax1.twinx()

    ax1_twin_eps.spines['right'].set_position(('outward', 60))
    color = 'tab:red'
    ax1_twin_eps.set_ylabel('Epsilon', color=color, fontsize=12)
    ax1_twin_eps.plot(episodes, epsilon_values, color=color, linestyle='--', label='Epsilon')
    ax1_twin_eps.tick_params(axis='y', labelcolor=color)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin_len.get_legend_handles_labels()
    lines3, labels3 = ax1_twin_eps.get_legend_handles_labels()
    ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper center')


    if not hasattr(env, 'graph'):
        print("Warning: env.graph not found. Policy plot will be empty.")
        ax2.set_title('Policy and Value Map (Error: env.G not found)')
    else:
        G = env.graph
        pos = nx.kamada_kawai_layout(G)
        
        node_values = {}
        best_actions = {}

        for node in env.nodes:

            state_tensor = state_to_tensor(node, env, device).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)[0]
            
            node_values[node] = q_values.max().item()

            valid_actions = env.get_actions(node)
            if valid_actions:
                valid_indices = [env.nodes.index(a) for a in valid_actions]
                valid_q_values = q_values[valid_indices]
                best_action_node = valid_actions[valid_q_values.argmax().item()]
                best_actions[node] = best_action_node

        ax2.set_title('Learned Policy and State-Value Function', fontsize=16)
        
        values_list = [node_values.get(node, 0) for node in G.nodes()]
        
        nodes = nx.draw_networkx_nodes(
            G, pos, 
            node_color=values_list, 
            cmap=plt.cm.coolwarm, 
            ax=ax2,
            node_size=200
        )
        
        nx.draw_networkx_edges(
            G, pos, 
            ax=ax2, 
            edge_color='gray', 
            alpha=0.3
        )
        
        policy_edges = [(node, best_actions[node]) for node in best_actions if node in G]
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=policy_edges, 
            edge_color='black',
            style='dashed',
            ax=ax2,
            arrows=True,  
            arrowsize=15
        )
        
        nx.draw_networkx_labels(G, pos, ax=ax2, font_size=8)
        
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.coolwarm, 
            norm=plt.Normalize(vmin=min(values_list), vmax=max(values_list))
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2, orientation='vertical', pad=0.02)
        cbar.set_label('State-Value Estimate (V(s))', fontsize=12)
        
        ax2.axis('off')

    fig.tight_layout(pad=3.0)
    plt.savefig(save_path)
    plt.close(fig) 
    print("Diagnostic plots saved.")

class DebugLogger:
    DEBUG = False 

    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def log_step(step, state, action, reward, next_state, q_val, epsilon, done):
        if DebugLogger.DEBUG:
            r_color = DebugLogger.GREEN if reward > 0 else (DebugLogger.FAIL if reward < 0 else DebugLogger.BLUE)
            print(f"  {DebugLogger.BOLD}Step {step:03d}{DebugLogger.ENDC} | "
                  f"State: {DebugLogger.CYAN}{state:<5}{DebugLogger.ENDC} -> "
                  f"Action: {DebugLogger.CYAN}{action:<5}{DebugLogger.ENDC} -> "
                  f"Next: {DebugLogger.CYAN}{next_state:<5}{DebugLogger.ENDC} | "
                  f"R: {r_color}{reward:6.2f}{DebugLogger.ENDC} | "
                  f"Q(s,a): {q_val:6.2f} | "
                  f"Eps: {epsilon:.2f} | "
                  f"{'DONE' if done else ''}")

    @staticmethod
    def log_episode_start(episode, start_node):
        if DebugLogger.DEBUG:
            print(f"{DebugLogger.HEADER}{'='*60}")
            print(f"EPISODE {episode} STARTED | Start Node: {start_node}")
            print(f"{'='*60}{DebugLogger.ENDC}")

    @staticmethod
    def log_episode_summary(episode, total_reward, steps, loss, reason):
        if DebugLogger.DEBUG:
            color = DebugLogger.GREEN if total_reward > 0 else DebugLogger.FAIL
            print(f"{DebugLogger.HEADER}{'-'*60}{DebugLogger.ENDC}")
            print(f"EPISODE {episode} ENDED ({reason}) | "
                  f"Steps: {steps} | "
                  f"Total Reward: {color}{total_reward:.2f}{DebugLogger.ENDC} | "
                  f"Total Loss: {loss:.4f}")
            print(f"\n")

    @staticmethod
    def log_period_summary(start_ep, end_ep, total_ep, avg_reward, avg_loss, epsilon, mem_size, mem_capacity, success_count, fail_count, avg_steps):
        r_color = DebugLogger.GREEN if avg_reward > 0 else DebugLogger.FAIL
        s_color = DebugLogger.GREEN if success_count > 0 else DebugLogger.FAIL
        
        print(f"{DebugLogger.BLUE}[PERIOD SUMMARY] Episodes {start_ep}-{end_ep}/{total_ep}{DebugLogger.ENDC} | "
              f"Wins: {s_color}{success_count}{DebugLogger.ENDC} (Fails: {fail_count}) | "
              f"Avg Steps: {avg_steps:.1f} | "
              f"Avg R: {r_color}{avg_reward:6.2f}{DebugLogger.ENDC} | "
              f"Mem: {mem_size}/{mem_capacity} | "
              f"Eps: {epsilon:.2f} | "
              f"Loss: {avg_loss:.4f}")