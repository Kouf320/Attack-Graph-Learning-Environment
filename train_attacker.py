import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import json
import os
from collections import deque
from datetime import datetime

from environment import GraphEnvironment
from dqn_model import DQNetwork
from replay_memory import ReplayMemory

# Configuration
CONFIG_PATH = 'config.json'
GRAPH_PATH = 'ag.json'
MODEL_SAVE_PATH = 'attacker_model.pth'
LOG_DIR = 'logs'

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
LR = 1e-4
MEMORY_CAPACITY = 10000
NUM_EPISODES = 500
TARGET_UPDATE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json_graph(path):
    with open(path, 'r') as f:
        return json.load(f)

def state_to_tensor(state, env):
    # One-hot encoding of the current node
    state_index = env.nodes.index(state)
    state_tensor = torch.zeros(env.num_nodes, device=device)
    state_tensor[state_index] = 1
    return state_tensor

def select_action(state, model, epsilon, env):
    actions = env.get_actions(state)
    if not actions:
        return None

    if random.random() < epsilon:
        return random.choice(actions)
    else:
        with torch.no_grad():
            state_tensor = state_to_tensor(state, env).unsqueeze(0)
            q_values = model(state_tensor)
            # Filter Q-values for valid actions only
            valid_indices = [env.nodes.index(a) for a in actions]
            q_values_valid = q_values[0, valid_indices]
            best_action_idx = torch.argmax(q_values_valid).item()
            return actions[best_action_idx]

def train():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    print(f"Loading graph from {GRAPH_PATH}...")
    graph_data = load_json_graph(GRAPH_PATH)
    env = GraphEnvironment(graph_data)
    
    input_dim = env.num_nodes
    output_dim = env.num_nodes # Output is Q-value for each node (as a potential next step)

    print(f"Environment initialized with {input_dim} nodes.")

    policy_net = DQNetwork(input_dim, output_dim).to(device)
    target_net = DQNetwork(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_CAPACITY)

    epsilon = EPSILON_START
    
    best_reward = -float('inf')

    print("Starting training...")
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Limit steps per episode to prevent infinite loops
        for t in range(100):
            action = select_action(state, policy_net, epsilon, env)
            
            if action is None:
                break # No valid moves

            next_state, reward, alerts, done = env.step(action)
            episode_reward += reward

            # Store in memory
            # We store state/action names or indices? 
            # ReplayMemory expects standard objects. 
            # But for training batch, we need to convert to tensors.
            memory.push(state, action, reward, next_state, done)

            state = next_state

            # Optimization Step
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                # Transpose the batch
                batch = list(zip(*transitions))
                
                state_batch = torch.stack([state_to_tensor(s, env) for s in batch[0]])
                action_batch = torch.tensor([env.nodes.index(a) for a in batch[1]], device=device).unsqueeze(1)
                reward_batch = torch.tensor(batch[2], device=device)
                non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch[4])), device=device, dtype=torch.bool)
                
                # Compute Q(s, a)
                state_action_values = policy_net(state_batch).gather(1, action_batch)

                # Compute V(s_{t+1}) for all next states.
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                
                # We need to handle next_states efficiently.
                # For this simple implementation, let's re-convert next_states to tensors
                # A more optimized way would be to store indices in memory.
                non_final_next_states = [s for s, d in zip(batch[3], batch[4]) if not d]
                
                if non_final_next_states:
                    non_final_next_states_tensor = torch.stack([state_to_tensor(s, env) for s in non_final_next_states])
                    next_state_values[non_final_mask] = target_net(non_final_next_states_tensor).max(1)[0].detach()

                expected_state_action_values = (next_state_values * GAMMA) + reward_batch

                # Huber Loss
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            if done:
                break

        # Update Target Network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay Epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.4f}")
            
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
            #print(f"  -> New best model saved!")

    print("Training Complete.")
    torch.save(policy_net.state_dict(), "final_attacker_model.pth")

if __name__ == "__main__":
    train()
