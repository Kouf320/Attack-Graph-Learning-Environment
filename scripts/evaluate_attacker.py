import torch
import json
import numpy as np
import networkx as nx
import re
from environment.graph_env import GraphEnvironment

MODEL_PATH = 'policy-models/paper/attacker/attacker_dqn_v1.pth'
CONFIG_PATH = 'ag.json'
NUM_EPISODES = 10
MAX_STEPS = 1000
TARGET_NODE = '307'   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def state_to_tensor(state, env):
    state_index = env.nodes.index(state)
    state_tensor = torch.zeros(env.num_nodes, device=DEVICE)
    state_tensor[state_index] = 1
    return state_tensor.unsqueeze(0)

def get_best_action(state, model, actions, env):
    """Selects the action with the highest Q-value (Epsilon = 0)"""
    with torch.no_grad():
        state_tensor = state_to_tensor(state, env)
        q_values = model(state_tensor)
        
        valid_indices = [env.nodes.index(a) for a in actions]
        valid_q_values = q_values[0, valid_indices]
        
        return actions[torch.argmax(valid_q_values).item()]

def run_evaluation():
    print(f"Loading environment from {CONFIG_PATH}...")
    with open(CONFIG_PATH, 'r') as f:
        json_data = json.load(f)
    env = GraphEnvironment(json_data)
    
    print(f"Loading model from {MODEL_PATH}...")
    model = torch.load(MODEL_PATH, weights_only=False)
    model.eval()

    success_count = 0
    all_rewards = []
    all_paths = []

    print(f"\nEvaluating {NUM_EPISODES} episodes...")
    print("-" * 50)

    for i in range(NUM_EPISODES):
        state = env.reset_recon(i)
        print("STATE ---->  " + env.graph.nodes[str(state)].get('name', 'Unknown'))
        episode_reward = 0
        path = [state]
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            steps += 1
            actions = env.get_actions(state)
            
            if not actions:
                break

            action = get_best_action(state, model, actions, env)
            next_state, reward, _, done = env.step(action)
            
            episode_reward += reward
            path.append(next_state)
            state = next_state

            if state == TARGET_NODE:
                success_count += 1
                done = True

        all_rewards.append(episode_reward)
        all_paths.append(path)
        
        status = "WIN" if state == TARGET_NODE else "FAIL"
        print(f"Episode {i+1:02d}: {status} | Steps: {steps:03d} | Reward: {episode_reward:7.2f}")

        print("\n")

    win_rate = (success_count / NUM_EPISODES) * 100
    avg_reward = np.mean(all_rewards)
    
    print("-" * 50)
    print(f"EVALUATION SUMMARY")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    
    if success_count > 0:
        winning_paths = [p for p in all_paths if p[-1] == TARGET_NODE]
        for win in winning_paths:
            print(str(win[0]))
            print("\nIdentified Optimal Attack Path for : " + str(win) )
            for idx, node in enumerate(win):
                name = env.graph.nodes[str(node)].get('name', 'Unknown')
                print(f"  Step {idx:02d}: {node} ({name})")

if __name__ == "__main__":
    run_evaluation()