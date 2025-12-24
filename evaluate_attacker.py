import torch
import json
import random
import sys
from environment import GraphEnvironment
from dqn_model import DQNetwork

GRAPH_PATH = 'ag.json'
MODEL_PATH = 'attacker_model.pth' # Default model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json_graph(path):
    with open(path, 'r') as f:
        return json.load(f)

def state_to_tensor(state, env):
    state_index = env.nodes.index(state)
    state_tensor = torch.zeros(env.num_nodes, device=device)
    state_tensor[state_index] = 1
    return state_tensor

def evaluate(model_path=MODEL_PATH):
    print(f"Loading environment from {GRAPH_PATH}...")
    graph_data = load_json_graph(GRAPH_PATH)
    env = GraphEnvironment(graph_data)
    
    input_dim = env.num_nodes
    output_dim = env.num_nodes

    print(f"Loading model from {model_path}...")
    try:
        model = DQNetwork(input_dim, output_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Model file not found. Please train a model first using train_attacker.py")
        return

    print("\n--- Starting Evaluation ---")
    state = env.reset()
    print(f"Initial State: {env.graph.nodes[str(state)]['name']}")
    
    total_reward = 0
    path = [env.graph.nodes[str(state)]['name']]
    
    for t in range(50):
        actions = env.get_actions(state)
        if not actions:
            print("No more actions available.")
            break

        with torch.no_grad():
            state_tensor = state_to_tensor(state, env).unsqueeze(0)
            q_values = model(state_tensor)
            
            # Mask invalid actions
            valid_indices = [env.nodes.index(a) for a in actions]
            q_values_valid = q_values[0, valid_indices]
            best_action_local_idx = torch.argmax(q_values_valid).item()
            action = actions[best_action_local_idx]

        next_state, reward, alerts, done = env.step(action)
        
        node_name = env.graph.nodes[str(next_state)]['name']
        print(f"Step {t+1}: Moved to {node_name} | Reward: {reward}")
        if alerts:
            print(f"  -> Alerts Generated: {len(alerts)}")

        total_reward += reward
        path.append(node_name)
        state = next_state

        if done:
            print("Goal Reached!")
            break
            
    print("\n--- Evaluation Summary ---")
    print(f"Total Reward: {total_reward}")
    print(f"Path Length: {len(path)}")
    print(f"Path: {' -> '.join(path)}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    evaluate(path)
