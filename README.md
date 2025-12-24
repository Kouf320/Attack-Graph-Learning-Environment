# Attack Graph Reinforcement Learning Environment

## Project Overview
The environment simulates a network attack graph where nodes represent states (Reconnaissance, Vulnerabilities, Privileges, etc.) and edges represent possible transitions. The agent (Attacker) learns to navigate this graph to reach a target goal while generating realistic network alerts that can be used to train Defender systems.

## Key Features
- **Standalone Environment:** Easy to integrate into other projects without external dependencies from the original PhD repository.
- **Realistic Alert Generation:** Built-in `AlertGenerator` that creates Suricata-style logs (Recon, Local, Network, Noise) based on agent actions.
- **Configurable Rewards:** Uses a quantitative CVSS-based reward model for realistic scoring of exploit attempts.
- **Extensible:** Support for loading custom attack graphs (`ag.json`) and configuration (`config.json`).
- **Training & Evaluation:** Includes scripts to train new DQN models or evaluate existing ones.

---

## File Structure
```text
AttackGraph_Standalone/
├── environment.py         # The core GraphEnvironment Class
├── dqn_model.py           # PyTorch implementation of the DQN architecture
├── replay_memory.py       # Experience replay buffer for training
├── train_attacker.py      # Script to train a new attacker model
├── evaluate_attacker.py   # Script to run and visualize a trained model
├── config.json            # Environment and NVD API settings
├── ag.json                # The attack graph data (Assets and Associations)
├── requirements.txt       # Python dependencies
├── logs/                  # Environment and training logs
├── rewards/
│   └── RewardModel.py     # CVSS-based reward logic
└── utils/
    ├── Alert.py           # Suricata-style alert generator
    └── Colors.py          # Terminal styling utilities
```

---

## Installation

1. **Clone or Copy** this directory to your local workspace.
2. **Install Dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Use

### 1. Training a New Attacker
To train a new DQN agent from scratch using the current `ag.json` graph:
```bash
python train_attacker.py
```
- This will run 500 episodes (configurable in the script).
- The best model will be saved as `attacker_model.pth`.
- Training progress will be printed to the console.

### 2. Evaluating a Trained Model
To see a trained attacker in action and view the path it takes through the graph:
```bash
python evaluate_attacker.py attacker_model.pth
```
- The script will output the sequence of nodes visited.
- It will also display rewards earned and the number of alerts generated at each step.

### 3. Integrating with Other Agents
You can use the `GraphEnvironment` in your own scripts as follows:
```python
from environment import GraphEnvironment
import json

# Load graph
with open('ag.json', 'r') as f:
    graph_data = json.load(f)

# Initialize Env
env = GraphEnvironment(graph_data)
state = env.reset()

# Step through
action = env.get_actions(state)[0]
next_state, reward, alerts, done = env.step(action)
```

---

## Customization
- **Change the Graph:** Replace `ag.json` with your own attack graph generated from tools like MulVAL or custom scripts, provided they follow the same JSON schema.
- **Reward Logic:** Modify `rewards/RewardModel.py` to change how the agent is incentivized (e.g., higher penalties for detection).
- **Alert Frequency:** Adjust `noise_prob` and `detection_prob` in `environment.py` or the `ag.json` node attributes to control the volume of IDS logs.

## Credits
This standalone environment is derived from the **DQN-AttackGraphs-Phd** project, designed for cybersecurity research involving Deep Reinforcement Learning.
