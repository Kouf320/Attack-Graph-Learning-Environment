# Attack-Graph-Learning-Environment

A reinforcement learning framework for modeling adversarial cyber-attack and defense interactions on enterprise network attack graphs. The environment encodes a network topology as a directed graph derived from an attack graph model (e.g., MAL — Meta Attack Language), where edge weights are derived directly from CVSSv3 exploitability and impact metrics.

## Overview

The framework implements a Stackelberg game between two RL agents:

- **Attacker (DQN)**: Navigates the attack graph from a reconnaissance entry node to a target asset, maximising cumulative CVSS-weighted reward.
- **Defender (GAT-AC / GAT-DQN)**: Observes Suricata-like alert streams processed through a Graph Attention Network (GAT) and applies structural countermeasures (network filtering, software patching, host isolation).

The environment is fully compatible with the **Gymnasium** interface (`gym.Env`), supporting both `reset(seed, options)` and `step(action)` with standard observation and action spaces.

**Goal nodes are configurable**: pass any node ID when constructing the environment, or let it auto-detect the deepest `Access` metaconcept node.

## Project Structure

```
rl-network-defense/
├── environment/
│   ├── graph_env.py          # GraphEnvironment — core MDP with CVSS-weighted graph
│   └── gym_env.py            # Gymnasium wrappers: NetworkAttackEnv, NetworkDefenderEnv
│
├── agents/
│   ├── attacker/
│   │   ├── dqn_network.py    # DQN and DQN_Def network architectures
│   │   └── random_attacker.py
│   ├── defender/
│   │   ├── ac_defender.py    # GAT Actor-Critic defender
│   │   └── dqn_defender.py   # GAT DQN defender
│   └── baselines/
│       ├── random_defender.py
│       └── rule_based_defender.py
│
├── memory/
│   └── replay_buffer.py      # ReplayMemory and ReplayBuffer (experience replay)
│
├── rewards/
│   ├── attacker_reward.py    # DefaultRewardModel — CVSS-based attacker utility
│   └── defender_reward.py    # RewardModelPPO — PBRS + availability cost shaping
│
├── utils/
│   ├── alert_generator.py    # Suricata-format synthetic alert generator
│   ├── colors.py             # Terminal colour helpers
│   ├── metrics.py            # StreamingEntropy, bin_risk_score
│   └── helpers.py            # RunningMeanStd, update_attack_path_with_uncertainty
│
├── attack_graphs/
│   ├── ag.json               # Main training attack graph
│   └── variants/             # 1000 generated attack graph variants for zero-shot eval
│       ├── generation_log.json
│       └── variation_XXXX.json
│
├── scripts/
│   ├── train_attacker.py     # Train DQN attacker on attack graph
│   ├── train_defender.py     # Train AC or DQN defender (--agent ac|dqn)
│   ├── evaluate_attacker.py  # Run greedy attacker evaluation
│   └── evaluate_defenders.py # Cross-graph zero-shot defender evaluation
│
├── database/
│   └── vulnerability-remediation-database.db  # Local NVD CVSSv3 SQLite cache
│
├── policy-models/
│   ├── attacker/             # Saved attacker checkpoints (.pth)
│   └── defender/             # Saved defender checkpoints (.pth)
│
├── logs/
├── config.json
├── requirements.txt
└── .gitignore
```

## Environment

### GraphEnvironment

The core MDP. Initialised as `GraphEnvironment(graph_json, goal_node=None, config_path='config.json')`. Pass any node ID for `goal_node`, or leave it as `None` to auto-detect the deepest `Access` metaconcept node. State space is the attack graph G = (V, E) where:

- Each node v ∈ V represents an attack step (Reconnaissance, CVE exploit, Privilege escalation, Host compromise, etc.)
- Each edge (u, v) ∈ E carries a weight derived from the CVSSv3 vector of the associated vulnerability.

**Attacker reward** for traversing edge (u → v) involving CVE c:

```
R_exploit(c) = AV · AC · PR · UI · NORM_F_ATTEMPT    (exploitability component)
R_impact(c)  = (1-C') · (1-I') · (1-A') · NORM_F    (CIA impact component)
```

where AV, AC, PR, UI, C', I', A' are the normalised CVSSv3 metric values.

**Observation** (per-node features, dim = 17):

| # | Feature | Description |
|---|---------|-------------|
| 0 | active | Node mask (1 = active, 0 = patched/removed) |
| 1 | risk | Unconditional compromise probability (noisy-OR forward propagation) |
| 2 | centrality | Betweenness centrality |
| 3 | vuln | Binary: 1 if CVE node |
| 4 | critical | Binary: 1 if goal node |
| 5 | z_score | Normalised z-score of severity stream for node's IP |
| 6 | entropy | Shannon entropy of alert severity distribution |
| 7 | alerted | Binary: 1 if node has active alert |
| 8 | alert_vol_src | Normalised source alert volume |
| 9 | alert_vol_dst | Normalised destination alert volume |
| 10 | cum_alerts | Cumulative per-node alert count / step |
| 11 | alert_recency | 1/(steps_since_last_alert + 1) |
| 12 | goal_dist | Topological hop distance to goal, normalised |
| 13 | max_cvss | Max CVSS on incoming edges, normalised |
| 14 | def_filtered | Defender action flag: filtered |
| 15 | def_patched | Defender action flag: patched |
| 16 | def_restored | Defender action flag: restored |

**Risk propagation** uses iterative noisy-OR forward pass:

```
P(v) = 1 - ∏_{u ∈ parents(v)} (1 - P(u) · exp(-k · dist(u,v)))
```

until convergence with tolerance ε = 1e-6.

### Defender Actions

| ID | Action | Effect |
|----|--------|--------|
| 0 | Do Nothing | No structural change |
| 1 | Network Filtering | Mask all outgoing edges from target node |
| 2 | Restore Software (Patch) | Mask target CVE node |
| 3 | Restore Connection | Unmask last removed edge |

### Gymnasium Wrappers

`environment/gym_env.py` provides two standard `gymnasium.Env` wrappers:

- **`NetworkAttackEnv`** — attacker perspective. `Discrete(N)` action space, one-hot observation of length N.
- **`NetworkDefenderEnv`** — defender perspective. `MultiDiscrete([4, N])` action space, flat node-feature observation of shape `(N × 17,)`.

Both expose a `valid_action_mask()` method for action masking with frameworks such as Stable Baselines 3 with MaskablePPO or RLlib.

## Agent Architectures

### DQN Attacker

A 4-layer MLP (128 hidden units, ReLU activations) with a one-hot state encoding over graph nodes. Trained with Double DQN (target network with soft updates, τ = 0.005), action masking, and ε-greedy exploration with exponential decay.

### GAT Actor-Critic Defender

```
Input: node features x ∈ R^{N×17}, edge_index
  → GATConv(17, 64, heads=4) + BatchNorm → 256-dim
  → GATConv(256, 256, heads=1) + BatchNorm → 256-dim
  → global_mean_pool → R^256
  → shared_fc(256, 256)
  ↗ actor_type(512, 4)       — action type distribution
  ↗ actor_node_score(N, 1)   — node selection score (masked softmax)
  ↗ critic(256, 1)            — V(s) estimate
```

Training uses advantage-based policy gradient with GAE(λ=0.95):

```
Â_t = ∑_{k=0}^{∞} (γλ)^k δ_{t+k},  δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
L(θ) = L^{ACTOR} + c₁·L^{CRITIC} − c₂·L^{ENT}
```

### GAT DQN Defender

Same GAT encoder as the Actor-Critic, with a single Q-network head outputting one scalar per `(action_type, target_node)` pair. Trained with Double DQN, experience replay, and soft target-network updates (τ = 0.005).

### Defender Reward (PBRS)

The reward uses potential-based reward shaping (PBRS) anchored on goal-node unconditional risk Φ(s) = −P_risk(goal):

```
r_step = R_availability(action) + bonus_risk_targeting + PBRS
PBRS   = λ · (Φ(s) − γ · Φ(s')) = λ · (P_risk_before − γ · P_risk_after)
```

Terminal rewards:
- **Defense success**: `+50 · (1 + 0.1 · steps_remaining) · (α + (1-α) · precision)`
- **Attacker goal reached**: `−50 · (1 + 0.1 · steps_taken)`

## Installation

```bash
git clone <repo-url>
cd rl-network-defense
pip install -r requirements.txt
```

PyTorch Geometric requires a matching CUDA build:

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
```

Set your NVD API key in `config.json`:

```json
"NVD_KEY": "YOUR_KEY_HERE"
```

## Usage

### Train the Attacker

```bash
python scripts/train_attacker.py
```

### Train the Actor-Critic Defender

```bash
python scripts/train_defender.py --agent ac
```

Or with a custom goal node:

```python
import json
from environment.graph_env import GraphEnvironment
from agents.defender.ac_defender import AC_Def_Agent

with open("attack_graphs/ag.json") as f:
    graph_data = json.load(f)

env = GraphEnvironment(graph_data, goal_node="42")  # protect node 42
agent = AC_Def_Agent()
agent.train_agent(env, num_episodes=5000,
                  output_path="policy-models/defender/ac_defender.pth",
                  attacker_model_path="policy-models/attacker/dqn_attacker.pth")
```

### Train the DQN Defender

```bash
python scripts/train_defender.py --agent dqn
```

### Zero-Shot Evaluation Across Attack Graph Variants

```bash
python scripts/evaluate_defenders.py \
    --variants attack_graphs/variants/ \
    --ac-model  policy-models/defender/ac_defender.pth \
    --dqn-model policy-models/defender/dqn_defender.pth
```

## Attack Graph Variants

The `attack_graphs/variants/` directory contains 1,000 structurally diverse attack graph variants used for zero-shot generalisation evaluation. Variants are generated by randomising the topology while preserving the core asset structure. The `generation_log.json` records the generation parameters for each variant. The goal node is auto-detected per variant as the highest-ID `Access` node.

## Configuration

All hyperparameters are centralised in `config.json`. Key sections:

- `environment` — CVSS normalisation factors, NVD key, database path
- `attacker` — DQN training hyperparameters
- `defender` — GAT-PPO training hyperparameters
- `replay_training` — Alert-replay specific settings
- `paths` — Model and data file paths

## License

See `LICENSE`.
