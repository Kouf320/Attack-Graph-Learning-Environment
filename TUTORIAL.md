# RL Network Defense — End-to-End Tutorial

This tutorial walks you from installation through training and evaluating both an attacker and a defender on your own attack graph. All code examples assume you are at the repository root.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Attack Graph Format](#2-attack-graph-format)
3. [Loading the Environment](#3-loading-the-environment)
4. [Using the Environment as a Gymnasium Gym](#4-using-the-environment-as-a-gymnasium-gym)
5. [Training the Attacker (DQN)](#5-training-the-attacker-dqn)
6. [Training a Defender — Actor-Critic](#6-training-a-defender--actor-critic)
7. [Training a Defender — DQN](#7-training-a-defender--dqn)
8. [Evaluating Defenders Across Graph Variants](#8-evaluating-defenders-across-graph-variants)
9. [Baselines](#9-baselines)
10. [Key Concepts and Maths](#10-key-concepts-and-maths)

---

## 1. Installation

```bash
git clone <your-repo-url>
cd rl-network-defense

pip install -r requirements.txt

# PyTorch Geometric needs a CUDA-matched build
pip install torch-geometric
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
```

If you want to use the NVD API to pull live CVSSv3 scores, add your key to `config.json`:

```json
"NVD_KEY": "your-nvd-api-key"
```

Without a key the environment falls back to a local SQLite cache (`database/vulnerability-remediation-database.db`).

---

## 2. Attack Graph Format

The environment reads attack graphs produced by MAL (Meta Attack Language) exporters. The JSON schema has two top-level keys:

```json
{
  "assets": {
    "<node_id>": {
      "name": "Reconnaissance:192.168.1.10",
      "metaconcept": "Reconnaissance",
      "eid": "<node_id>"
    },
    ...
  },
  "associations": [
    {
      "association": {
        "sourceAssets": ["<source_id>"],
        "targetAssets": ["<target_id>"],
        "cve": "CVE-2021-44228"
      }
    },
    ...
  ]
}
```

Each `asset` is a graph node (attack step). Each `association` is a directed edge. Edge weights are derived from CVSSv3 exploitability and CIA impact scores:

```
R_exploit = AV · AC · PR · UI · NORM_F_ATTEMPT
R_impact  = (1 − C') · (1 − I') · (1 − A') · NORM_F
```

where `NORM_F_ATTEMPT = 21.147` and `NORM_F = 17.857` normalise the products to a common range.

The **goal node** is the asset the attacker aims to compromise and the defender aims to protect. It is typically the deepest `Access` metaconcept node in the graph.

---

## 3. Loading the Environment

### Auto-detect goal node

```python
import json
from environment.graph_env import GraphEnvironment

with open("attack_graphs/ag.json") as f:
    graph_data = json.load(f)

# goal_node=None → auto-detects the deepest Access node
env = GraphEnvironment(graph_data)
print(f"Goal node: {env.end_node}")
print(f"Graph size: {env.num_nodes} nodes")
```

### Specify your own goal node

```python
env = GraphEnvironment(graph_data, goal_node="42")
```

Pass any node ID that exists in the graph. A `ValueError` is raised immediately if the ID is not found, so mistakes are caught at load time.

### Use a different config file

```python
env = GraphEnvironment(graph_data, goal_node="42", config_path="my_config.json")
```

### Inspect the graph

```python
import networkx as nx

# NetworkX DiGraph
G = env.graph

# Node metadata
for node, data in list(G.nodes(data=True))[:3]:
    print(node, data['name'])

# Edge weights (CVSS-derived distance)
for u, v, data in list(G.edges(data=True))[:3]:
    print(u, "→", v, "distance:", data.get('distance'))
```

---

## 4. Using the Environment as a Gymnasium Gym

Two thin wrappers expose the environment through the standard Gymnasium interface.

### Attacker wrapper — `NetworkAttackEnv`

```python
from environment.gym_env import NetworkAttackEnv

env = NetworkAttackEnv(
    graph_source="attack_graphs/ag.json",
    goal_node=None,       # auto-detect
    max_steps=200,
)

obs, info = env.reset(seed=0)
print("Observation shape:", obs.shape)   # (N,) one-hot over graph nodes
print("Action space:", env.action_space) # Discrete(N)

done = False
while not done:
    mask = env.valid_action_mask()       # boolean array, True = reachable
    valid_actions = mask.nonzero()[0]
    action = int(valid_actions[0]) if len(valid_actions) > 0 else 0
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

The observation is a one-hot vector of length N (number of nodes). The action is an integer index into `env.env.nodes` — the environment translates it to a node ID internally. Invalid moves (non-edge, masked node) return a large penalty and terminate the episode.

### Defender wrapper — `NetworkDefenderEnv`

```python
from environment.gym_env import NetworkDefenderEnv

env = NetworkDefenderEnv(
    graph_source="attack_graphs/ag.json",
    goal_node=None,
    max_steps=10,
    attacker_model_path="policy-models/attacker/dqn_attacker.pth",  # or None for random
)

obs, info = env.reset(seed=0)
print("Observation shape:", obs.shape)   # (N × 17,) flat node features
print("Action space:", env.action_space) # MultiDiscrete([4, N])

done = False
while not done:
    type_mask, node_mask = env.valid_action_mask()
    action_type = 1   # Network filtering
    target_node  = int(node_mask.nonzero()[0][0])
    obs, reward, terminated, truncated, info = env.step([action_type, target_node])
    done = terminated or truncated
    print(f"Reward: {reward:.3f} | Attacker reached goal: {info['attacker_reached_goal']}")

env.close()
```

The defender's observation is the flattened node-feature matrix (shape `N × 17`). The action is a pair `[action_type, target_node_index]`.

**Defender action types:**

| ID | Name | Effect |
|----|------|--------|
| 0 | Do Nothing | No structural change |
| 1 | Network Filtering | Mask all outgoing edges from target node |
| 2 | Restore Software (Patch) | Mask target CVE node |
| 3 | Restore Connection | Unmask the last removed edge |

---

## 5. Training the Attacker (DQN)

### Via script (recommended)

```bash
python scripts/train_attacker.py
```

This reads graph and training parameters from `config.json` and saves the checkpoint to `policy-models/attacker/dqn_attacker.pth`.

### Via Python API

```python
import json
from environment.graph_env import GraphEnvironment
from agents.attacker.dqn_network import DQNetwork
from memory.replay_buffer import ReplayMemory
import torch, random
import torch.nn.functional as F

with open("attack_graphs/ag.json") as f:
    graph_data = json.load(f)

env = GraphEnvironment(graph_data, goal_node=None)
N = env.num_nodes

policy_net = DQNetwork(N, N)
target_net = DQNetwork(N, N)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
memory = ReplayMemory(100_000)

GAMMA, TAU, BATCH = 0.99, 0.005, 128
epsilon = 1.0

for episode in range(10_000):
    env.reset_recon(episode)
    state_idx = env.node_to_idx[env.current_node]

    for t in range(200):
        state_t = torch.zeros(N); state_t[state_idx] = 1.0

        valid = env.get_valid_actions(env.current_node)
        if not valid:
            break

        if random.random() < epsilon:
            next_node = random.choice(valid)
        else:
            with torch.no_grad():
                q = policy_net(state_t.unsqueeze(0))[0]
            valid_idxs = [env.node_to_idx[v] for v in valid]
            next_node = env.nodes[max(valid_idxs, key=lambda i: q[i].item())]

        next_node_id, reward, _, done = env.step(next_node)
        next_idx = env.node_to_idx.get(next_node_id, state_idx)

        memory.push(state_idx, env.node_to_idx[next_node], reward, next_idx, done)
        state_idx = next_idx

        if len(memory) >= BATCH:
            batch = memory.sample(BATCH)
            # ... Double DQN update (see agents/attacker/dqn_network.py) ...

        if done:
            break

    epsilon = max(0.05, epsilon * (1 - 1 / 2000))

    # Soft-update target network: θ' ← τ·θ + (1−τ)·θ'
    for p, pt in zip(policy_net.parameters(), target_net.parameters()):
        pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)

torch.save(policy_net.state_dict(), "policy-models/attacker/dqn_attacker.pth")
```

**Architecture:** 4-layer MLP with 128 hidden units and ReLU activations. The input is a one-hot vector over nodes; the output is a Q-value per node. Action masking is applied before argmax so the agent never selects an unreachable node.

**Double DQN update:**

```
y_t = r_t + γ · Q_target(s_{t+1}, argmax_a Q_online(s_{t+1}, a))
L   = MSE(Q_online(s_t, a_t), y_t)
```

Exploration follows ε-greedy with exponential decay: ε(k) = ε_end + (ε_start − ε_end) · exp(−k / ε_decay).

---

## 6. Training a Defender — Actor-Critic

### Via script (recommended)

```bash
python scripts/train_defender.py \
    --agent ac \
    --graph attack_graphs/ag.json \
    --episodes 5000 \
    --attacker-model policy-models/attacker/dqn_attacker.pth \
    --output policy-models/defender/ac_defender.pth
```

Specify `--goal-node <ID>` to protect a custom node instead of the auto-detected one.

### Via Python API

```python
import json
from environment.graph_env import GraphEnvironment
from agents.defender.ac_defender import AC_Def_Agent

with open("attack_graphs/ag.json") as f:
    graph_data = json.load(f)

env = GraphEnvironment(graph_data, goal_node=None)
agent = AC_Def_Agent()

agent.train_agent(
    env,
    num_episodes=5000,
    output_path="policy-models/defender/ac_defender.pth",
    attacker_model_path="policy-models/attacker/dqn_attacker.pth",
)
```

**Architecture (GAT Actor-Critic):**

```
Input: x ∈ R^{N×17}, edge_index
  → GATConv(17, 64, heads=4) + BatchNorm  →  256-dim per node
  → GATConv(256, 256, heads=1) + BatchNorm →  256-dim per node
  → global_mean_pool  →  R^256
  → shared_fc(256, 256)
  ↗ actor_node_score(N, 1) — node selection scores (masked softmax)
  ↗ actor_type(512, 4)     — action type distribution
  ↗ critic(256, 1)          — V(s) baseline
```

The policy is factored: first a node is sampled from a softmax over node scores (masked to only alerted/active nodes), then an action type is sampled conditioned on that node. This keeps the joint action space tractable even for large graphs.

**Defender reward (PBRS):**

Potential-Based Reward Shaping anchors on the unconditional compromise risk of the goal node Φ(s) = −P_risk(goal):

```
r_step = R_availability(a) + bonus_risk_targeting + λ·(Φ(s) − γ·Φ(s'))
       = R_availability(a) + bonus_risk_targeting + λ·(P_risk_before − γ·P_risk_after)
```

Terminal rewards:
- Defense success: `+50 · (1 + 0.1 · steps_remaining) · (α + (1−α) · precision)`
- Attacker goal reached: `−50 · (1 + 0.1 · steps_taken)`

PBRS is policy-invariant: the optimal policy is unchanged by adding Φ(s') − γΦ(s) to any reward function, so it accelerates learning without biasing the final policy.

---

## 7. Training a Defender — DQN

```bash
python scripts/train_defender.py \
    --agent dqn \
    --graph attack_graphs/ag.json \
    --episodes 5000 \
    --attacker-model policy-models/attacker/dqn_attacker.pth \
    --output policy-models/defender/dqn_defender.pth
```

### Via Python API

```python
import json
from environment.graph_env import GraphEnvironment
from agents.defender.dqn_defender import DQN_Def_Agent

with open("attack_graphs/ag.json") as f:
    graph_data = json.load(f)

env = GraphEnvironment(graph_data)
agent = DQN_Def_Agent()

agent.train_agent(
    env,
    num_episodes=5000,
    output_path="policy-models/defender/dqn_defender.pth",
    attacker_model_path="policy-models/attacker/dqn_attacker.pth",
)
```

The DQN defender uses the same GAT encoder but replaces the actor-critic heads with a single Q-network head that outputs one scalar per `(action_type, target_node)` pair. Experience replay and a target network (soft updates, τ = 0.005) provide stable training.

---

## 8. Evaluating Defenders Across Graph Variants

The `attack_graphs/variants/` directory contains 1,000 structurally diverse attack graph variants for zero-shot generalisation evaluation. None of these graphs are used during training.

```bash
python scripts/evaluate_defenders.py \
    --variants attack_graphs/variants/ \
    --ac-model  policy-models/defender/ac_defender.pth \
    --dqn-model policy-models/defender/dqn_defender.pth \
    --limit 100 \
    --out evaluation_out/
```

Remove `--limit` to evaluate all 1,000 variants. Results are written to:

- `evaluation_out/eval_ac.csv` — per-graph metrics for AC
- `evaluation_out/eval_dqn.csv` — per-graph metrics for DQN
- `evaluation_out/evaluation_report.md` — aggregate comparison table

**Metrics:**

| Metric | Description |
|--------|-------------|
| Win rate | Fraction of episodes where the attacker failed to reach the goal |
| Precision | Of unique blocking actions (Filter/Patch), the share that landed on a node visited by the attacker |
| Validity | Of non-idle action attempts, the share that produced a structural change in the graph |
| Reward | Cumulative episode reward under the PBRS reward function |

---

## 9. Baselines

Two baselines are included for comparison:

### Random defender

```python
from agents.baselines.random_defender import RandomDefender
from environment.graph_env import GraphEnvironment
import json

with open("attack_graphs/ag.json") as f:
    graph_data = json.load(f)

env = GraphEnvironment(graph_data)
defender = RandomDefender()

env.reset_recon(0)
for _ in range(10):
    x, ei = env.get_graph_observation()
    mask = env.get_valid_action_mask()
    a_type, target_idx = defender.act(x, ei, mask)
    target_node = env.node_list[target_idx]
    env.apply_action(a_type, target_node=target_node)
```

### Rule-based defender

The rule-based defender implements a priority heuristic: prefer patching CVE nodes on the shortest attack path, fall back to network filtering on high-risk nodes.

```python
from agents.baselines.rule_based_defender import RuleBasedDefender
defender = RuleBasedDefender()
```

---

## 10. Key Concepts and Maths

### Risk propagation

Node unconditional compromise probability is computed via iterative noisy-OR forward pass over the attack graph:

```
P(v) = 1 − ∏_{u ∈ parents(v)} (1 − P(u) · exp(−k · dist(u, v)))
```

The iteration converges when ‖P^{t+1} − P^t‖_∞ < ε = 10⁻⁶. This models the probability that at least one parent attack step propagates compromise to v, attenuated by topological distance. The exponential decay term exp(−k · d(u, v)) penalises longer attack chains.

### Node feature vector (17-dim)

Each node in the defender's observation carries:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | active | {0, 1} | 1 if node is not patched/filtered |
| 1 | risk | [0, 1] | Unconditional compromise probability |
| 2 | centrality | [0, 1] | Betweenness centrality |
| 3 | vuln | {0, 1} | 1 if node is a CVE exploit step |
| 4 | critical | {0, 1} | 1 if node is the goal |
| 5 | z_score | [−1, 1] | Normalised severity z-score for node's IP |
| 6 | entropy | [0, 1] | Shannon entropy of alert severity distribution |
| 7 | alerted | {0, 1} | 1 if node has an active alert |
| 8 | alert_vol_src | [0, 1] | Normalised source alert volume |
| 9 | alert_vol_dst | [0, 1] | Normalised destination alert volume |
| 10 | cum_alerts | [0, ∞) | Cumulative per-node alert count / step |
| 11 | alert_recency | (0, 1] | 1 / (steps_since_last_alert + 1) |
| 12 | goal_dist | [0, 1] | Topological hop distance to goal, normalised |
| 13 | max_cvss | [0, 1] | Max CVSS on incoming edges, normalised |
| 14 | def_filtered | {0, 1} | Defender action flag: filtered |
| 15 | def_patched | {0, 1} | Defender action flag: patched |
| 16 | def_restored | {0, 1} | Defender action flag: restored |

### Stackelberg game structure

The attacker-defender interaction is modelled as a Stackelberg security game. The defender (leader) commits first to a monitoring and response policy; the attacker (follower) plays the best response. In the simultaneous training setting used here this is approximated as a Nash equilibrium via independent self-play: each agent trains against the current policy of the other.

### GAT message passing

Each Graph Attention Network layer computes:

```
h_i^{(l+1)} = σ( ∑_{j ∈ N(i)} α_{ij} · W · h_j^{(l)} )

α_{ij} = softmax_j( LeakyReLU( a^T [W·h_i ‖ W·h_j] ) )
```

where `‖` is concatenation, `a` is a learnable attention vector, and `W` is the weight matrix. Multi-head attention (4 heads in layer 1) captures heterogeneous structural roles across the graph.

### PPO-clip objective (reference)

Although PPO defenders are not included in this release, the Actor-Critic defender is trained with a compatible advantage-based objective using Generalised Advantage Estimation (GAE, λ = 0.95):

```
Â_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)

L^{ACTOR} = −E_t[ log π(a_t|s_t) · Â_t ]
L^{CRITIC} = E_t[ (V(s_t) − V_t^{target})² ]
L^{ENT}   = −E_t[ H(π(·|s_t)) ]

L(θ) = L^{ACTOR} + c₁·L^{CRITIC} − c₂·L^{ENT}
```

Entropy regularisation with coefficient c₂ = 0.01 prevents premature policy collapse to deterministic strategies.
