"""
Gymnasium-compatible wrappers for GraphEnvironment.

NetworkAttackEnv  — attacker perspective (Discrete action space, one-hot obs).
NetworkDefenderEnv — defender perspective (MultiDiscrete action space, flat node-feature obs).
"""

import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.graph_env import GraphEnvironment, NODE_FEAT_DIM


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


class NetworkAttackEnv(gym.Env):
    """
    Single-agent Gym wrapper — attacker perspective.

    Observation
    -----------
    One-hot vector of length N (number of graph nodes) identifying the
    attacker's current position.

    Action
    ------
    Discrete(N): index of the target node to move to.  Invalid moves
    (non-edges, masked nodes) return a large negative reward and terminate
    the episode.

    Parameters
    ----------
    graph_source : str | dict
        Path to a JSON attack-graph file **or** an already-loaded dict.
    goal_node : str | int | None
        Target node ID the attacker is trying to reach.  If None the
        environment auto-detects the deepest Access node.
    config_path : str
        Path to config.json.
    max_steps : int
        Episode step budget.  Overrides config.json value when given.
    """

    metadata = {"render_modes": []}

    def __init__(self, graph_source, goal_node=None, config_path="config.json",
                 max_steps=None):
        super().__init__()

        if isinstance(graph_source, str):
            graph_json = _load_json(graph_source)
        else:
            graph_json = graph_source

        self._env = GraphEnvironment(graph_json, goal_node=goal_node,
                                     config_path=config_path)

        cfg = _load_json(config_path)
        self._max_steps = max_steps or cfg.get("attacker", {}).get("max_steps", 200)

        N = self._env.num_nodes
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(N,), dtype=np.float32)
        self.action_space = spaces.Discrete(N)

        self._step_count = 0

    def _obs(self):
        idx = self._env.node_to_idx.get(self._env.current_node, 0)
        obs = np.zeros(self._env.num_nodes, dtype=np.float32)
        obs[idx] = 1.0
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._env.reset_recon(episode=0)
        self._step_count = 0
        return self._obs(), {}

    def step(self, action):
        node_id = self._env.nodes[int(action)]
        next_node, reward, alerts, done = self._env.step(node_id)
        self._step_count += 1
        truncated = (self._step_count >= self._max_steps) and not done
        info = {"alerts": alerts, "current_node": next_node}
        return self._obs(), float(reward), done, truncated, info

    def valid_action_mask(self):
        """Boolean mask (length N) — True for reachable successor nodes."""
        valid = self._env.get_valid_actions(self._env.current_node)
        mask = np.zeros(self._env.num_nodes, dtype=bool)
        for v in valid:
            idx = self._env.node_to_idx.get(v)
            if idx is not None:
                mask[idx] = True
        return mask

    def render(self):
        pass

    def close(self):
        pass


class NetworkDefenderEnv(gym.Env):
    """
    Single-agent Gym wrapper — defender perspective.

    Observation
    -----------
    Flat numpy array of shape (N × NODE_FEAT_DIM,) derived from
    ``GraphEnvironment.get_graph_observation()``.

    Action
    ------
    MultiDiscrete([4, N]):
        - action_type ∈ {0=DoNothing, 1=NetworkFilter, 2=Patch, 3=RestoreConn}
        - target_node ∈ [0, N)  (ignored when action_type == 0)

    The defender operates against a *fixed* attacker; call
    ``set_attacker_episode()`` each episode to supply the attacker's move
    sequence, or leave it as None to use random attacker moves.

    Parameters
    ----------
    graph_source : str | dict
        Path to a JSON attack-graph file or an already-loaded dict.
    goal_node : str | int | None
        Node ID the defender is protecting.
    config_path : str
        Path to config.json.
    max_steps : int
        Defender episode step budget.
    attacker_model_path : str | None
        Optional path to a trained DQN attacker .pth file.  When None a
        random attacker is used.
    """

    metadata = {"render_modes": []}

    def __init__(self, graph_source, goal_node=None, config_path="config.json",
                 max_steps=None, attacker_model_path=None):
        super().__init__()

        if isinstance(graph_source, str):
            graph_json = _load_json(graph_source)
        else:
            graph_json = graph_source

        self._env = GraphEnvironment(graph_json, goal_node=goal_node,
                                     config_path=config_path)

        cfg = _load_json(config_path)
        self._max_steps = max_steps or cfg.get("defender", {}).get("max_steps", 10)

        N = self._env.num_nodes
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(N * NODE_FEAT_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([4, N])

        self._attacker = self._build_attacker(attacker_model_path, cfg)
        self._step_count = 0

    def _build_attacker(self, model_path, cfg):
        if model_path is None:
            from agents.attacker.random_attacker import RandomAttacker
            return RandomAttacker()
        from agents.attacker.dqn_network import DQNetwork
        import torch
        n = self._env.num_nodes
        net = DQNetwork(n, n)
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        net.eval()
        return net

    def _attacker_act(self):
        env = self._env
        valid = env.get_valid_actions(env.current_node)
        if not valid:
            return env.current_node
        import torch
        try:
            one_hot = torch.zeros(env.num_nodes)
            idx = env.node_to_idx.get(env.current_node, 0)
            one_hot[idx] = 1.0
            with torch.no_grad():
                q = self._attacker(one_hot.unsqueeze(0))
            valid_idxs = [env.node_to_idx[v] for v in valid if v in env.node_to_idx]
            best = max(valid_idxs, key=lambda i: q[0, i].item())
            return env.nodes[best]
        except Exception:
            import random
            return random.choice(valid)

    def _obs(self):
        x, _ = self._env.get_graph_observation()
        return x.numpy().reshape(-1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._env.reset_recon(episode=0)
        self._step_count = 0
        return self._obs(), {}

    def step(self, action):
        action_type = int(action[0])
        target_idx = int(action[1])
        target_node = self._env.nodes[target_idx]

        attacker_done = False
        attacker_node = self._env.current_node
        if hasattr(self._attacker, "act"):
            atk_action = self._attacker.act(self._env)
        else:
            atk_action = self._attacker_act()

        _, _, _, attacker_done = self._env.step(atk_action)

        from rewards.defender_reward import RewardModelPPO
        reward_model = RewardModelPPO()

        risk_before = self._env.graph.nodes.get(
            str(self._env.end_node), {}
        ).get("unconditional_risk", 0.0)

        self._env.apply_action(action_type, target_node=target_node)

        risk_after = self._env.graph.nodes.get(
            str(self._env.end_node), {}
        ).get("unconditional_risk", 0.0)

        self._env.compute_and_update_risk(decay_factor=0.2)

        self._step_count += 1
        terminated = attacker_done
        truncated = (self._step_count >= self._max_steps) and not terminated

        reward = reward_model.get_reward(
            action_type=action_type,
            risk_before=risk_before,
            risk_after=risk_after,
            attacker_reached_goal=attacker_done,
            steps_taken=self._step_count,
            steps_remaining=max(0, self._max_steps - self._step_count),
        )

        info = {
            "action_type": action_type,
            "target_node": target_node,
            "attacker_node": attacker_node,
            "attacker_reached_goal": attacker_done,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def valid_action_mask(self):
        """
        Returns a flat boolean mask of length (4 * N).

        The first 4 slots correspond to action types (always valid).
        The remaining slots are not used for masking here; mask on the
        full MultiDiscrete space should be factored per sub-space.
        """
        node_mask = self._env.get_valid_action_mask()
        type_mask = np.ones(4, dtype=bool)
        return type_mask, node_mask.numpy()

    def render(self):
        pass

    def close(self):
        pass
