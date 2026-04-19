"""
Train a defender agent (Actor-Critic or DQN) against a trained DQN attacker.

Usage
-----
Train with Actor-Critic (default):
    python scripts/train_defender.py --agent ac

Train with DQN defender:
    python scripts/train_defender.py --agent dqn

Specify a custom attack graph and goal node:
    python scripts/train_defender.py \\
        --agent ac \\
        --graph attack_graphs/ag.json \\
        --goal-node 42 \\
        --episodes 5000 \\
        --output policy-models/defender/ac_defender.pth
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from environment.graph_env import GraphEnvironment


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--agent', choices=['ac', 'dqn'], default='ac',
                   help='Defender architecture: ac (Actor-Critic) or dqn (DQN). Default: ac')
    p.add_argument('--graph', default='attack_graphs/ag.json',
                   help='Path to attack graph JSON. Default: attack_graphs/ag.json')
    p.add_argument('--goal-node', default=None,
                   help='Goal node ID to defend. If omitted, auto-detects deepest Access node.')
    p.add_argument('--episodes', type=int, default=None,
                   help='Number of training episodes. Defaults to config.json value.')
    p.add_argument('--attacker-model', default='policy-models/attacker/dqn_attacker.pth',
                   help='Path to trained DQN attacker checkpoint.')
    p.add_argument('--output', default=None,
                   help='Output path for the trained defender checkpoint.')
    p.add_argument('--config', default='config.json',
                   help='Path to config.json. Default: config.json')
    args = p.parse_args()

    with open(args.graph) as f:
        graph_data = json.load(f)

    print(f"Loading environment from: {args.graph}")
    env = GraphEnvironment(graph_data, goal_node=args.goal_node, config_path=args.config)
    print(f"Goal node: {env.end_node}  |  Nodes: {env.num_nodes}")

    with open(args.config) as f:
        config = json.load(f)

    if args.agent == 'ac':
        from agents.defender.ac_defender import AC_Def_Agent
        agent = AC_Def_Agent()
        num_episodes = args.episodes or config.get('defender', {}).get('num_episodes', 5000)
        output_path = args.output or 'policy-models/defender/ac_defender.pth'

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Training Actor-Critic defender for {num_episodes} episodes...")
        agent.train_agent(
            env,
            num_episodes=num_episodes,
            output_path=output_path,
            attacker_model_path=args.attacker_model,
        )

    elif args.agent == 'dqn':
        from agents.defender.dqn_defender import DQN_Def_Agent
        agent = DQN_Def_Agent()
        num_episodes = args.episodes or config.get('defender', {}).get('num_episodes', 5000)
        output_path = args.output or 'policy-models/defender/dqn_defender.pth'

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Training DQN defender for {num_episodes} episodes...")
        agent.train_agent(
            env,
            num_episodes=num_episodes,
            output_path=output_path,
            attacker_model_path=args.attacker_model,
        )

    print(f"\nTraining complete. Model saved to: {output_path}")


if __name__ == '__main__':
    main()
